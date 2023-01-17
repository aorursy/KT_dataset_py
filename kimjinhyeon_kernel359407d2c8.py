# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python



# Input data files are available in the read-only "../input/" directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import csv

import re



import torch

from tqdm.auto import tqdm
import os

import time



from transformers import BertForQuestionAnswering



model_name = 'bert-base-uncased'

pretrained_path = f'../input/pretrainedbert/{model_name}'

model = BertForQuestionAnswering.from_pretrained(pretrained_path).cuda()



timestamp = int(time.time())

checkpoint = f'checkpoint/{timestamp}'

os.makedirs(checkpoint, exist_ok=True)



data_dir = '../input/tweet-sentiment-extraction'
NUM_EPOCH = 4

BATCH_SIZE = 32

LEARNING_RATE = 5e-5
from transformers import BertTokenizer



special_tokens = ['<POS>', '<NEG>', '<NEU>', '<LINK>', '<MENTION>']



tokenizer = BertTokenizer.from_pretrained(pretrained_path)

tokenizer.add_tokens(special_tokens)



model.resize_token_embeddings(len(tokenizer))

tokenizer.tokenize('<NEU>')
link_pattern = r'http.*?(?=[\s)]|$)'

mention_pattern = r'@\w+(?!@)'

broken_pattern = 'ï¿½'





def normalize(string):

    string = string.replace(broken_pattern, '`')

    string = re.sub(link_pattern, '<LINK>', string)

    string = re.sub(mention_pattern, '<MENTION>', string)

    return string
def unnormalize(predicted, original):

    predicted = re.sub(r"n['`]t\b", r" not ", predicted)

    predicted = re.sub(r'(\W)', r' \1 ', predicted)

    pattern = re.escape(predicted)

    pattern = pattern.replace(r"\ not\ ", r"\ n('|`|o)t\ ")

    pattern = pattern.replace('`', f'(?:`|{broken_pattern})')

    pattern = pattern.replace(r'<\ link\ >', link_pattern).replace(r'<\ mention\ >', mention_pattern)

    pattern = re.sub(r'(\\ )+', r'\\s*', pattern)

    match = re.search(pattern, original, re.IGNORECASE)

    return match and match[0].strip()
def _trim(token):

    return token[2:] if token.startswith('##') else token



def subfinder(haystack, needle):

    if not needle:

        return

    length = len(needle)

    for i, token in enumerate(haystack[:len(haystack)-length+1]):

        if _trim(token) == _trim(needle[0]) and haystack[i+1:i + length] == needle[1:]:

            return i, i + length - 1
sentiment_table = {'positive': '<POS>', 'negative': '<NEG>', 'neutral': '<NEU>'}



def _tokenize(s):

    return tokenizer.tokenize(normalize(s))



def prepare_input_token(text, selected, sentiment):

    selected_tokens = _tokenize(selected)

    if selected:

        before, after = text.split(selected, 1)

        text = _tokenize(before) + selected_tokens + _tokenize(after)

    else:

        text = _tokenize(text)

    sentiment = sentiment_table[sentiment]

    return [sentiment] + text, selected_tokens
MAX_LENGTH = 128
def load_train_data():

    with open(f'{data_dir}/train.csv', 'r', newline='') as f:

        reader = csv.reader(f)

        print(next(reader))



        for textid, text, selected, sentiment in reader:

            tokens, selected_tokens = prepare_input_token(text, selected, sentiment)

            span = subfinder(tokens, selected_tokens)

            if not span:

                continue  # discard



            ids = tokenizer.encode(tokens,

                                   pad_to_max_length=True,

                                   max_length=MAX_LENGTH,

                                   truncation_strategy='do_not_truncate')

            yield torch.tensor(ids), torch.tensor(span) + 1  # [BOS]



train_data = list(load_train_data())
import random



random.shuffle(train_data)



total = len(train_data)

dev_size = total // 21

dev_data = train_data[:dev_size]

train_data = train_data[dev_size:]

total, total - dev_size, dev_size
from torch.utils.data import DataLoader



train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=False)
import torch.nn.functional as F



def token_jaccard(spans):

    '''spans: a Tensor of shape [num_spans, batch_size, 2]'''

    earlier_starts, later_starts = spans[..., 0].min(dim=0)[0], spans[..., 0].max(dim=0)[0]

    earlier_ends, later_ends = spans[..., 1].min(dim=0)[0], spans[..., 1].max(dim=0)[0]

    intersections = F.relu(earlier_ends - later_starts + 1).float()

    unions = later_ends - earlier_starts + 1

    return (intersections / unions).mean()
from transformers import AdamW



optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

_criterion = torch.nn.CrossEntropyLoss()



def calculate_loss(logits, spans):

    _, _, hidden_size = logits.shape

    return _criterion(logits.view(-1, hidden_size), spans.t().flatten())
def predict(model, input_ids):

    pad_mask = (input_ids == 0)

    logits = model(input_ids)

    logits = torch.stack(logits, dim=0)  # [2, B, T]

    logits[:, pad_mask] = float('-inf')

    return logits
def train_step(batch):

    input_ids, spans = (item.cuda() for item in batch)

    optimizer.zero_grad()

    logits = predict(model, input_ids)

    loss = calculate_loss(logits, spans)

    loss.backward()

    optimizer.step()

    return loss.item()



def eval_step(batch):

    input_ids, spans = (item.cuda() for item in batch)

    logits = predict(model, input_ids)

    loss = calculate_loss(logits, spans)



    output = logits.argmax(dim=-1).t()  # [B, 2]

    score = token_jaccard(torch.stack([spans, output], dim=0))

    return loss.item(), score.item()
def update_average(acc, cur, i):

    return (acc * i + cur) / (i + 1)
train_losses = []

eval_losses = []

eval_scores = []



best_score = 0.0

for epoch in range(NUM_EPOCH):

    loss_avg = 0.0

    model.train()

    pbar = tqdm(train_loader)

    for i, batch in enumerate(pbar):

        loss = train_step(batch)

        loss_avg = update_average(loss_avg, loss, i)

        pbar.set_description(f'Train loss {loss_avg:.4f}')

    train_losses.append(loss_avg)



    loss_avg, score_avg = 0.0, 0.0

    model.eval()

    with torch.no_grad():

        pbar = tqdm(dev_loader)

        for i, batch in enumerate(pbar):

            loss, score = eval_step(batch)

            loss_avg = update_average(loss_avg, loss, i)

            score_avg = update_average(score_avg, score, i)

            pbar.set_description(f'Eval loss {loss_avg:.4f} score {score_avg:.4f}')

    eval_losses.append(loss_avg)

    eval_scores.append(score_avg)



    if best_score < score_avg:

        model.save_pretrained(checkpoint)

        best_score = score_avg



train_losses, eval_losses, eval_scores
del model

model = BertForQuestionAnswering.from_pretrained(checkpoint).cuda()
test_bank = {}

with open(f'{data_dir}/test.csv', 'r', newline='') as f:

    reader = csv.reader(f)

    print(next(reader))

    for textid, text, sentiment in reader:

        test_bank[textid] = (text, sentiment)
def infer_start_end(logits):  # [2, 1, T]

    logits = logits.squeeze(1)  # [2, T]

    eos_pos = (logits[0, :] != float('-inf')).sum() - 1

    if eos_pos <= 2:

        return logits.argmax(dim=-1)

    probs = torch.nn.functional.softmax(logits, dim=-1)  # [2, T]

    start, end = probs[0, 2:eos_pos], probs[1, 2:eos_pos]

    joint_probs = start.unsqueeze(1) * end.unsqueeze(0)  # [T, T]

    pos = torch.triu(joint_probs).argmax()

    start_pos, end_pos = (pos // (eos_pos - 2) + 2), (pos % (eos_pos - 2) + 2)

    return start_pos, end_pos
def submission(textid):

    text, sentiment = test_bank[textid]

    tokens, _ = prepare_input_token(text, '', sentiment)

    input_ids = tokenizer.encode(tokens, pad_to_max_length=True)

    logits = predict(model, torch.tensor([input_ids]).cuda())  # [2, 1, T]

    start, end = infer_start_end(logits)

    prediction = tokenizer.decode(input_ids[start:end+1])

    return unnormalize(prediction, text), prediction
error_cases = {}

with torch.no_grad():

    with open(f'{data_dir}/sample_submission.csv', 'r') as f, open(f'submission.csv', 'w') as g:

        g.write(next(f))

        for line in tqdm(f, total=3534):

            textid = line[:10]

            output, prediction = submission(textid)

            if output is None:

                error_cases[textid] = prediction

                output = test_bank[textid][0]  # echo back

            g.write(f'{textid},"{output}"\n')

len(error_cases)
with open('error_cases.txt', 'w') as f:

    f.write('\n'.join(f'{textid},{prediction}' for textid, prediction in error_cases.items()))