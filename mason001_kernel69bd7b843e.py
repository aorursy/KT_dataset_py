# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import string

from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import transformers
import os
import tokenizers

from sklearn import model_selection
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.save_pretrained('/kaggle/working/')
tokenizer.save_pretrained('/kaggle/working/')
MAX_LEN = 128
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 24
EPOCHS = 3
BERT_PATH = '/kaggle/working/'
MODEL_PATH = '/kaggle/working/model.bin'
TRAINING_FILE = '/kaggle/input/tweet-sentiment-extraction/train.csv'
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    os.path.join(BERT_PATH, "vocab.txt"),
    lowercase=True
)

class BERTBaseUncased(nn.Module):

    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(BERT_PATH)
        self.l0 = nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids):
        # not using sentiment at all
        sequence_output, pooled_output = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        # (batch_size, num_tokens, 768)
        logits = self.l0(sequence_output)
        # (batch_size, num_tokens, 2)
        # (batch_size, num_tokens, 1), (batch_size, num_tokens, 1)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # (batch_size, num_tokens), (batch_size, num_tokens)

        return start_logits, end_logits
class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        tweet = " ".join(str(self.tweet[item]).split())
        selected_text = " ".join(str(self.selected_text[item]).split())

        len_st = len(selected_text)
        idx0 = -1
        idx1 = -1
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
            if tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0, idx1 + 1):
                if tweet[j] != " ":
                    char_targets[j] = 1

        tok_tweet = self.tokenizer.encode(sequence=self.sentiment[item], pair=tweet)
        tok_tweet_tokens = tok_tweet.tokens
        tok_tweet_ids = tok_tweet.ids
        tok_tweet_offsets = tok_tweet.offsets[3:-1]

        targets = [0] * (len(tok_tweet_tokens) - 4)
        if self.sentiment[item] == "positive" or self.sentiment[item] == "negative":
            sub_minus = 8
        else:
            sub_minus = 7

        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
            if sum(char_targets[offset1 - sub_minus:offset2 - sub_minus]) > 0:
                targets[j] = 1

        targets = [0] + [0] + [0] + targets + [0]

        targets_start = [0] * len(targets)
        targets_end = [0] * len(targets)

        non_zero = np.nonzero(targets)[0]
        if len(non_zero) > 0:
            targets_start[non_zero[0]] = 1
            targets_end[non_zero[-1]] = 1

        mask = [1] * len(tok_tweet_ids)
        token_type_ids = [0] * 3 + [1] * (len(tok_tweet_ids) - 3)

        padding_length = self.max_len - len(tok_tweet_ids)
        ids = tok_tweet_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        targets = targets + ([0] * padding_length)
        targets_start = targets_start + ([0] * padding_length)
        targets_end = targets_end + ([0] * padding_length)

        sentiment = [1, 0, 0]
        if self.sentiment[item] == "positive":
            sentiment = [0, 0, 1]
        if self.sentiment[item] == "negative":
            sentiment = [0, 1, 0]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'tweet_tokens': " ".join(tok_tweet_tokens),
            'targets': torch.tensor(targets, dtype=torch.long),
            'targets_start': torch.tensor(targets_start, dtype=torch.long),
            'targets_end': torch.tensor(targets_end, dtype=torch.long),
            'padding_len': torch.tensor(padding_length, dtype=torch.long),
            'orig_tweet': self.tweet[item],
            'orig_selected': self.selected_text[item],
            'sentiment': torch.tensor(sentiment, dtype=torch.float),
            'orig_sentiment': self.sentiment[item]
        }
def loss_fn(o1, o2, t1, t2):
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    l2 = nn.BCEWithLogitsLoss()(o2, t2)
    return l1 + l2


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets_start = d['targets_start']
        targets_end = d['targets_end']
        
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        loss = loss_fn(o1, o2, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)
        
        
def eval_fn(data_loader, model, device):
    model.eval()
    fin_output_start = []
    fin_output_end = []
    fin_padding_len = []
    fin_tweet_tokens = []
    fin_orig_sentiment = []
    fin_orig_selected = []
    fin_orig_tweet = []
    
    for bi, d in enumerate(data_loader):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        tweet_tokens = d['tweet_tokens']
        padding_len = d['padding_len']
        orig_sentiment = d['orig_sentiment']
        original_selected = d['orig_selected']
        orig_tweet = d['orig_tweet']
        
        
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        
        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        fin_output_start.append(torch.sigmoid(o1).cpu().detach().numpy())
        fin_output_end.append(torch.sigmoid(o2).cpu().detach().numpy())
        fin_padding_len.extend(padding_len.cpu().numpy().tolist())
        fin_tweet_tokens.extend(tweet_tokens)
        fin_orig_sentiment.extend(orig_sentiment)
        fin_orig_selected.extend(original_selected)
        fin_orig_tweet.extend(orig_tweet)
        
    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)
    
    threshold = 0.2
    jaccards = []
    for j in range(len(fin_tweet_tokens)):
        target_string = fin_orig_selected[j]
        tweet_tokens = fin_tweet_tokens[j]
        padding_len = fin_padding_len[j]
        original_tweet = fin_orig_tweet[j]
        sentiment = fin_orig_sentiment[j]
        
        if padding_len > 0:
            mask_start = fin_output_start[j, :][:-padding_len] >= threshold
            mask_end = fin_output_end[j, :][:-padding_len] >= threshold
        else:
            mask_start = fin_output_start[j, :] >= threshold
            mask_end = fin_output_end[j, :] >= threshold
        mask = [0] * len(mask_start)
        idx_start = np.nonzero(mask_start)[0]
        idx_end = np.nonzero(mask_end)[0]
        
        if len(idx_start) > 0:
            idx_start = idx_start[0]
            if len(idx_end) > 0:
                idx_end = idx_end[0]
            else:
                idx_end = idx_start
        else:
            idx_start, idx_end = 0, 0
            
        for mj in range(idx_start, idx_end+1):
            mask[mj] = 1
            
        output_tokens = [x for p, x in enumerate(tweet_tokens.split()) if mask[p] == 1]
        output_tokens = [x for x in output_tokens if x not in ("[CLS]",  "[SEP]")]
        
        final_output = ""
        for ot in output_tokens:
            if ot.startswith("##"):
                final_output = final_output+ot[2:]
            elif len(ot) == 1 and ot in string.punctuation:
                final_output = final_output + ot
            else:
                final_output = final_output + ' ' + ot
        final_output = final_output.strip()
        if sentiment == 'neutral' or len (original_tweet.split()) < 4:
            final_output = original_tweet
        jac = jaccard(target_string.strip(), final_output.strip())
        jaccards.append(jac)
    mean_jac = np.mean(jaccards)
    return mean_jac
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def run():
    dfx = pd.read_csv(TRAINING_FILE).dropna().reset_index(drop=True)
    
    print(dfx.shape)
    
    df_train, df_valid = model_selection.train_test_split(
        dfx,
        test_size=0.15,
        random_state=49,
        stratify=dfx.sentiment.values
    )
    
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    
    train_dataset = TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values
    )
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=3
    )
    
    valid_dataset = TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values
    )
    
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        num_workers=1
    )
    
    device = torch.device('cuda')
    model = BERTBaseUncased()
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    
    optimizer_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.001,
        },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
        }
    ]
    
    num_train_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )

    best_jaccard = 0
    for epoch in range(EPOCHS):
        train_fn(train_data_loader, model, optimizer, device, scheduler)
        jaccard = eval_fn(valid_data_loader, model, device)
        print(f"Jaccard Score = {jaccard}")
        if jaccard > best_jaccard:
            torch.save(model.state_dict(), MODEL_PATH)
            best_jaccard = jaccard
run()
device = torch.device("cuda")
model = BERTBaseUncased()
model.to(device)
# model = nn.DataParallel(model)
model.load_state_dict(torch.load("/kaggle/working/model.bin"))
model.eval()
df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
df_test.loc[:, "selected_text"] = df_test.text.values

test_dataset = TweetDataset(
        tweet=df_test.text.values,
        sentiment=df_test.sentiment.values,
        selected_text=df_test.selected_text.values
    )

data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=VALID_BATCH_SIZE,
    num_workers=1
)
all_outputs = []
fin_outputs_start = []
fin_outputs_end = []
fin_outputs_start2 = []
fin_outputs_end2 = []
fin_tweet_tokens = []
fin_padding_lens = []
fin_orig_selected = []
fin_orig_sentiment = []
fin_orig_tweet = []
fin_tweet_token_ids = []

with torch.no_grad():
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        tweet_tokens = d["tweet_tokens"]
        padding_len = d["padding_len"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_sentiment = d["orig_sentiment"]
        orig_tweet = d["orig_tweet"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        sentiment = sentiment.to(device, dtype=torch.float)

        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        fin_outputs_start.append(torch.sigmoid(outputs_start).cpu().detach().numpy())
        fin_outputs_end.append(torch.sigmoid(outputs_end).cpu().detach().numpy())

        fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())
        fin_tweet_token_ids.append(ids.cpu().detach().numpy().tolist())

        fin_tweet_tokens.extend(tweet_tokens)
        fin_orig_sentiment.extend(orig_sentiment)
        fin_orig_selected.extend(orig_selected)
        fin_orig_tweet.extend(orig_tweet)

fin_outputs_start = np.vstack(fin_outputs_start)
fin_outputs_end = np.vstack(fin_outputs_end)

fin_tweet_token_ids = np.vstack(fin_tweet_token_ids)
jaccards = []
threshold = 0.3
for j in range(len(fin_tweet_tokens)):
    target_string = fin_orig_selected[j]
    tweet_tokens = fin_tweet_tokens[j]
    padding_len = fin_padding_lens[j]
    original_tweet = fin_orig_tweet[j]
    sentiment_val = fin_orig_sentiment[j]

    if padding_len > 0:
        mask_start = fin_outputs_start[j, 3:-1][:-padding_len] >= threshold
        mask_end = fin_outputs_end[j, 3:-1][:-padding_len] >= threshold
        tweet_token_ids = fin_tweet_token_ids[j, 3:-1][:-padding_len]
    else:
        mask_start = fin_outputs_start[j, 3:-1] >= threshold
        mask_end = fin_outputs_end[j, 3:-1] >= threshold
        tweet_token_ids = fin_tweet_token_ids[j, 3:-1]

    mask = [0] * len(mask_start)
    idx_start = np.nonzero(mask_start)[0]
    idx_end = np.nonzero(mask_end)[0]
    if len(idx_start) > 0:
        idx_start = idx_start[0]
        if len(idx_end) > 0:
            idx_end = idx_end[0]
        else:
            idx_end = idx_start
    else:
        idx_start = 0
        idx_end = 0

    for mj in range(idx_start, idx_end + 1):
        mask[mj] = 1

    output_tokens = [x for p, x in enumerate(tweet_token_ids) if mask[p] == 1]

    filtered_output = TOKENIZER.decode(output_tokens)
    filtered_output = filtered_output.strip().lower()

    if sentiment_val == "neutral":
        filtered_output = original_tweet

    all_outputs.append(filtered_output.strip())
sample = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
sample.loc[:, 'selected_text'] = all_outputs
sample.to_csv("/kaggle/working/submission.csv", index=False)

sample.head()
sample.to_csv('submission.csv', index=False)
df_test['selected_text']=all_outputs
result = df_test[['textID', 'selected_text']].copy()
result.to_csv('submission.csv', index=False)
