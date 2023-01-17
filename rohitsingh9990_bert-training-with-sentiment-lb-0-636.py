import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import os

import tokenizers

import string

import torch

import transformers

import torch.nn as nn

from tqdm import tqdm

import re



from sklearn import model_selection

from sklearn import metrics

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup
MAX_LEN = 128

TRAIN_BATCH_SIZE = 40

VALID_BATCH_SIZE = 16

EPOCHS = 2





BERT_PATH = "../input/bert-base-uncased/" # running from run.sh



# MODEL_PATH = "models/bert_w_sent/model.bin"

TRAINING_FILE = "../input/tweet-sentiment-extraction/train.csv"



TOKENIZER = tokenizers.BertWordPieceTokenizer(

    os.path.join(BERT_PATH, 'vocab.txt'),

    lowercase=True

)

class TweetDataSet:

    def __init__(self, tweet, sentiment, selected_text):

        self.tweet = tweet

        self.sentiment = sentiment

        self.selected_text = selected_text

        self.max_len = MAX_LEN

        self.tokenizer = TOKENIZER



    def __len__(self):

        return len(self.tweet)



    def __getitem__(self, item):

        # removes all extra spaces in between words

        tweet = " ".join(str(self.tweet[item]).split())

        selected_text = " ".join(str(self.selected_text[item]).split())



        start_ind = tweet.find(selected_text) # same as idx0

        end_ind = start_ind + len(selected_text) # same as idx1

        # print(start_ind, end_ind)



        len_selected_text = len(selected_text)

        idx0 = -1

        idx1 = -1

        # replace this with your find code



        for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):

            if tweet[ind: ind + len_selected_text] == selected_text:

                idx0 = ind

                idx1 = ind + len_selected_text - 1

                break

        # print(idx0, idx1)



        char_targets = [0] * len(tweet)

        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0]

        if idx0 != -1 and idx1 != -1:

            for j in range(idx0, idx1 + 1):

                if tweet[j] != " ":

                    char_targets[j] = 1

        # [0,0,0,0,1,1,1,0,1,1,1,0,1,1,1,0,0,,0,0,0,0,0,0] 0 between 1's denotes space

        # tok_tweet = self.tokenizer.encode(tweet)

        tok_tweet = self.tokenizer.encode(sequence=self.sentiment[item], pair=tweet)

        

        tok_tweet_tokens = tok_tweet.tokens

        tok_tweet_ids = tok_tweet.ids

        # with sentiment # CLS, 0,0,1, SEP

        tok_tweet_offsets = tok_tweet.offsets[3:-1] # as first and last tokens are always cls and sep

        targets = [0] * (len(tok_tweet_tokens) - 4) # -4 for cls and sep and sentiment and cls

        # targets = [0] * (len(tok_tweet_tokens) - 2) # -2 for cls and sep



        if self.sentiment[item] == "positive" or self.sentiment[item] == "negative":

            sub_minus = 8

        else:

            sub_minus = 7



        # [0,0,0,0,0,0,0]

        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):

            # if sum(char_targets[offset1:offset2]) > 0: # this is done to encounter cases where characters in between word comes in selected text | to encounter partial match

            if sum(char_targets[offset1 - sub_minus :offset2 - sub_minus]) > 0:

                targets[j] = 1

        # [0,0,1,1,1,0,0]

        # from here on we don't need char_targets, we only need targets

        targets = [0] + [0] + [0] + targets + [0] # cls + sentiment + cls + targets +  sep

        targets_start = [0] * len(targets)

        targets_end = [0] * len(targets)

        non_zero = np.nonzero(targets)[0] 

        if len(non_zero) > 0:

            targets_start[non_zero[0]] = 1

            targets_end[non_zero[-1]] = 1



        mask = [1] * len(tok_tweet_ids)

        # token_type_ids = [0] * len(tok_tweet_ids)

        token_type_ids = [0] * 3 + [1] * (len(tok_tweet_ids) - 3)

        

        padding_len = self.max_len - len(tok_tweet_ids)

        ids = tok_tweet_ids + ([0] * padding_len)

        mask = mask + ([0] * padding_len)

        token_type_ids = token_type_ids + ([0] * padding_len)

        targets = targets + ([0] * padding_len)

        targets_start = targets_start + ([0] * padding_len)

        targets_end = targets_end + ([0] * padding_len)



        sentiment = [1, 0, 0]

        if self.sentiment[item] == "positive":

            sentiment = [0, 0, 1]

        if self.sentiment[item] == "negative":

            sentiment = [0, 1, 0]



        return {

            "ids": torch.tensor(ids, dtype=torch.long),

            "mask": torch.tensor(mask, dtype=torch.long),

            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),

            "targets": torch.tensor(targets, dtype=torch.long),

            "targets_start": torch.tensor(targets_start, dtype=torch.long),

            "targets_end": torch.tensor(targets_end, dtype=torch.long),

            "padding_len": torch.tensor(padding_len, dtype=torch.long),

            "tweet_tokens": " ".join(tok_tweet_tokens),

            "orig_tweet": self.tweet[item],

            "sentiment": torch.tensor(sentiment, dtype=torch.long),

            "orig_sentiment": self.sentiment[item],

            "orig_selected": self.selected_text[item]

        }
class AverageMeter():

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





def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
def loss_fn(o1, o2, t1, t2):

    l1 = nn.BCEWithLogitsLoss()(o1, t1)

    l2 = nn.BCEWithLogitsLoss()(o2, t2)

    return l1 + l2





def train_fn(data_loader, model, optimizer, device, scheduler):

    model.train()

    losses = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):

        ids = d["ids"]

        token_type_ids = d["token_type_ids"]

        mask = d["mask"]

        targets_start = d["targets_start"]

        targets_end = d["targets_end"]



        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets_start = targets_start.to(device, dtype=torch.float)

        targets_end = targets_end.to(device, dtype=torch.float)

      

        optimizer.zero_grad()

        o1, o2 = model(

            ids = ids,

            mask = mask,

            token_type_ids = token_type_ids

        )



        loss = loss_fn(o1, o2, targets_start, targets_end)

        loss.backward()

        optimizer.step()

        losses.update(loss.item(), ids.size(0))

        tk0.set_postfix(loss=losses.avg)







def eval_fn(data_loader, model, device):

    model.eval()

    fin_output_start = []

    fin_output_end = []

    fin_padding_lens = []

    fin_tweet_tokens = []

    fin_orig_sentiment = []

    fin_orig_selected = []

    fin_orig_tweet = []



    losses = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):

        ids = d["ids"]

        token_type_ids = d["token_type_ids"]

        mask = d["mask"]

        targets_start = d["targets_start"]

        targets_end = d["targets_end"]



        tweet_tokens = d["tweet_tokens"]

        padding_len = d["padding_len"]

        # sentiment = d["sentiment"]

        orig_sentiment = d["orig_sentiment"]

        orig_selected = d["orig_selected"]

        orig_tweet = d["orig_tweet"]





        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets_start = targets_start.to(device, dtype=torch.float)

        targets_end = targets_end.to(device, dtype=torch.float)

      

        o1, o2 = model(

            ids = ids,

            mask = mask,

            token_type_ids = token_type_ids

        )



        fin_output_start.append(torch.sigmoid(o1).cpu().detach().numpy())

        fin_output_end.append(torch.sigmoid(o2).cpu().detach().numpy())

        fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())



        fin_tweet_tokens.extend(tweet_tokens)

        fin_orig_sentiment.extend(orig_sentiment)

        fin_orig_selected.extend(orig_selected)

        fin_orig_tweet.extend(orig_tweet)





    fin_output_start = np.vstack(fin_output_start)

    fin_output_end = np.vstack(fin_output_end)



    threshold = 0.3

    jaccards = []

    for j in range(len(fin_tweet_tokens)):

        target_string = fin_orig_selected[j]

        tweet_tokens = fin_tweet_tokens[j]

        padding_len = fin_padding_lens[j]

        original_tweet = fin_orig_tweet[j]

        sentiment = fin_orig_sentiment[j]



        if padding_len > 0:

            mask_start = fin_output_start[j, 3:][:-padding_len] >= threshold

            mask_end = fin_output_end[j,3:][:-padding_len] >= threshold

        else:

            mask_start = fin_output_start[j, 3:] >= threshold

            mask_end = fin_output_end[j,3:] >= threshold



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



        output_tokens = [x for p, x in enumerate(tweet_tokens.split()[3:]) if mask[p] == 1]

        output_tokens = [x for x in output_tokens if x not in ("[CLS]", "[SEP]")]



        final_output = ""

        for ot in output_tokens:

            if ot.startswith("##"):

                final_output = final_output + ot[2:]

            elif len(ot) == 1 and ot in string.punctuation:

                final_output = final_output + ot

            else:

                final_output = final_output + " " + ot

        final_output = final_output.strip()

        # if sentiment == "neutral" or len(original_tweet.split()) < 4:

        if sentiment == "neutral" or len(original_tweet.split()) < 4:

            final_output = original_tweet

        jac = jaccard(target_string.strip(), final_output.strip())

        jaccards.append(jac)

    mean_jac = np.mean(jaccards)

    return mean_jac



class BERTBaseUncased(nn.Module):

    def __init__(self):

        super(BERTBaseUncased, self).__init__()

        self.bert = transformers.BertModel.from_pretrained(BERT_PATH)

        self.bert_drop = nn.Dropout(0.3)

        self.l0 = nn.Linear(768, 2)

    

    def forward(self, ids, mask, token_type_ids, sentiment=None):

        # not using sentiment for now

        sequence_output, pooled_output = self.bert(

            ids, 

            attention_mask=mask,

            token_type_ids=token_type_ids

        )

        # sequence_output will have (batch_size, num_tokens, 768)

        logits = self.l0(sequence_output)

        # logits (batch_size, num_tokens, 2)

        start_logits, end_logits = logits.split(1, dim=-1) # (batch_size, num_tokens, 1), (batch_size, num_tokens, 1)

        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)

        # (batch_size, num_tokens), (batch_size, num_tokens)

       

        return start_logits, end_logits
def run():



    dfx = pd.read_csv(TRAINING_FILE).dropna().reset_index(drop=True)

    print(dfx.shape)



    df_train, df_valid = model_selection.train_test_split(

        dfx,

        test_size=0.1,

        random_state=42,

        stratify=dfx.sentiment.values

    )



    df_train = df_train.reset_index(drop=True)

    df_valid = df_valid.reset_index(drop=True)



    train_dataset = TweetDataSet(

        tweet=df_train.text.values,

        sentiment=df_train.sentiment.values,

        selected_text=df_train.selected_text.values

    )

    



    train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=TRAIN_BATCH_SIZE,

        num_workers=0

    )



    # print(train_dataset[0])



    valid_dataset = TweetDataSet(

        tweet=df_valid.text.values,

        sentiment=df_valid.sentiment.values,

        selected_text=df_valid.selected_text.values

    )



    valid_data_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=VALID_BATCH_SIZE,

        num_workers=0

    )



    device = torch.device("cuda")

    model = BERTBaseUncased()

    model.to(device)

    

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

    ]



    num_train_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)

    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=0,

        num_training_steps=num_train_steps

    )



    if torch.cuda.device_count() > 1:

        model = nn.DataParallel(model)



    best_jaccard = 0

    for epoch in range(EPOCHS):

        print(f"Epoch {epoch}/{EPOCHS}")

        train_fn(train_data_loader, model, optimizer, device, scheduler)

        jaccard = eval_fn(valid_data_loader, model, device)



        if jaccard > best_jaccard:

            torch.save(model.state_dict(), 'model.bin')

            best_jaccard = jaccard

        print(f"Jaccard Score = {jaccard}, best jaccard Score = {best_jaccard}")
run()