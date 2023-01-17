import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import os
from tqdm import tqdm 
train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
ss_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
print(train_df.shape, test_df.shape)
import transformers
import tokenizers
import torch
import torch.nn as nn
# Config variables
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 10
BERT_PATH = '/kaggle/input/bert-base-uncased'
MODEL_PATH = '/kaggle/working/model.h5'
TRAINING_FILE = '/kaggle/input/tweet-sentiment-extraction/train.csv'
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    os.path.join(BERT_PATH, 'vocab.txt'),
    lowercase=True
)
#Evaluation metric
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
class AverageMeter(object):
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
#Creating dataset
class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.max_len = MAX_LEN
        self.tokenizer = TOKENIZER
        
    def __len__(self):
        return len(self.sentiment)
    
    def __getitem__(self, idx):
        tweet = " ".join(self.tweet[idx].split())
        selected_text = " ".join(self.selected_text[idx].split())
        
        len_sel_txt = len(selected_text)
        idx0 = 0
        idx1 = 1
        for ind in (i for i, e in enumerate(tweet) if e==selected_text[0]):
            if tweet[ind : ind+len_sel_txt] == selected_text:
                idx0 = ind
                idx1 = ind+len_sel_txt-1
                break
        char_targets = [0] * len(tweet)
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0, idx1+1):
                if tweet[j] != " ":
                    char_targets[j] = 1
        
        tok_tweet = self.tokenizer.encode(tweet)
        tok_tweet_tokens = tok_tweet.tokens
        tok_tweet_ids = tok_tweet.ids
        tok_tweet_offsets = tok_tweet.offsets[1:-1] #First and last are for [CLS] and [SEP]
        
        targets = [0] * (len(tok_tweet_tokens)-2)
        for i, (offset1, offset2) in enumerate(tok_tweet_offsets):
            if sum(char_targets[offset1:offset2])>0:
                targets[i] = 1
                
        targets = [0] + targets + [0]
        targets_start = [0] * len(targets)
        targets_end = [0] * len(targets)
        
        non_zero = np.nonzero(targets)[0] #Indices of non-zero values
        if len(non_zero) > 0:
            targets_start[non_zero[0]] = 1
            targets_end[non_zero[-1]] = 1
        
        mask = [1] * len(tok_tweet_ids)
        token_type_ids = [0] * len(tok_tweet_ids)
        
        padding_len = self.max_len-len(tok_tweet_ids)
        ids = tok_tweet_ids + [0]*padding_len
        mask = mask + [0]*padding_len
        token_type_ids = token_type_ids + [0]*padding_len
        targets = targets + [0]*padding_len
        targets_start = targets_start + [0]*padding_len
        targets_end = targets_end + [0]*padding_len
        
        sentiment = [1,0,0]
        if self.sentiment[idx] == "positive":
            sentiment = [0,0,1]
        if self.sentiment[idx] == "negative":
            sentiment = [0,1,0]
        
        return {
            "ids" : torch.tensor(ids, dtype=torch.long),
            "mask" : torch.tensor(mask, dtype=torch.long),
            "token_type_ids" : torch.tensor(token_type_ids, dtype=torch.long),
            "targets" : torch.tensor(targets, dtype=torch.long),
            "targets_start" : torch.tensor(targets_start, dtype=torch.long),
            "targets_end" : torch.tensor(targets_end, dtype=torch.long),
            "tweet_tokens" : " ".join(tok_tweet_tokens),
            "orig_tweet" : self.tweet[idx],
            "sentiment" : torch.tensor(sentiment, dtype=torch.long),
            "orig_sentiment" : self.sentiment[idx],
            "orig_selected_text" : self.selected_text[idx]
        }
# Model definition
class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(BERT_PATH)
        self.bert_drop = nn.Dropout(0.2)
        self.l0 = nn.Linear(768, 2)
    
    def forward(self, ids, mask, token_type_ids):
        sequence_output, pooled_output = self.bert(
            ids,
            mask,
            token_type_ids
        )
        logits = self.l0(sequence_output)
        #Splitting 768X2 tensor into 2 768X1 tensors
        start_logit, end_logit = logits.split(1, dim=-1)
        start_logit = start_logit.squeeze(-1)
        end_logit = end_logit.squeeze(-1)
        return start_logit, end_logit
def loss_fn(o1, o2, t1, t2):
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    l2 = nn.BCEWithLogitsLoss()(o2, t2)
    return l1+l2
def train_fn(data_loader, model, optiizer, device, scheduler):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    for index, data in enumerate(tk0):
        ids = data["ids"]
        mask = data["mask"]
        token_type_ids = data["token_type_ids"]
        targets_start = data["targets_start"]
        targets_end = data["targets_end"]
        
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        start, end = model(
            ids,
            mask,
            token_type_ids
        )
        
        loss = loss_fn(start, end, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
df_train, df_valid = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df.sentiment.values)
df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)
train_dataset = TweetDataset(
    tweet = df_train.text.values,
    sentiment = df_train.sentiment.values, 
    selected_text = df_train.selected_text.values
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size = TRAIN_BATCH_SIZE
    #num_workers
)

valid_dataset = TweetDataset(
    tweet = df_valid.text.values,
    sentiment = df_valid.sentiment.values, 
    selected_text = df_valid.selected_text.values
)

valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset, 
    batch_size = VALID_BATCH_SIZE
    #num_workers
)

model = BERTBaseUncased()
model.to(device)
parameters = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

optimizer_params = [
    {
        'params' : [
            p for n, p in parameters if not any(nd in n for nd in no_decay)
        ],   
        'weight_decay':0.001
    }, #Check syntax
    {'params' : [p for n, p in parameters if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
]
num_train_steps = int(EPOCHS* len(df_train)/TRAIN_BATCH_SIZE)
optimizer = transformers.AdamW(optimizer_params, lr=3e-5)
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0, #What is this?
    num_training_steps = num_train_steps
)
best_jaccard = 0
for epoch in range(EPOCHS):
    print("Epoch {}".format(epoch))
    train_fn(train_dataloader, model, optimizer, device, scheduler)
#     jaccard = eval_fn(valid_dataloader, model, device)
#     print("jaccard score : {}".format(accuracy))
#     if accuracy > best_acc:
#         torch.save(model.state_dict(), f"{MODEL_PATH}/model_{epoch}.pkl")
#         best_jaccard = jaccard
