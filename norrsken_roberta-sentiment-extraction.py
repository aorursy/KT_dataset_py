import numpy as np

import pandas as pd

import os

import re

import string

import tokenizers

import transformers



import torch

import torch.nn as nn



from sklearn import model_selection

from torch.nn import functional as F

from tqdm import tqdm

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup
MAX_LEN = 192

TRAIN_BATCH_SIZE = 32

VALID_BATCH_SIZE = 8

EPOCHS = 5

ROBERTA_PATH = "../input/roberta-base"

TOKENIZER = tokenizers.ByteLevelBPETokenizer(

    vocab_file=f"{ROBERTA_PATH}/vocab.json", 

    merges_file=f"{ROBERTA_PATH}/merges.txt", 

    lowercase=True,

    add_prefix_space=True

)

DEVICE = 'cuda'

TRAIN_PATH = '../input/tweet-sentiment-extraction/train.csv'

FOLDS = 5
train_df = pd.read_csv(TRAIN_PATH)

train_df['kfolds'] = -1

kf = model_selection.StratifiedKFold(n_splits = FOLDS, shuffle = False, random_state = 10)

for fold, (train_idx, val_idx) in enumerate(kf.split(X = train_df, y=train_df.sentiment.values)):

    print(len(train_idx), len(val_idx))

    train_df.loc[val_idx, 'kfolds'] = fold

train_df
class AverageMeter:

    """

    Computes and stores the average and current value

    """

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





class EarlyStopping:

    def __init__(self, patience=7, mode="max", delta=0.001):

        self.patience = patience

        self.counter = 0

        self.mode = mode

        self.best_score = None

        self.early_stop = False

        self.delta = delta

        if self.mode == "min":

            self.val_score = np.Inf

        else:

            self.val_score = -np.Inf



    def __call__(self, epoch_score, model, model_path):



        if self.mode == "min":

            score = -1.0 * epoch_score

        else:

            score = np.copy(epoch_score)



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(epoch_score, model, model_path)

        elif score < self.best_score + self.delta:

            self.counter += 1

            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(epoch_score, model, model_path)

            self.counter = 0



    def save_checkpoint(self, epoch_score, model, model_path):

        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:

            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))

            torch.save(model.state_dict(), model_path)

        self.val_score = epoch_score





def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
class TweetModel(transformers.BertPreTrainedModel):

    def __init__(self, conf):

        super(TweetModel, self).__init__(conf)

        # Let's take pretrained weights for our model

        config = transformers.RobertaConfig.from_pretrained(

            '../input/roberta-base/config.json', output_hidden_states=True)    

        self.roberta = transformers.RobertaModel.from_pretrained(

            '../input/roberta-base/pytorch_model.bin', config=config)

        self.drop_out = nn.Dropout(0.3)

        # The final layer will have two output features for start and end indexes.

        self.fc = nn.Linear(768, 2)

#         nn.init.normal_(self.l0.weight, std = 0.02)

        nn.init.normal_(self.fc.weight, std=0.02)

        nn.init.normal_(self.fc.bias, 0)

        

    def forward(self, ids, mask):

        _, _, out = self.roberta(

            ids,

            attention_mask = mask

        )

        out = torch.stack([out[-1], out[-2], out[-3], out[-4]])

        out = torch.mean(out, 0)

        out = self.drop_out(out)

        logits = self.fc(out)

        start_logits, end_logits = logits.split(1, dim = -1)

        

        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)

        

        return start_logits, end_logits
def process_data(tweet, selected_text, sentiment, tokenizer, max_len):

    '''

    tweet - Tweet from which we have to perform sentiment extraction.

    selected_text - Expected output of sentiment extraction

    sentiment - sentiment to extract (positive, negative or neutral)

    tokenizer - tokenizer to be used for creating tokens

    max_len - max length of tweet.

    '''

    tweet = " " + " ".join(str(tweet).split())

    selected_text = " " + " ".join(str(selected_text).split())

    

    len_st = len(selected_text) - 1

    idx0 = None

    idx1 = None

    

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):

        if " " + tweet[ind: ind + len_st] == selected_text:

            idx0 = ind

            idx1 = ind + len_st - 1

            break

            

    char_targets = [0] * len(tweet)

    if idx0 != None and idx1 != None:

        char_targets[idx0: idx1 + 1] = [1]*(idx1 + 1 - idx0)



    tok_tweets = tokenizer.encode(tweet)

    input_ids_orig = tok_tweets.ids

    tweet_offsets = tok_tweets.offsets

    

    target_idx = []

    

    for j, (offset1, offset2) in enumerate(tweet_offsets):

        if sum(char_targets[offset1: offset2]) > 0:

            target_idx.append(j)

            

    targets_start = target_idx[0]

    targets_end = target_idx[-1]

    sentiment_id = {

        'positive': 1313,

        'negative': 2430,

        'neutral': 7974

    }

    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]

    token_type_ids = [0, 0, 0, 0] + [1] * (len(input_ids_orig) + 1)

    mask = [1] * len(token_type_ids)

    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0,0)]

    targets_start += 4

    targets_end += 4

    

    padding_length = max_len - len(input_ids)

    if padding_length > 0:

        input_ids = input_ids + ([1] * padding_length)

        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

        mask = mask + [0] * padding_length

        token_type_ids += [0] * padding_length

    

    return {

        'ids': input_ids,

        'mask': mask,

        'token_type_ids': token_type_ids,

        'targets_start': targets_start,

        'targets_end': targets_end,

        'orig_tweet': tweet,

        'orig_selected': selected_text,

        'sentiment': sentiment,

        'offsets': tweet_offsets

    }
df_train = train_df[train_df.kfolds == 1].reset_index(drop = True)

print('Tweet: ' + df_train.iloc[0]['text'])

print('Sentiment: ' + df_train.iloc[0]['sentiment'])

print('Expected Output: ' + df_train.iloc[0]['selected_text'])

output = process_data(df_train.iloc[0]['text'], df_train.iloc[0]['selected_text'], df_train.iloc[0]['sentiment'],TOKENIZER, MAX_LEN)



print('Tokens: ')

print(output['ids'])

print('Token Types:')

print(output['token_type_ids'])

print('Mask:')

print(output['mask'])

print('Offsets:')

print(output['offsets'])
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

        data = process_data(

            self.tweet[item],

            self.selected_text[item],

            self.sentiment[item],

            self.tokenizer,

            self.max_len

        )

        

        return {

            'ids': torch.tensor(data['ids'], dtype = torch.long),

            'mask': torch.tensor(data['mask'], dtype = torch.long),

            'token_type_ids': torch.tensor(data['token_type_ids'], dtype = torch.long),

            'targets_start': torch.tensor(data['targets_start'], dtype = torch.long),

            'targets_end': torch.tensor(data['targets_end'], dtype = torch.long),

            'orig_tweet': data['orig_tweet'],

            'orig_selected': data['orig_selected'],

            'sentiment': data['sentiment'],

            'offsets': torch.tensor(data['offsets'], dtype = torch.long)

        }
def calculate_jaccard_score(

    original_tweet, 

    target_string, 

    sentiment_val, 

    idx_start, 

    idx_end, 

    offsets,

    verbose=False):

    

    if idx_end < idx_start:

        idx_end = idx_start

    

    filtered_output  = ""

    for ix in range(idx_start, idx_end + 1):

        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]

        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:

            filtered_output += " "



    if sentiment_val == "neutral" or len(original_tweet.split()) < 2:

        filtered_output = original_tweet



    jac = jaccard(target_string.strip(), filtered_output.strip())

    return jac, filtered_output
def loss_fn(start_logits, end_logits, start_positions, end_positions):

    loss_fct = nn.CrossEntropyLoss()

    start_loss = loss_fct(start_logits, start_positions)

    end_loss = loss_fct(end_logits, end_positions)

    total_loss = (start_loss + end_loss)

    return total_loss


def train(data_loader, model, optimizer, device, scheduler = None):

    model.train()

    losses = AverageMeter()

    jaccards = AverageMeter()

    

    tk0 = tqdm(data_loader, total = len(data_loader))

    

    for bi, d in enumerate(tk0):

        ids = d["ids"]

        mask = d["mask"]

        token_type_ids = d["token_type_ids"]

        targets_start = d["targets_start"]

        targets_end = d["targets_end"]

        orig_selected = d["orig_selected"]

        orig_tweet = d["orig_tweet"]

        sentiment = d["sentiment"]

        offsets = d["offsets"]

        

        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets_start = targets_start.to(device, dtype=torch.long)

        targets_end = targets_end.to(device, dtype=torch.long)

        

        model.zero_grad()

        predicted_start, predicted_end = model(ids, mask)

        loss = loss_fn(predicted_start, predicted_end, targets_start, targets_end)

        loss.backward()

        optimizer.step()

        scheduler.step()

        

        

        predicted_start = torch.softmax(predicted_start, dim = 1).cpu().detach().numpy()

        predicted_end = torch.softmax(predicted_end, dim = 1).cpu().detach().numpy()

        

        jaccard_scores = []

        

        for i, tweet in enumerate(orig_tweet):

            selected_tweet = orig_selected[i]

            tweet_sentiment = sentiment[i]

            jaccard_score, _ = calculate_jaccard_score(

                                original_tweet = tweet,

                                target_string = selected_tweet,

                                sentiment_val = tweet_sentiment,

                                idx_start = np.argmax(predicted_start[i, :]),

                                idx_end = np.argmax(predicted_end[i, :]),

                                offsets = offsets[i]

                                )

            jaccard_scores.append(jaccard_score)

            

            jaccards.update(np.mean(jaccard_scores), ids.size(0))

            losses.update(loss.item(), ids.size(0))

            

            tk0.set_postfix(loss = losses.avg, jaccard = jaccards.avg)

        
def eval_fn(data_loader, model, device, scheduler = None):

    model.eval()

    losses = AverageMeter()

    jaccards = AverageMeter()

    with torch.no_grad():

        tk0 = tqdm(data_loader, total = len(data_loader))

        for bi, d in enumerate(tk0):

            ids = d["ids"]

            mask = d["mask"]

            token_type_ids = d["token_type_ids"]

            targets_start = d["targets_start"]

            targets_end = d["targets_end"]

            orig_selected = d["orig_selected"]

            orig_tweet = d["orig_tweet"]

            sentiment = d["sentiment"]

            offsets = d["offsets"]



            ids = ids.to(device, dtype=torch.long)

            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            targets_start = targets_start.to(device, dtype=torch.long)

            targets_end = targets_end.to(device, dtype=torch.long)



            predicted_start, predicted_end = model(ids, mask)

            

            loss = loss_fn(predicted_start, predicted_end, targets_start, targets_end)

            

            predicted_start = torch.softmax(predicted_start, dim = 1).cpu().detach().numpy()

            predicted_end = torch.softmax(predicted_end, dim = 1).cpu().detach().numpy()

            

            jaccard_scores = []



            for i, tweet in enumerate(orig_tweet):

                selected_tweet = orig_selected[i]

                tweet_sentiment = sentiment[i]

                jaccard_score, _ = calculate_jaccard_score(

                                    original_tweet = tweet,

                                    target_string = selected_tweet,

                                    sentiment_val = tweet_sentiment,

                                    idx_start = np.argmax(predicted_start[i, :]),

                                    idx_end = np.argmax(predicted_end[i, :]),

                                    offsets = offsets[i]

                                    )

                jaccard_scores.append(jaccard_score)



            jaccards.update(np.mean(jaccard_scores), ids.size(0))

            losses.update(loss.item(), ids.size(0))



            tk0.set_postfix(loss = losses.avg, jaccard = jaccards.avg)

    print(f"Jaccard = {jaccards.avg}")

    return jaccards.avg
def main(fold):

    df_train = train_df[train_df.kfolds != fold].reset_index(drop = True)

    df_val = train_df[train_df.kfolds == fold].reset_index(drop = True)

    train_dataset = TweetDataset(

                    tweet = df_train.text.values,

                    sentiment = df_train.sentiment.values,

                    selected_text = df_train.selected_text.values)

    train_dataloader = torch.utils.data.DataLoader(

                            train_dataset,

                            batch_size = TRAIN_BATCH_SIZE,

                            num_workers = 0

                        )

    valid_dataset = TweetDataset(

                    tweet = df_val.text.values,

                    sentiment = df_val.sentiment.values,

                    selected_text = df_val.selected_text.values)

    valid_dataloader = torch.utils.data.DataLoader(

                            valid_dataset,

                            batch_size = VALID_BATCH_SIZE,

                            num_workers = 0

                        )

    

    model_config = transformers.RobertaConfig.from_pretrained(ROBERTA_PATH)

    model_config.output_hidden_states = True

    model = TweetModel(conf=model_config)

    model.to(DEVICE)

    

    num_train_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

    ]

    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(

        optimizer, 

        num_warmup_steps=0, 

        num_training_steps=num_train_steps

    )



    es = EarlyStopping(patience=2, mode="max")

    print(f"Training is Starting for fold={fold}")

    

    for epoch in range(EPOCHS):

        train(train_dataloader, model, optimizer, DEVICE, scheduler=scheduler)

        jaccard = eval_fn(valid_dataloader, model, DEVICE)

        print(f"Jaccard Score = {jaccard}")

        es(jaccard, model, model_path=f"model_{fold}.bin")

        if es.early_stop:

            print("Early stopping")

            break

    
main(0)
main(1)
main(2)
main(3)
main(4)