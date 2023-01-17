import os

import torch

import pandas as pd

import torch.nn as nn

import numpy as np

import torch.nn.functional as F

from torch.optim import lr_scheduler

from sklearn import model_selection

from sklearn.model_selection import KFold

from sklearn import metrics

from transformers import RobertaModel, RobertaConfig

import tokenizers

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup

from tqdm.autonotebook import tqdm

import utils
class config:

    MAX_LEN = 128

    TRAIN_BATCH_SIZE = 64

    VALID_BATCH_SIZE = 16

    EPOCHS = 3

    MODEL_PATH = "../input/roberta-base/pytorch_model.bin"

    TRAINING_FILE = "../input/tweet-sentiment-extraction/train.csv"

    TOKENIZER = tokenizers.ByteLevelBPETokenizer(

            vocab_file='../input/roberta-base/vocab.json', 

            merges_file='../input/roberta-base/merges.txt', 

            lowercase=True,

            add_prefix_space=True)





def process_data(tweet, selected_text, sentiment, tokenizer, max_len):

    tweet = " " + " ".join(str(tweet).split())

    selected_text = " " + " ".join(str(selected_text).split())



    len_st = len(selected_text) - 1

    idx0 = None

    idx1 = None



    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):

        if " " + tweet[ind: ind+len_st] == selected_text:

            idx0 = ind

            idx1 = ind + len_st - 1

            break



    char_targets = [0] * len(tweet)

    if idx0 != None and idx1 != None:

        for ct in range(idx0, idx1 + 1):

            char_targets[ct] = 1

    

    tok_tweet = tokenizer.encode(tweet)

    input_ids_orig = tok_tweet.ids

    tweet_offsets = tok_tweet.offsets

    

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

    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)

    mask = [1] * len(token_type_ids)

    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]

    targets_start += 4

    targets_end += 4



    padding_length = max_len - len(input_ids)

    if padding_length > 0:

        input_ids = input_ids + ([1] * padding_length)

        mask = mask + ([0] * padding_length)

        token_type_ids = token_type_ids + ([0] * padding_length)

        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

    

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



class TweetDataset:

    def __init__(self, tweet, sentiment, selected_text):

        self.tweet = tweet

        self.sentiment = sentiment

        self.selected_text = selected_text

        self.tokenizer = config.TOKENIZER

        self.max_len = config.MAX_LEN

    

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

            'ids': torch.tensor(data["ids"], dtype=torch.long),

            'mask': torch.tensor(data["mask"], dtype=torch.long),

            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),

            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),

            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),

            'orig_tweet': data["orig_tweet"],

            'orig_selected': data["orig_selected"],

            'sentiment': data["sentiment"],

            'offsets': torch.tensor(data["offsets"], dtype=torch.long)

        }



class TweetModel(nn.Module):

    def __init__(self):

        super(TweetModel, self).__init__()

        config = RobertaConfig.from_pretrained(

            '../input/roberta-base/config.json', output_hidden_states=True)    

        self.roberta = RobertaModel.from_pretrained(

            '../input/roberta-base/pytorch_model.bin', config=config)

        self.drop_out = nn.Dropout(0.2)

        self.l0 = nn.Linear(768 * 2, 2)

        torch.nn.init.normal_(self.l0.weight, std=0.02)

        

    

    def forward(self, ids, mask, token_type_ids):

        _, _, out = self.roberta(

            ids,

            attention_mask=mask,

            token_type_ids=token_type_ids

        )



        out = torch.cat((out[-1], out[-2]), dim=-1)

        out = self.drop_out(out)

        logits = self.l0(out)



        start_logits, end_logits = logits.split(1, dim=-1)



        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)



        return start_logits, end_logits

    

def loss_fn(start_logits, end_logits, start_positions, end_positions):

    loss_fct = nn.CrossEntropyLoss()

    start_loss = loss_fct(start_logits, start_positions)

    end_loss = loss_fct(end_logits, end_positions)

    total_loss = (start_loss + end_loss)

    return total_loss

    

    

def train_fn(data_loader, model, optimizer, device, scheduler=None):

    model.train()

    losses = utils.AverageMeter()

    jaccards = utils.AverageMeter()



    tk0 = tqdm(data_loader, total=len(data_loader))

    

    for bi, d in enumerate(tk0):



        ids = d["ids"]

        token_type_ids = d["token_type_ids"]

        mask = d["mask"]

        targets_start = d["targets_start"]

        targets_end = d["targets_end"]

        sentiment = d["sentiment"]

        orig_selected = d["orig_selected"]

        orig_tweet = d["orig_tweet"]

        targets_start = d["targets_start"]

        targets_end = d["targets_end"]

        offsets = d["offsets"]



        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets_start = targets_start.to(device, dtype=torch.long)

        targets_end = targets_end.to(device, dtype=torch.long)



        model.zero_grad()

        outputs_start, outputs_end = model(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids,

        )

        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)

        loss.backward()

        optimizer.step()

        scheduler.step()



        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()

        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

        jaccard_scores = []

        for px, tweet in enumerate(orig_tweet):

            selected_tweet = orig_selected[px]

            tweet_sentiment = sentiment[px]

            jaccard_score, _ = calculate_jaccard_score(

                original_tweet=tweet,

                target_string=selected_tweet,

                sentiment_val=tweet_sentiment,

                idx_start=np.argmax(outputs_start[px, :]),

                idx_end=np.argmax(outputs_end[px, :]),

                offsets=offsets[px]

            )

            jaccard_scores.append(jaccard_score)



        jaccards.update(np.mean(jaccard_scores), ids.size(0))

        losses.update(loss.item(), ids.size(0))

        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

        

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



    jac = utils.jaccard(target_string.strip(), filtered_output.strip())

    return jac, filtered_output





def eval_fn(data_loader, model, device):

    model.eval()

    losses = utils.AverageMeter()

    jaccards = utils.AverageMeter()

    

    with torch.no_grad():

        tk0 = tqdm(data_loader, total=len(data_loader))

        for bi, d in enumerate(tk0):

            ids = d["ids"]

            token_type_ids = d["token_type_ids"]

            mask = d["mask"]

            sentiment = d["sentiment"]

            orig_selected = d["orig_selected"]

            orig_tweet = d["orig_tweet"]

            targets_start = d["targets_start"]

            targets_end = d["targets_end"]

            offsets = d["offsets"].numpy()



            ids = ids.to(device, dtype=torch.long)

            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            targets_start = targets_start.to(device, dtype=torch.long)

            targets_end = targets_end.to(device, dtype=torch.long)



            outputs_start, outputs_end = model(

                ids=ids,

                mask=mask,

                token_type_ids=token_type_ids

            )

            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)

            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()

            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

            jaccard_scores = []

            for px, tweet in enumerate(orig_tweet):

                selected_tweet = orig_selected[px]

                tweet_sentiment = sentiment[px]

                jaccard_score, _ = calculate_jaccard_score(

                    original_tweet=tweet,

                    target_string=selected_tweet,

                    sentiment_val=tweet_sentiment,

                    idx_start=np.argmax(outputs_start[px, :]),

                    idx_end=np.argmax(outputs_end[px, :]),

                    offsets=offsets[px]

                )

                jaccard_scores.append(jaccard_score)



            jaccards.update(np.mean(jaccard_scores), ids.size(0))

            losses.update(loss.item(), ids.size(0))

            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

    

    print(f"Jaccard = {jaccards.avg}")

    return jaccards.avg
def run(fold):

    dfx = pd.read_csv(config.TRAINING_FILE)

    

    kf = KFold(n_splits = 10, shuffle = True, random_state = 42)

    kf_r = next(kf.split(dfx), None)



    df_train = dfx.iloc[kf_r[0]].reset_index(drop=True)

    df_valid = dfx.iloc[kf_r[1]].reset_index(drop=True)

    

    train_dataset = TweetDataset(

        tweet=df_train.text.values,

        sentiment=df_train.sentiment.values,

        selected_text=df_train.selected_text.values

    )



    train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=config.TRAIN_BATCH_SIZE,

        num_workers=4

    )



    valid_dataset = TweetDataset(

        tweet=df_valid.text.values,

        sentiment=df_valid.sentiment.values,

        selected_text=df_valid.selected_text.values

    )



    valid_data_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=config.VALID_BATCH_SIZE,

        num_workers=2

    )



    device = torch.device("cuda")

    

    model = TweetModel()

    model.to(device)



    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

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



    es = utils.EarlyStopping(patience=2, mode="max")

    

    

    for epoch in range(3):

        train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler)

        jaccard = eval_fn(valid_data_loader, model, device)

        print(f"Jaccard Score = {jaccard}")

        es(jaccard, model, model_path=f"modelo_{fold}.bin")

        if es.early_stop:

            print("Early stopping")

            break
for fold in range(3):

    print(f"Fold={fold}")

    run(fold)
device = torch.device("cuda")



modelo1 = TweetModel()

modelo1.to(device)

modelo1.load_state_dict(torch.load("modelo_0.bin"))

modelo1.eval()



modelo2 = TweetModel()

modelo2.to(device)

modelo2.load_state_dict(torch.load("modelo_1.bin"))

modelo2.eval()



modelo3 = TweetModel()

modelo3.to(device)

modelo3.load_state_dict(torch.load("modelo_2.bin"))

modelo3.eval()

final_output = []

df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

df_test.loc[:, "selected_text"] = df_test.text.values
test_dataset = TweetDataset(

        tweet=df_test.text.values,

        sentiment=df_test.sentiment.values,

        selected_text=df_test.selected_text.values

    )



data_loader = torch.utils.data.DataLoader(

    test_dataset,

    shuffle=False,

    batch_size=config.VALID_BATCH_SIZE,

    num_workers=1

)





with torch.no_grad():

    tk0 = tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):

        ids = d["ids"]

        token_type_ids = d["token_type_ids"]

        mask = d["mask"]

        sentiment = d["sentiment"]

        orig_selected = d["orig_selected"]

        orig_tweet = d["orig_tweet"]

        targets_start = d["targets_start"]

        targets_end = d["targets_end"]

        offsets = d["offsets"].numpy()



        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets_start = targets_start.to(device, dtype=torch.long)

        targets_end = targets_end.to(device, dtype=torch.long)



        outputs_start1, outputs_end1 = modelo1(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )

        

        outputs_start2, outputs_end2 = modelo2(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )

        

        outputs_start3, outputs_end3 = modelo3(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )

        

        

        outputs_start = (outputs_start1 + outputs_start2 + outputs_start3 ) / 3

        outputs_end = (outputs_end1 + outputs_end2 + outputs_end3 ) / 3

        

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()

        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

        jaccard_scores = []

        for px, tweet in enumerate(orig_tweet):

            selected_tweet = orig_selected[px]

            tweet_sentiment = sentiment[px]

            _, output_sentence = calculate_jaccard_score(

                original_tweet=tweet,

                target_string=selected_tweet,

                sentiment_val=tweet_sentiment,

                idx_start=np.argmax(outputs_start[px, :]),

                idx_end=np.argmax(outputs_end[px, :]),

                offsets=offsets[px]

            )

            final_output.append(output_sentence)

            
submission = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

submission.loc[:, 'selected_text'] = final_output

submission.to_csv("submission.csv", index=False)
submission.head()