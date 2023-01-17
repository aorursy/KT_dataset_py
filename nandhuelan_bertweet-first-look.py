!pip install fairseq

!pip install fastBPE
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import torch

import argparse

from sklearn import model_selection

import tokenizers



from transformers import RobertaConfig

from transformers import RobertaModel





import torch.nn as nn

import torch.nn.functional as F

from torch.optim import lr_scheduler



from sklearn import model_selection

from sklearn import metrics

import transformers

import tokenizers

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup

from tqdm.autonotebook import tqdm

import random

import gc



from types import SimpleNamespace

from fairseq.data.encoders.fastbpe import fastBPE

from fairseq.data import Dictionary

from fairseq import options  



path='../input/tweet-sentiment-extraction/'

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")

df=df.dropna()



df["kfold"] = -1



df = df.sample(frac=1).reset_index(drop=True)



kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)





for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.sentiment.values)):

    print(len(train_idx), len(val_idx))

    df.loc[val_idx, 'kfold'] = fold





df.to_csv("train_folds.csv", index=False)
!tail train_folds.csv
parser = options.get_preprocessing_parser()  

parser.add_argument('--bpe-codes', type=str,default="../input/bertweet-base-transformers/bpe.codes")  
class BERTweetTokenizer():

    

    def __init__(self,pretrained_path = '../input/bertweet-base-transformers/',parser=parser):

        



        self.bpe = fastBPE(args=parser.parse_args(args=[]))

        self.vocab = Dictionary()

        self.vocab.add_from_file(pretrained_path + "dict.txt")

        self.cls_token_id = 0

        self.pad_token_id = 1

        self.sep_token_id = 2

        self.pad_token = '<pad>'

        self.cls_token = '<s> '

        self.sep_token = ' </s>'

        

    def bpe_encode(self,text):

        return self.bpe.encode(text)

    

    def encode(self,text,add_special_tokens=False):

        subwords = self.cls_token + self.bpe.encode(text) + self.sep_token

        input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=True).long().tolist()

        return input_ids

    

    def tokenize(self,text):

        return self.bpe_encode(text).split()

    

    def convert_tokens_to_ids(self,tokens):

        input_ids = self.vocab.encode_line(' '.join(tokens), append_eos=False, add_if_not_exist=False).long().tolist()

        return input_ids



    

    def decode_id(self,id):

        return self.vocab.string(id, bpe_symbol = '@@')
class config:

    MAX_LEN = 128

    TRAIN_BATCH_SIZE = 64

    VALID_BATCH_SIZE = 16

    EPOCHS = 5

    config = RobertaConfig.from_pretrained(

    "../input/bertweet-base-transformers/config.json")

    config.output_hidden_states = True



    BERTweet = RobertaModel.from_pretrained(

    "../input/bertweet-base-transformers/model.bin",

    config=config)

    BERTweetpath="../input/bertweet-base-transformers/"

    TRAINING_FILE = "train_folds.csv"

    TOKENIZER = BERTweetTokenizer('../input/bertweet-base-transformers/',parser=parser)
config.TOKENIZER.encode('positive negative neutral'),config.TOKENIZER.decode_id([1809,4])
def process_data(tweet, selected_text, sentiment, tokenizer, max_len):

    

    tweet_orig = " ".join(str(tweet).split())

    selected_text = " ".join(str(selected_text).split())



    len_st = len(selected_text)





    idx = tweet_orig.find(selected_text)

    char_targets = np.zeros((len(tweet_orig)))

    char_targets[idx:idx+len(selected_text)]=1



    tok_tweet = config.TOKENIZER.encode(tweet_orig)



    # Convert into torch tensor

    all_input_ids = torch.tensor([tok_tweet], dtype=torch.long)



    tok_tweet=tok_tweet[1:-1]



    # ID_OFFSETS

    offsets = []; idx=0

    

    try:

        for t in tok_tweet:

            ix=0

            w = config.TOKENIZER.decode_id([t])



            #print("==",w,len(w))

            if tweet[tweet.find(w)-1]==' ':   #to consider spaces in the offsets

                offsets.append((idx+1,idx+1+len(w)))

                idx =idx+1+ len(w)

                ix=ix+1+ len(w)

            else:

                offsets.append((idx,idx+len(w)))

                idx += len(w)

                ix+=len(w)



            tweet=tweet[ix:]

    except:

        print("***",tweet_orig)

        pass



    input_ids_orig = tok_tweet

    tweet_offsets = offsets





    target_idx = []

    for j, (offset1, offset2) in enumerate(tweet_offsets):

        if np.sum(char_targets[offset1:offset2])> 0:

            target_idx.append(j)



    if  len(target_idx)>0:

        targets_start = target_idx[0]

        targets_end = target_idx[-1]



    else:

        targets_start = 0

        targets_end= len(char_targets)





    #print(targets_start,targets_end)    

    sentiment_id = {

        'positive': 1809,

        'negative': 3392,

        'neutral': 14058

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

        'orig_tweet': tweet_orig,

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
class TweetModel(transformers.BertPreTrainedModel):

    def __init__(self, conf):

        super(TweetModel, self).__init__(conf)

        self.roberta = RobertaModel.from_pretrained("../input/bertweet-base-transformers/model.bin",config=conf)

        

        self.drop_out = nn.Dropout(0.1)

        self.Cov1S = nn.Conv1d(768 , 128 , kernel_size = 2 ,stride = 1 )

        self.Cov1E = nn.Conv1d(768, 128, kernel_size = 2 ,stride = 1 )

        self.Cov2S = nn.Conv1d(128 , 64 , kernel_size = 2 ,stride = 1)

        self.Cov2E = nn.Conv1d(128 , 64 , kernel_size = 2 ,stride = 1)

        self.lS = nn.Linear(64 , 1)

        self.lE = nn.Linear(64 , 1)

        

        #self.lstm = nn.LSTM(1536,768,1, batch_first=True)

        

        torch.nn.init.normal_(self.lS.weight, std=0.02)

        torch.nn.init.normal_(self.lE.weight, std=0.02)

    

    def forward(self, ids, mask, token_type_ids):

        _, _, out = self.roberta(

            ids,

            attention_mask=mask,

            token_type_ids=token_type_ids

        )



        #out = torch.cat((out[-1], out[-2]), dim=-1)

        

        out = self.drop_out(out[-1])

        out = out.permute(0,2,1)     #m,768,192

        

        same_pad1 = torch.zeros(out.shape[0] , 768 , 1).cuda()  #m,768,1

        same_pad2 = torch.zeros(out.shape[0] , 128 , 1).cuda()    #m,128,1



        out1 = torch.cat((same_pad1 , out), dim = 2)   #m,768,193

        

        

        out1 = self.Cov1S(out1)              #m,128,192

        

        out1 = torch.cat((same_pad2 , out1), dim = 2)   #m,128,193

        out1 = self.Cov2S(out1)     #m,64,192

        out1 = F.leaky_relu(out1)

        out1 = out1.permute(0,2,1)  #m,192,64

        

        

        start_logits = self.lS(out1).squeeze(-1) #m,192

        #print(start_logits.shape)



        out2 = torch.cat((same_pad1 , out), dim = 2)

        out2 = self.Cov1E(out2)

        out2 = torch.cat((same_pad2 , out2), dim = 2)

        out2 = self.Cov2E(out2)

        out2 = F.leaky_relu(out2)

        out2 = out2.permute(0,2,1)

        end_logits = self.lE(out2).squeeze(-1)



        return start_logits, end_logits
def dist_between(start_logits, end_logits, device='cuda', max_seq_len=128):

    """get dist btw. pred & ground_truth"""



    linear_func = torch.tensor(np.linspace(0, 1, max_seq_len, endpoint=False), requires_grad=False)

    linear_func = linear_func.to(device)

    assert start_logits.shape == start_logits.shape



    start_pos = (start_logits*linear_func).sum(axis=1)

    end_pos = (end_logits*linear_func).sum(axis=1)



    diff = end_pos-start_pos



    return diff.sum(axis=0)/diff.size(0)





def dist_loss_fn(start_logits, end_logits, start_positions, end_positions, device, max_seq_len=128, scale=1):

    """

    calculate distance loss between prediction's length & GT's length

    

    Input

    - start_logits ; shape (batch, max_seq_len{128})

        - logits for start index

    - end_logits

        - logits for end index

    - start_positions ; shape (batch, 1)

        - start index for GT

    - end_positions

        - end index for GT

    """

    start_logits = torch.nn.Softmax(1)(start_logits) # shape ; (batch, max_seq_len)

    end_logits = torch.nn.Softmax(1)(end_logits)

    

    # one hot encoding for GT (start_positions, end_positions)

    start_logits_gt = torch.zeros([len(start_positions), max_seq_len], requires_grad=False).to(device)

    end_logits_gt = torch.zeros([len(end_positions), max_seq_len], requires_grad=False).to(device)

    for idx, _ in enumerate(start_positions):

        _start = start_positions[idx]

        _end = end_positions[idx]

        start_logits_gt[idx][_start] = 1

        end_logits_gt[idx][_end] = 1



    pred_dist = dist_between(start_logits, end_logits, device, max_seq_len)

    gt_dist = dist_between(start_logits_gt, end_logits_gt, device, max_seq_len) # always positive

    diff = (gt_dist-pred_dist)



    rev_diff_squared = 1-torch.sqrt(diff*diff) # as diff is smaller, make it get closer to the one

    loss = -torch.log(rev_diff_squared) # by using negative log function, if argument is near zero -> inifinite, near one -> zero



    return loss*scale
def loss_fn(start_logits, end_logits, start_positions, end_positions,device):

    loss_fct = nn.CrossEntropyLoss()

    

    start_loss = loss_fct(start_logits, start_positions)

    end_loss = loss_fct(end_logits, end_positions)

    

    dist_loss = dist_loss_fn(start_logits, end_logits,start_positions, end_positions,device, config.MAX_LEN) 

    

    total_loss = (start_loss + end_loss)

    

    return total_loss+dist_loss



def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))





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
def train_fn(data_loader, model, optimizer, device, scheduler=None):

    

    model.train()

    losses = AverageMeter()

    jaccards = AverageMeter()



    tk0 = tqdm(data_loader, total=len(data_loader))

    

    for bi, d in enumerate(tk0):

        #print("(())",bi,d['orig_selected'])

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

        

        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end,device)

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
def eval_fn(data_loader, model, device):

    model.eval()

    losses = AverageMeter()

    jaccards = AverageMeter()

    

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

            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end,device)

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
def run(fold,seed=None):

    dfx = pd.read_csv(config.TRAINING_FILE)

    

    dfx['text']=dfx['text'].apply(lambda x: x.strip())

    dfx['selected_text']=dfx['selected_text'].apply(lambda x: x.strip())

    

    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)

    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    

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

#     model_config = config.config

#     model_config.output_hidden_states = True

    model = TweetModel(config.config)

    model.to(device)



    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

    ]

    optimizer = AdamW(optimizer_parameters, lr=4e-5)

    scheduler = get_linear_schedule_with_warmup(

        optimizer, 

        num_warmup_steps=0, 

        num_training_steps=num_train_steps

    )



    es = EarlyStopping(patience=3, mode="max")

    print(f"Training is Starting for fold={fold}")

    

    # I'm training only for 3 epochs even though I specified 5!!!

    for epoch in range(config.EPOCHS):

        train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler)

        jaccard = eval_fn(valid_data_loader, model, device)

        print(f"Jaccard Score = {jaccard}")

        es(jaccard, model, model_path=f"model_{fold}.bin")

        

        if es.early_stop:

            print("Early stopping")

            break
run(fold=0)

run(fold=1)

run(fold=2)

run(fold=3)

run(fold=4)
gc.collect()
# def process_data(tweet, selected_text, sentiment, tokenizer, max_len):

    

#     tweet_orig = " ".join(str(tweet).split())

#     selected_text = " ".join(str(selected_text).split())



#     len_st = len(selected_text)





#     idx = tweet_orig.find(selected_text)

#     char_targets = np.zeros((len(tweet_orig)))

#     char_targets[idx:idx+len(selected_text)]=1



#     tok_tweet = config.TOKENIZER.encode(tweet_orig)



#     # Convert into torch tensor

#     all_input_ids = torch.tensor([tok_tweet], dtype=torch.long)



#     tok_tweet=tok_tweet[1:-1]



#     # ID_OFFSETS

#     offsets = []; idx=0

    

#     try:

#         for t in tok_tweet:

#             ix=0

#             w = config.TOKENIZER.decode_id([t])



#             print("==",w,len(w))

#             if tweet[tweet.find(w)-1]==' ':

#                 offsets.append((idx+1,idx+1+len(w)))

#                 idx =idx+1+ len(w)

#                 ix=ix+1+ len(w)

#             else:

                

#                 offsets.append((idx,idx+len(w)))

#                 idx += len(w)

#                 ix+=len(w)



#             tweet=tweet[ix:]

#     except:

#         print("***",tweet_orig)

        



#     input_ids_orig = tok_tweet

#     tweet_offsets = offsets





#     target_idx = []

#     for j, (offset1, offset2) in enumerate(tweet_offsets):

#         if np.sum(char_targets[offset1:offset2])> 0:

#             target_idx.append(j)



#     if  len(target_idx)>0:

#         targets_start = target_idx[0]

#         targets_end = target_idx[-1]



#     else:

#         targets_start = 0

#         targets_end= len(char_targets)





#     #print(targets_start,targets_end)    

#     sentiment_id = {

#         'positive': 1809,

#         'negative': 3392,

#         'neutral': 14058

#     }



#     input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]

#     token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)

#     mask = [1] * len(token_type_ids)

#     tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]

#     targets_start += 4

#     targets_end += 4



#     padding_length = max_len - len(input_ids)

#     if padding_length > 0:

#         input_ids = input_ids + ([1] * padding_length)

#         mask = mask + ([0] * padding_length)

#         token_type_ids = token_type_ids + ([0] * padding_length)

#         tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

    

#     return {

#         'ids': input_ids,

#         'mask': mask,

#         'token_type_ids': token_type_ids,

#         'targets_start': targets_start,

#         'targets_end': targets_end,

#         'orig_tweet': tweet_orig,

#         'orig_selected': selected_text,

#         'sentiment': sentiment,

#         'offsets': tweet_offsets

#     }
# dfx = pd.read_csv(config.TRAINING_FILE)

# #dfx['text']=dfx['text'].apply(lambda x: x.strip())

# dfx=dfx[dfx['textID']=="77e645b46c"]

# dfx

# #,dfx[dfx['selected_text']=='found any decently priced breakfast yet? i hope you do']['selected_text']
#a=process_data(dfx.loc[287]['text'],dfx.loc[287]['selected_text'],dfx.loc[287]['sentiment'],config.TOKENIZER,config.MAX_LEN)
# filtered_output  = ""

# for ix in range(a['targets_start'], a['targets_end'] + 1):

#     filtered_output += a['orig_tweet'][a['offsets'][ix][0]: a['offsets'][ix][1]]

#     if (ix+1) < len(a['offsets']) and a['offsets'][ix][1] < a['offsets'][ix+1][0]:

#         filtered_output += " "

        

# filtered_output