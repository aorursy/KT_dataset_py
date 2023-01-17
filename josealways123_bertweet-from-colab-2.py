!pip install ../input/fairseq-and-fastbpe/sacrebleu-1.4.9-py3-none-any.whl

!pip install ../input/fairseq-and-fastbpe/fairseq-0.9.0-cp37-cp37m-linux_x86_64.whl

!pip install ../input/fairseq-and-fastbpe/fastBPE-0.1.0-cp37-cp37m-linux_x86_64.whl
import numpy as np

import pandas as pd

import os

import argparse

import warnings

import random

import torch 

from torch import nn

import torch.optim as optim

from sklearn.model_selection import StratifiedKFold

import tokenizers

from transformers import RobertaModel, RobertaConfig

warnings.filterwarnings('ignore')

seed=18
from nltk.tokenize import TweetTokenizer

from emoji import demojize

import re



tokenizer = TweetTokenizer()



def normalizeToken(token):

    lowercased_token = token.lower()

    if token.startswith("@"):

        return "@USER"

    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):

        return "HTTPURL"

    elif len(token) == 1:

        return demojize(token)

    else:

        if token == "’":

            return "'"

        elif token == "…":

            return "..."

        else:

            return token



def normalizeTweet(tweet):

    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))

    normTweet = " ".join([normalizeToken(token) for token in tokens])



    normTweet = normTweet.replace("cannot ", "can not ").replace("n't ", " n't ").replace("n 't ", " n't ").replace("ca n't", "can't").replace("ai n't", "ain't")

    normTweet = normTweet.replace("'m ", " 'm ").replace("'re ", " 're ").replace("'s ", " 's ").replace("'ll ", " 'll ").replace("'d ", " 'd ").replace("'ve ", " 've ")

    normTweet = normTweet.replace(" p . m .", "  p.m.") .replace(" p . m ", " p.m ").replace(" a . m .", " a.m.").replace(" a . m ", " a.m ")



    normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)

    normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)

    normTweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet)

    

    return " ".join(normTweet.split())
normalizeTweet(' I`d have responded, if I were going')
base_path='../input/bertweet-dataset'

config = RobertaConfig.from_pretrained(

    os.path.join(base_path,"BERTweet_base_transformers/config.json"), output_hidden_states=True)
from fairseq.data.encoders.fastbpe import fastBPE

from fairseq.data import Dictionary

args = argparse.Namespace(bpe_codes= os.path.join(base_path,"BERTweet_base_transformers/bpe.codes"))

bpe = fastBPE(args)

vocab = Dictionary()

vocab.add_from_file(os.path.join(base_path,"BERTweet_base_transformers/dict.txt"))
class TweetDataset(torch.utils.data.Dataset):

    def __init__(self, df, bpe, vocab, max_len=96):

        self.df = df

        self.labeled = 'selected_text' in df

        self.bpe = bpe

        self.vocab = vocab

        self.max_len=max_len

        

    def __getitem__(self, index):

        data={}

        row=self.df.iloc[index]

        #print(row.text)

        #print(row.selected_text)

        ids, masks, tweets_encoded = self.get_input_data(row)

        data['ids'] = ids

        data['masks'] = masks

        data['tweets_encoded'] = tweets_encoded

        data['tweet'] = row.text

        data['sentiment']=row.sentiment

        if self.labeled:

            data['selected_tweet'] = row.selected_text

            start_idx, end_idx = self.get_target_idx(row, tweets_encoded)

            data['start_idx'] = start_idx

            data['end_idx'] = end_idx

        return data

    

    def __len__(self):

        return len(self.df)

    

    def get_input_data(self, row):

        normalized_tweets = normalizeTweet(row.text)

        normalized_tweets = " " + " ".join(normalized_tweets.split())

        tweets_encoded = self.bpe.encode(normalized_tweets)

        encoding_ids = self.vocab.encode_line(tweets_encoded, append_eos=False, add_if_not_exist=False).long().tolist()

        sentiment_id = self.vocab.encode_line(self.bpe.encode(row.sentiment), append_eos=False, add_if_not_exist=False).long().tolist()

        ids = [0]+sentiment_id+[2,2]+encoding_ids+[2]

        

        pad_len = self.max_len-len(ids)

        if pad_len>0:

            ids += [1] * pad_len

        ids = torch.tensor(ids)

        masks = torch.where(ids!=1, torch.tensor(1), torch.tensor(0))

        

        return ids, masks, tweets_encoded

    

    def get_target_idx(self, row, tweets_encoded):

        normalized_selected_tweets = normalizeTweet(row.selected_text)

        normalized_selected_tweets = ' '+' '.join(normalized_selected_tweets.split())

        normalized_tweets = normalizeTweet(row.text)

        normalized_tweets = " " + " ".join(normalized_tweets.split())

        #print(normalized_selected_tweets)

        #print(normalized_tweets)

        

        len_st = len(normalized_selected_tweets) - 1

        idx0 = None

        idx1 = None

        for ind in (i for i, e in enumerate(normalized_tweets) if e == normalized_selected_tweets[1]):

            if " " + normalized_tweets[ind: ind+len_st] == normalized_selected_tweets:

                idx0 = ind

                idx1 = ind+len_st-1

                break

        if idx0==None and len(normalized_selected_tweets.split())>1:

            normalized_selected_tweets_1=' '+' '.join(normalized_selected_tweets.split()[1:])

            #print(normalized_selected_tweets_1)

            len_st_1 = len(normalized_selected_tweets_1) - 1

            for ind in (i for i, e in enumerate(normalized_tweets) if e == normalized_selected_tweets_1[1]):

                if " " + normalized_tweets[ind: ind+len_st_1] == normalized_selected_tweets_1:

                    idx0 = ind

                    idx1 = ind+len_st_1-1

                    break

        if idx0==None and len(normalized_selected_tweets.split())>1:

            normalized_selected_tweets_2=' '+' '.join(normalized_selected_tweets.split()[:-1])

            #print(normalized_selected_tweets_2)

            len_st_2 = len(normalized_selected_tweets_2) - 1

            for ind in (i for i, e in enumerate(normalized_tweets) if e == normalized_selected_tweets_2[1]):

                if " " + normalized_tweets[ind: ind+len_st_2] == normalized_selected_tweets_2:

                    idx0 = ind

                    idx1 = ind+len_st_2-1

                    break

        if idx0==None and len(normalized_selected_tweets.split())>1:

            normalized_selected_tweets_3=' '+' '.join(normalized_selected_tweets_2.split()[:-1])

            #print(normalized_selected_tweets_3)

            len_st_3 = len(normalized_selected_tweets_3) - 1

            for ind in (i for i, e in enumerate(normalized_tweets) if e == normalized_selected_tweets_3[1]):

                if " " + normalized_tweets[ind: ind+len_st_3] == normalized_selected_tweets_3:

                    idx0 = ind

                    idx1 = ind+len_st_3-1

                    break 

        sum_tot=-1

        flag = 0

        if idx0 != None and idx1 != None:

            for i, token in enumerate(tweets_encoded.split()):

                if '@@' not in token:

                    sum_tot += len(token)+1

                else:

                    sum_tot += len(token)-2

                if sum_tot>=idx0 and flag==0:

                    start_idx = i

                    flag = 1

                if sum_tot>=idx1:

                    end_idx = i

                    break

        if idx0==None or idx1==None:

            start_idx=0

            end_idx=0

        return start_idx+4, end_idx+4
def get_train_val_loaders(df, train_idx, val_idx, batch_size=32):

    train_df = df.iloc[train_idx]

    val_df = df.iloc[val_idx]

    

    train_loader = torch.utils.data.DataLoader(

        TweetDataset(train_df, bpe, vocab), 

        batch_size=batch_size, 

        shuffle=True,

        drop_last=False)



    val_loader = torch.utils.data.DataLoader(

        TweetDataset(val_df, bpe, vocab), 

        batch_size=batch_size, 

        shuffle=False, 

        num_workers=2)



    dataloaders_dict = {"train": train_loader, "val": val_loader}



    return dataloaders_dict
import torch.nn as nn

import torch.optim as optim



class BERTweetModel(nn.Module):

    def __init__(self, conf):

        super(BERTweetModel, self).__init__()

        self.roberta = RobertaModel.from_pretrained(os.path.join(base_path,"BERTweet_base_transformers/model.bin"),config=conf)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(conf.hidden_size*4,2)

        nn.init.xavier_uniform_(self.fc.weight)

        nn.init.normal_(self.fc.bias,0)

        

    def forward(self, input_ids, attention_mask):

        a, b, h = self.roberta(input_ids, attention_mask)

        x = torch.cat([h[-1],h[-2],h[-3], h[-4]],dim=-1)

        x = self.fc(self.dropout(x))

        start_logits, end_logits = x.split(1, -1)

        

        return start_logits.squeeze(-1), end_logits.squeeze(-1)
def loss_fn(start_logits, end_logits, start_positions, end_positions):

    ce_loss = nn.CrossEntropyLoss()

    start_loss = ce_loss(start_logits, start_positions)

    end_loss = ce_loss(end_logits, end_positions)    

    total_loss = start_loss + end_loss

    return total_loss
def get_selected_text(tweets_encoded, start_idx, end_idx):

    selected_text = ""

    for i, token in enumerate(tweets_encoded.split()[start_idx-4:end_idx-3]):

            token=' '+token

            selected_text+=token

    selected_text=re.sub('@@ ', '', selected_text)

    selected_text=re.sub('@@', '', selected_text)

    return selected_text

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



def compute_jaccard_score(tweets_encoded, start_idx, end_idx, start_logits, end_logits):

    start_pred = np.argmax(start_logits)

    end_pred = np.argmax(end_logits)

    #print('labels, outputs',start_idx, end_idx, start_pred, end_pred)    

    length = len(tweets_encoded.split())

    if start_pred<4:

        start_pred=4

    if end_pred>3+length:

        end_pred=3+length

    if start_pred > end_pred:

        start_pred=4

        end_pred=3+length

        pred = get_selected_text(tweets_encoded, start_pred, end_pred).strip()

    else:

        pred = get_selected_text(tweets_encoded, start_pred, end_pred).strip()    

    true = get_selected_text(tweets_encoded, start_idx, end_idx).strip()

    #print(true)

    #print(pred)

    

    return jaccard(true, pred)
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):

    if torch.cuda.is_available():

        model.cuda()

    

    loss_check=1000

    for epoch in range(num_epochs):

        for phase in ['train', 'val']:

            if phase == 'train':

                model.train()

            else:

                model.eval()



            epoch_loss = 0.0

            epoch_jaccard = 0.0

            count=0

            for data in (dataloaders_dict[phase]):

                if count%100==0:

                    print(count)

                count+=1

                ids = data['ids']

                masks = data['masks']

                tweets_encoded = data['tweets_encoded']

                

                selected_tweet = data['selected_tweet']

                start_idx = data['start_idx']

                end_idx = data['end_idx']

                #print(tweets_encoded[0])

                #print(tweets_encoded[1])



                if torch.cuda.is_available():

                  ids=ids.cuda()

                  masks=masks.cuda()

                  start_idx=start_idx.cuda()

                  end_idx=end_idx.cuda()



                optimizer.zero_grad()



                with torch.set_grad_enabled(phase == 'train'):



                    start_logits, end_logits = model(ids, masks)



                    loss = criterion(start_logits, end_logits, start_idx, end_idx)

                    

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                    epoch_loss += loss.item() * len(ids)

                    

                    start_idx = start_idx.cpu().detach().numpy()

                    end_idx = end_idx.cpu().detach().numpy()

                    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()

                    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

                    

                    for i in range(len(ids)): 

                        #print(selected_tweet[i])                       

                        jaccard_score = compute_jaccard_score(

                            tweets_encoded[i],

                            start_idx[i],

                            end_idx[i],

                            start_logits[i], 

                            end_logits[i])

                        epoch_jaccard += jaccard_score

                    

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)

            epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)

            

            print('Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}'.format(

                epoch + 1, num_epochs, phase, epoch_loss, epoch_jaccard))

        if epoch_loss<loss_check:

            loss_check=epoch_loss

            print("Saving model")

            torch.save(model.state_dict(), filename)

        elif epoch>1:

            print('Training stopping')

            break
num_epochs = 10

batch_size = 32

skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)
def run(fold):

    train_df = pd.read_csv('tweet-sentiment-extraction/train.csv').dropna().reset_index(drop=True)

    train_df['text'] = train_df['text'].astype(str)

    train_df['selected_text'] = train_df['selected_text'].astype(str)



    (train_idx, val_idx) = list(skf.split(train_df, train_df.sentiment))[fold]

    print(f'Fold: {fold}')

    model = BERTweetModel(conf=config)

    torch.save(model.state_dict(), f'drive/My Drive/Kaggle/check.pth')

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999))

    criterion = loss_fn    

    dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)

    print('starting training')

    train_model(

        model, 

        dataloaders_dict,

        criterion, 

        optimizer, 

        num_epochs,

        f'drive/My Drive/Kaggle/roberta_fold{fold}.pth')
def get_test_loader(df, batch_size=32):

    loader = torch.utils.data.DataLoader(

        TweetDataset(df, bpe, vocab), 

        batch_size=batch_size, 

        shuffle=False, 

        num_workers=2)    

    return loader
base_path='../input/bertweet-dataset'

config = RobertaConfig.from_pretrained(

    os.path.join(base_path,"BERTweet_base_transformers/config.json"), output_hidden_states=True)
def postprocessing(pred, tweet):

    pred_wo_spaces=''.join(pred.split())

    length = len(pred_wo_spaces)

    flag=0

    if tweet[-1]=='@':

        return tweet

    else:

        for index, value in enumerate(tweet):

            count=0

            letter=pred_wo_spaces[count]

            if value==letter:

                start_idx=index

                end_idx=index

                count+=1

                end_idx+=1

                while True:

                    if tweet[end_idx]==' ':

                        end_idx+=1

                    elif tweet[end_idx]=='!' and pred_wo_spaces[count]!='!':

                        end_idx+=1

                    elif tweet[end_idx]=='.' and pred_wo_spaces[count]!='.':

                        end_idx+=1

                    elif tweet[end_idx]=='*' and pred_wo_spaces[count]!='*':

                        end_idx+=1

                    elif tweet[end_idx]=='-' and pred_wo_spaces[count]!='-':

                        end_idx+=1

                    elif tweet[end_idx]=='?' and pred_wo_spaces[count]!='?':

                        end_idx+=1

                    elif tweet[end_idx]==pred_wo_spaces[count]:

                        end_idx+=1

                        count+=1

                    else:

                        break

                    if count==length:

                        flag=1

                        break

                if flag==1:

                    break

        if flag==1:

            #print(tweet[start_idx:end_idx])

            while start_idx>0:

                if tweet[start_idx-1]==' ':

                    break

                else:

                    print('start_idx_changed')

                    start_idx=start_idx-1

            while end_idx<len(tweet)-1:

                if tweet[end_idx]==' ':

                    break

                else:

                    print('end_idx_changed')

                    end_idx=end_idx+1

            #print(tweet[start_idx:end_idx])

            return tweet[start_idx:end_idx]

        elif 'HTTPURL' in pred.split():

            return tweet

        elif '@USER' in pred.split():

            return tweet

        else:

            print('No match found')

            print(pred)

            print(tweet)

            return tweet
test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

test_df['text'] = test_df['text'].astype(str)

#test_df['selected_text'] = test_df['selected_text'].astype(str)



test_loader = get_test_loader(test_df)

predictions = []

models = []

for fold in range(skf.n_splits):

    model = BERTweetModel(conf=config)

    if torch.cuda.is_available():

        model.cuda()

    model.load_state_dict(torch.load(f'../input/mosh1-data-orig/roberta_fold{fold}.pth',map_location=torch.device('cpu')))

    model.eval()

    models.append(model)

count=0

count_real=0

jaccard_total=0

for data in test_loader:

    #print(count)

    ids = data['ids']

    masks = data['masks']

    tweets_encoded = data['tweets_encoded']

    tweet = data['tweet']

    sentiment = data['sentiment']

    

    #selected_tweet = data['selected_tweet']



    if torch.cuda.is_available():

        ids=ids.cuda()

        masks=masks.cuda()



    start_logits = []

    end_logits = []

    for model in models:

        with torch.no_grad():

            output = model(ids, masks)

            start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())

            end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())

    

    start_logits = np.mean(start_logits, axis=0)

    end_logits = np.mean(end_logits, axis=0)

    for i in range(len(ids)):  

        count_real+=1

        #print('Coount', count_real)

        start_pred = np.argmax(start_logits[i])

        end_pred = np.argmax(end_logits[i])

        #print(tweet[i].strip())

        #print(tweets_encoded[i])

        length = len(tweets_encoded[i].split())

        if start_pred<4:

            start_pred=4

        if end_pred>3+length:

            end_pred=3+length

        if start_pred > end_pred:

            start_pred=4

            end_pred=3+length

            pred = get_selected_text(tweets_encoded[i], start_pred, end_pred).strip()

        else:

            pred = get_selected_text(tweets_encoded[i], start_pred, end_pred).strip()    

        #print(pred,'\n')

        try:

            pred=postprocessing(pred, tweet[i].strip())

        except:

            print('Error')

            print(tweet[i].strip())

            print(pred)

            pred=tweet[i]

        

        #this if loop is only for testing purposes on train data

        #if pred.strip()!=selected_tweet[i].strip():

            #count+=1

            #print(count)

            #print(sentiment[i]) 

            #print(tweet[i].strip())

            #print(selected_tweet[i].strip())

            #print(pred.strip())

            #jaccard_total+=jaccard(selected_tweet[i].strip(),pred.strip())

        #this above if loop only for testing purposes

        

        predictions.append(pred)

        

#print('Count', count)

#print('Average jaccard', jaccard_total/count)
sub_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')

sub_df['selected_text'] = predictions

sub_df.to_csv('submission.csv', index=False)

sub_df.head()