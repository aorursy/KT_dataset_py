# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import transformers

# from transformers import AlbertTokenizer, AlbertModel

import torch

import torch.nn as nn

from torch.utils.data import DataLoader

import math



import plotly.graph_objects as go

from plotly.subplots import make_subplots



from tqdm import tqdm_notebook

import re

import os, sys

import random
def seed_everything(seed_value):

    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = True



seed = 42

seed_everything(seed)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
train.head()
train.isnull().sum()
print(len(train))

train = train.dropna()

print(len(train))
def compute_jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    if (len(a)==0) & (len(b)==0): 

        return 0.5

    else:

        return round(float(len(c)) / (len(a) + len(b) - len(c)), 2)
compute_jaccard(train.text[3], train.selected_text[3])
# Add histogram data

dist_score_pos = []

dist_score_neg = []

dist_score_neu = []



group_labels = ['positive', 'negative', 'neutral']



for label in group_labels:

    for t, s_t in zip(train.text[train.sentiment == label], train.selected_text[train.sentiment == label]):

        score = compute_jaccard(t, s_t)

        if label == 'positive':

            dist_score_pos.append(score)

        elif label == 'negative':

            dist_score_neg.append(score)

        else:

            dist_score_neu.append(score)



# Group data together

hist_data = [dist_score_pos, dist_score_neg, dist_score_neu]



fig = make_subplots(rows=3, cols=1)

rows_c = 1

for h_data, g_label in zip(hist_data, group_labels):

    fig.append_trace(go.Histogram(x=h_data, name=g_label), rows_c, 1)

    rows_c += 1



fig.update_layout(title='Distribution jaccard score each sentiment')

fig.show()
fig = go.Figure()

rows_c = 1

for h_data, g_label in zip(hist_data, group_labels):

    if rows_c == 3:

        continue

    fig.add_trace(go.Histogram(x=h_data, name=g_label))

    rows_c += 1



# Overlay both histograms

fig.update_layout(barmode='overlay', title='Distribution jaccard score in positive and negative sentiment')

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.5)

fig.show()
dist_score_neu.count(1) / len(dist_score_neu)
# use this command if u want to download sentencepiece model from kernel

# !wget -O "/kaggle/working/albert-base-v2-spiece.model" "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-spiece.model"

# !wget -O "/kaggle/working/sentencepiece_pb2.py" "https://raw.githubusercontent.com/google/sentencepiece/master/python/sentencepiece_pb2.py"

# and use import sentencepiece_pb2.py to generate offsets / spans

# !wget https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb
sys.path.append('/kaggle/input/sentencepiecepb2')

import sentencepiece as spm

import sentencepiece_pb2
path_spiece_model = '/kaggle/input/albert-model/albert-base-v2-spiece.model'

path_albert_config = '/kaggle/input/albert-model/albert-base-v2-config.json'

path_albert_model = '/kaggle/input/albert-model/albert-base-v2-pytorch_model.bin'
sp = spm.SentencePieceProcessor()

sp.load(path_spiece_model)
tokenizer = transformers.AlbertTokenizer.from_pretrained(path_spiece_model, do_lower_case=True)

tokenizer.tokenize("Test tokenizer")
sp.encode_as_pieces("Test tokenizer".lower())
class OffsetTokenizer():

    def __init__(self, path_model=path_spiece_model):

        self.spt = sentencepiece_pb2.SentencePieceText()

        self.sp = spm.SentencePieceProcessor()

        self.sp.load(path_model)

        

    def encode(self, text, return_tokens=False, lower=True):

        if lower:

            text = text.lower()

        offset = []

        ids = []

        self.spt.ParseFromString(self.sp.encode_as_serialized_proto(text))

        

        for piece in self.spt.pieces:

            offset.append((piece.begin, piece.end))

            ids.append(piece.id)

            

        if return_tokens:

            return sp.encode_as_pieces(text), ids, offset

        else:

            return ids, offset

    
o_tokenizer = OffsetTokenizer()

o_tokenizer.encode("Test tokenizer", return_tokens=False)
# spt = sentencepiece_pb2.SentencePieceText()

# spt.ParseFromString(sp.encode_as_serialized_proto(text))

# a = tokenizer.encode(text, add_special_tokens=False)

# b = tokenizer.tokenize(text)
albert_config = transformers.AlbertConfig.from_pretrained(path_albert_config)

albert_config.output_hidden_states=True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def pad_or_truncate(list_num, max_len, SEP_ID, is_input_ids=False):

    if len(list_num) < max_len:

        list_num = list_num + [0] * (max_len - len(list_num))

    else:

        if is_input_ids:

            list_num = list_num[:max_len-1] + SEP_ID

        else:

            return list_num[:max_len]

    return list_num
def issubset_sequence(s_text_ids, text_ids):

    s_text_ids = [str(i) for i in s_text_ids]

    ptn_s_text_ids = r'\b' + r'\b \b'.join(s_text_ids) + r'\b'

    text_ids = [str(i) for i in text_ids]

    text_ids = ' '.join(text_ids)

    

    if re.search(ptn_s_text_ids, text_ids):

        return True

    else:

        return False
def data_extractor(df, data_test=False, max_len=128, random=True):

    """

    Processes the tweet and outputs the features necessary for model training and inference

    """

    count = 0

    # ALBERT ids

    sentiment_id = {'positive' : 2221,

                    'neutral' : 8387,

                    'negative' : 3682}

    CLS_ID = tokenizer.encode('[CLS]', add_special_tokens=False)

    SEP_ID = tokenizer.encode('[SEP]', add_special_tokens=False)

    

    all_data = []

    if random:

        df = df.sample(frac=1, random_state=42)

    texts = df.text.tolist()

    sentiments = df.sentiment.tolist()

    

    if not data_test:

        sel_texts = df.selected_text.tolist()

        for idx, row in df.iterrows():

            start_target, end_target = 0, 0

            text = ' '+ ' '.join( row.text.lower().split() )

            s_text = ' '+ ' '.join( row.selected_text.lower().split() )

            sentiment = row.sentiment.lower().strip()

            text_ids, offsets = o_tokenizer.encode(text)

            s_text_ids, _ = o_tokenizer.encode(s_text)



            # fixing bug selection text

            for i in range(len(s_text_ids)):

                if issubset_sequence(s_text_ids, text_ids):

                    break

                else:

                    s_text_ids = s_text_ids[1:]



            if len(s_text_ids) == 0:

                s_text_ids = tokenizer.encode(s_text, add_special_tokens=False)

                for i in range(len(s_text_ids)):

                    if issubset_sequence(s_text_ids, text_ids):

                        break

                    else:

                        s_text_ids = s_text_ids[:-1]



            for i in range(len(text_ids)):

                if text_ids[i:i+len(s_text_ids)] == s_text_ids:

                    # +3 from cls_id, sentiment and sep_id and -1 for end_target

                    start_target, end_target = i+3, i+len(s_text_ids)+3-1 

                    if (start_target - end_target) == 1:

                        end_target += 1

#                         print(text, '__|__', s_text)

#                         print(text_ids)

#                         print(start_target, end_target)

#                         print(s_text_ids)

#                         raise ValueError('tes')

                    elif end_target-3 > len(text_ids):

                        print(text, '__|__', s_text)

                        print(text_ids)

                        print(start_target, end_target)

                        print(s_text_ids)

                        raise ValueError('tes, end_target > length text ids')

                    break



            input_ids = CLS_ID + [sentiment_id[sentiment]] + SEP_ID + text_ids + SEP_ID

            token_type_ids = [0] * 3 + [1] * (len(text_ids)+1)

            attention_mask = [1] * len(input_ids)



            input_ids = pad_or_truncate(input_ids, max_len, SEP_ID, is_input_ids=True)

            token_type_ids = pad_or_truncate(token_type_ids, max_len, SEP_ID)

            attention_mask = pad_or_truncate(attention_mask, max_len, SEP_ID)

            

            if len(offsets) < max_len:

                offsets = offsets + ( [(0,0)] * (max_len - len(offsets)) )

            else:

                offsets = offsets[:max_len]



            if start_target == 0 or end_target == 0:

                print(text, '_|_', s_text)

                print(tokenizer.tokenize(text), '_|_', tokenizer.tokenize(s_text))

                print(text_ids)

                print(s_text_ids)

                raise ValueError('tes')



            albert_input = { 'input_ids':torch.tensor(input_ids).to(device),

                            'token_type_ids':torch.tensor(token_type_ids).to(device),

                            'attention_mask':torch.tensor(attention_mask).to(device),

                            'start_targets':torch.tensor(start_target).to(device),

                            'end_targets':torch.tensor(end_target).to(device),

                            'offsets':torch.tensor(offsets).to(device),

                            'original_texts':text, 

                            'sentiments':sentiment, 

            }

            all_data.append(albert_input)

    else:

        for idx, row in df.iterrows():

            text = ' '+ ' '.join( row.text.lower().split() )

            sentiment = row.sentiment.strip()

            text_ids, offsets = o_tokenizer.encode(text)



            input_ids = CLS_ID + [sentiment_id[sentiment]] + SEP_ID + text_ids + SEP_ID

            token_type_ids = [0] * 3 + [1] * (len(text_ids)+1)

            attention_mask = [1] * len(input_ids)



            input_ids = pad_or_truncate(input_ids, max_len, SEP_ID, is_input_ids=True)

            token_type_ids = pad_or_truncate(token_type_ids, max_len, SEP_ID)

            attention_mask = pad_or_truncate(attention_mask, max_len, SEP_ID)

            if len(offsets) < max_len:

                offsets = offsets + ( [(0,0)] * (max_len - len(offsets)) )

            else:

                offsets = offsets[:max_len]



            albert_input = { 'input_ids':torch.tensor(input_ids).to(device),

                            'token_type_ids':torch.tensor(token_type_ids).to(device),

                            'attention_mask':torch.tensor(attention_mask).to(device),

                            'offsets':torch.tensor(offsets).to(device),

                            'original_texts':text, 

                            'sentiments':sentiment, 

                            

            }

            all_data.append(albert_input)

            

    return all_data

    
def generate_dataloader(albert_inputs, split=True, val_size=0.2, batch_size=32):

    if split:

        val_size = math.floor(len(albert_inputs)*val_size)

        train_data = albert_inputs[val_size:]

        val_data = albert_inputs[:val_size]

        train_dataloader = DataLoader(train_data, batch_size=batch_size)

        val_dataloader = DataLoader(val_data, batch_size=batch_size)



        return train_dataloader, val_dataloader

    else:

        return DataLoader(albert_inputs, batch_size=batch_size)
albert_inputs = data_extractor(train)

train_dataloader, val_dataloader = generate_dataloader(albert_inputs, split=True)
albert_inputs[0]
len ( (albert_inputs[0]['offsets'].sum(1) > 0).nonzero() ) 
# class TweetSelectionModel(transformers.AlbertPreTrainedModel):

class TweetSelectionModel(transformers.AlbertPreTrainedModel):

    def __init__(self, conf, hidden_dim=768):

        super(TweetSelectionModel, self).__init__(conf, hidden_dim)

        self.albert = transformers.AlbertModel.from_pretrained(path_albert_model, 

                                                               config=conf, from_tf=False)

        self.drop_out = nn.Dropout(0.3)

        self.linear = nn.Linear(hidden_dim*2, 2)

        torch.nn.init.normal_(self.linear.weight, std=0.02)

    

    def forward(self, ids, token_type_ids, mask):

        # config.output hiddenstates = True

        _, _, out = self.albert(

            ids,

            token_type_ids=token_type_ids,

            attention_mask=mask,

        )



        out = torch.cat((out[-1], out[-2]), dim=-1)

#         out = torch.cat((out[:,-1,:], out[:,-2,:]), dim=-1)

#         print('concat', out.shape)

        out = self.drop_out(out)

        logits = self.linear(out)



        start_logits, end_logits = logits.split(1, dim=-1)



        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)



        return start_logits, end_logits

def loss_fn(start_logits, end_logits, start_positions, end_positions):

    """

    Return the sum of the cross entropy losses for both the start and end logits

    """

    loss_fct = nn.CrossEntropyLoss()

    start_loss = loss_fct(start_logits, start_positions)

    end_loss = loss_fct(end_logits, end_positions)

    total_loss = (start_loss + end_loss)

    return total_loss



def eval(data, start_outputs, end_outputs):

    """evaluation data in batch"""

    scores = []

    input_ids = data['input_ids']

    start_targets = data['start_targets']

    end_targets = data['end_targets']

    offsets = data['offsets']

    texts = data['original_texts']

    sentiments = data['sentiments']

    c_wrong_end = 0

    start_outputs = torch.argmax(start_outputs, dim=1)

    end_outputs = torch.argmax(end_outputs, dim=1)

    # looping batch size

    for text, sentiment, offset, s_target, e_target, s_output, e_output in zip(texts, sentiments, offsets, start_targets, 

                                                                               end_targets, start_outputs, end_outputs):



        s_target, e_target, s_output, e_output = s_target.item()-3, e_target.item()-3, max(s_output.item()-3, 0), max(e_output.item()-3, 0)

        len_offset = len ( (offset.sum(1) > 0).nonzero() )

        offset = offset.tolist()

    

        # handle overlap offsets

        if s_output > e_output:

            if sentiment == 'neutral':

                e_output = len_offset

            else:

                s_output = e_output

            c_wrong_end += 1

        elif e_output > len_offset:

            e_output = len_offset



        s_char, e_char, s_char_output, e_char_output = offset[s_target][0], offset[e_target][1], offset[s_output][0], offset[e_output][1]

        s_text = text[s_char:e_char]

        pred_s_text = text[s_char_output:e_char_output]

        

        # I can't remove this print codes for debug, LOL

#         print('======')

#         print('text and pred text')

#         print(s_text)

#         print(pred_s_text)

#         print(s_char, e_char, s_char_output, e_char_output)

#         print(s_target, e_target, s_output, e_output)

#         print(text)

#         print(o_tokenizer.encode(text))

#         print(tokenizer.encode(text, add_special_tokens=False))

#         print(offset[:15])

        

        score_jaccard = compute_jaccard(s_text, pred_s_text)

        scores.append(score_jaccard)

    

    return c_wrong_end, round( sum(scores) / len(scores), 2 )

        

        
def eval_dataloader(model, dataloader):

    model.eval()

    total_c_wrong_end, jaccard_mean_score, loss_score = 0, 0, 0

    for data in dataloader:

        input_ids = data['input_ids'].to(device)

        token_type_ids = data['token_type_ids'].to(device)

        attention_mask = data['attention_mask'].to(device)

        start_targets = data['start_targets'].to(device)

        end_targets = data['end_targets'].to(device)



        with torch.no_grad():

            start_outputs, end_outputs = model(input_ids, token_type_ids, attention_mask)



            loss = loss_fn(start_outputs, end_outputs, start_targets, end_targets)



            c_wrong_end, jaccard_score = eval(data, start_outputs, end_outputs)

            total_c_wrong_end += c_wrong_end



            if loss_score == 0:

                loss_score = loss.item()

            else:

                loss_score = round( (loss_score + loss.item() ) / 2, 3 )



            if jaccard_mean_score == 0:

                jaccard_mean_score = jaccard_score

            else:

                jaccard_mean_score = round( (jaccard_score + jaccard_mean_score) / 2, 3 )

    

    return total_c_wrong_end, loss_score, jaccard_mean_score
def train_model(train_dataloader, val_dataloader, model, epochs=3):

    

    model = model.to(device)

    optimizer = transformers.AdamW(model.parameters(), weight_decay=0.001, lr=1e-5)

    

    for i in range(epochs):

        model.train()

        jaccard_mean_score, loss_score = 0, 0

        total_c_wrong_end = 0

        for data in train_dataloader:

            input_ids = data['input_ids'].to(device)

            token_type_ids = data['token_type_ids'].to(device)

            attention_mask = data['attention_mask'].to(device)

            start_targets = data['start_targets'].to(device)

            end_targets = data['end_targets'].to(device)

            

            model.zero_grad()

            start_outputs, end_outputs = model(input_ids, token_type_ids, attention_mask)



            loss = loss_fn(start_outputs, end_outputs, start_targets, end_targets)

            loss.backward()

            optimizer.step()



            c_wrong_end, jaccard_score = eval(data, start_outputs, end_outputs)

            total_c_wrong_end += c_wrong_end

            if loss_score == 0:

                loss_score = loss.item()

            else:

                loss_score = round( (loss_score + loss.item() ) / 2, 3 )



            if jaccard_mean_score == 0:

                jaccard_mean_score = jaccard_score

            else:

                jaccard_mean_score = round( (jaccard_score + jaccard_mean_score) / 2, 3 )

            

        val_total_c_wrong_end, val_loss, val_jaccard_mean_score = eval_dataloader(model, val_dataloader)

        print('epoch:', i, end='  ')

        print('train end < start:', str(total_c_wrong_end) + '/' + str(len(train_dataloader.dataset)), end='  ')

        print('val end < start:', str(val_total_c_wrong_end) + '/' + str(len(val_dataloader.dataset)), end='  ')

        print('train loss:', loss_score, end='  ')

        print('val loss:', val_loss, end='  ')

        print('train jaccard:', jaccard_score, end='  ')

        print('val jaccard:', val_jaccard_mean_score, end='  ')

        print()

        

#         torch.save({'model':model.state_dict(), 'optim':optimizer.state_dict()}, '/kaggle/working/albert_sentiment_extraction.pt')

    

    return model

    
model = TweetSelectionModel(albert_config)
%%time



model = train_model(train_dataloader, val_dataloader, model, epochs=5)
# test

model.eval()

model.to(device)

data = next(iter(val_dataloader))

start, stop = model(data['input_ids'], data['token_type_ids'], data['attention_mask'])

print( start.argmax(1), torch.argmax(stop, dim=1) )

print( data['start_targets'], data['end_targets'] )
test_albert_input = data_extractor(test, data_test=True, random=False)
def predict_submission(df, model, device='cpu'):

    test_albert_input = data_extractor(df, data_test=True, random=False)

    dataloader = generate_dataloader(test_albert_input, split=False)

    model.eval()

    model.to(device)

    predicts = []

    s_outputs, e_outputs, a_s_outputs, a_e_outputs, s_chars, e_chars = [], [], [], [], [], [] # for evaluate

    for data in dataloader:

        input_ids = data['input_ids'].to(device)

        token_type_ids = data['token_type_ids'].to(device)

        attention_mask = data['attention_mask'].to(device)

        offsets = data['offsets'].to(device)

        texts = data['original_texts']

        sentiments = data['sentiments']



        with torch.no_grad():

            start_outputs, end_outputs = model(input_ids, token_type_ids, attention_mask)

            start_outputs, end_outputs = start_outputs.argmax(1), end_outputs.argmax(1)

            

            for text, sentiment, s_output, e_output, offset in zip(texts, sentiments, start_outputs, end_outputs, offsets):

                s_output, e_output = max(s_output.item()-3, 0), max(e_output.item()-3, 0)

                len_offset = len ( (offset.sum(1) > 0).nonzero() )



                s_outputs.append(s_output)

                e_outputs.append(e_output)

                

                offset = offset.tolist()

                if s_output > e_output:

                    if sentiment == 'neutral':

                        e_output = len_offset

                    else:

                        s_output = e_output

                elif e_output > len_offset:

                    e_output = len_offset



                s_char_output, e_char_output = offset[s_output][0], offset[e_output][1]

                predict = text[s_char_output:e_char_output]

                predicts.append(predict)

                s_chars.append(s_char_output)

                e_chars.append(e_char_output)

                a_s_outputs.append(s_output)

                a_e_outputs.append(e_output)

    return s_outputs, e_outputs, a_s_outputs, a_e_outputs, s_chars, e_chars, predicts              
_, _, _, _, _, _, predicts = predict_submission(test, model, device=device)

len(predicts), len(test)
predicts[:10]
predicts = [p.strip() for p in predicts]
test['selected_text'] = predicts

test.head()
test[['textID','selected_text']].to_csv('submission.csv', index=False)
pd.set_option('display.max_colwidth', -1)
test_albert_inputs = data_extractor(train, random=False)

ori_s_targets, ori_e_targets = [], []

for d in test_albert_inputs:

    ori_s_targets.append( d['start_targets'].item() )

    ori_e_targets.append( d['end_targets'].item() )
train['ori_s_target'] = ori_s_targets

train['ori_e_target'] = ori_e_targets
train = train[:5000] # just evaluate for some data
s_outputs, e_outputs, a_s_outputs, a_e_outputs, s_chars, e_chars, t_predicts = predict_submission(train, model, device=device)

train['predict'] = t_predicts

train['s_target'] = s_outputs

train['e_target'] = e_outputs

train['a_s_target'] = s_outputs

train['a_e_target'] = e_outputs

train['s_char'] = s_chars

train['e_char'] = e_chars
train[train.sentiment == 'positive'][50:100]
# t_p = train[train.sentiment == 'negative']['predict'].values

# t_s = train[train.sentiment == 'negative']['selected_text'].values

# scores = []

# for p, s in zip(t_p, t_s):

#     score = compute_jaccard(p, s)

#     scores.append(score)

# print(np.mean(scores))