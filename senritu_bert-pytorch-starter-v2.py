# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import os

import torch

import pandas as pd

from scipy import stats

import numpy as np



from tqdm import tqdm

from collections import OrderedDict, namedtuple

import torch.nn as nn

from torch.optim import lr_scheduler

import joblib



import logging

import transformers

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

import sys

from sklearn import metrics, model_selection

import torch.nn.functional as F

import re

import warnings

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")



DATA_PATH='../input/tweet-sentiment-extraction/'

BERT_PATH='../input/bert-base-uncased/'



# Any results you write to the current directory are saved as output.
train=pd.read_csv(DATA_PATH+'train.csv')

test=pd.read_csv(DATA_PATH+'test.csv')

submission=pd.read_csv(DATA_PATH+'sample_submission.csv')



tokenizer=transformers.BertTokenizer.from_pretrained(BERT_PATH+'vocab.txt')
def create_targets(df):

    df['t_text'] = df['text'].apply(lambda x: tokenizer.tokenize(str(x)))

    df['t_selected_text'] = df['selected_text'].apply(lambda x: tokenizer.tokenize(str(x)))

    

    def func(row):

        x,y = row['t_text'],row['t_selected_text'][:]

        for offset in range(len(x)):

            d = dict(zip(x[offset:],y))

            #when k = v that means we found the offset

            check = [k==v for k,v in d.items()]

            if all(check)== True:

                break 

        return [0]*offset + [1]*len(y) + [0]* (len(x)-offset-len(y))

    

    df['targets'] = df.apply(func,axis=1)

    return df



train = create_targets(train)



print('MAX_SEQ_LENGTH_TEXT', max(train['t_text'].apply(len)))

print('MAX_TARGET_LENGTH',max(train['targets'].apply(len)))

MAX_TARGET_LEN = MAX_SEQUENCE_LENGTH = 108
## same way tokenize the test data also (for later use)

test['t_text'] = test['text'].apply(lambda x: tokenizer.tokenize(str(x)))



## pad all the targets

train['targets'] = train['targets'].apply(lambda x :x + [0] * (MAX_TARGET_LEN-len(x)))
class BERTBaseUncased(nn.Module):

    def __init__(self, bert_path):

        super(BERTBaseUncased, self).__init__()

        self.bert_path = bert_path

        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.bert_drop = nn.Dropout(0.3)

        self.out = nn.Linear(768 * 2, MAX_SEQUENCE_LENGTH)



    def forward(

            self,

            ids,

            mask,

            token_type_ids

    ):

        o1, o2 = self.bert(

            ids,

            attention_mask=mask,

            token_type_ids=token_type_ids)

        

        apool = torch.mean(o1, 1)

        mpool, _ = torch.max(o1, 1)

        cat = torch.cat((apool, mpool), 1)



        bo = self.bert_drop(cat)

        p2 = self.out(bo)

        p2 = F.sigmoid(p2)

        

        return p2
class BERTDatasetTraining:

    def __init__(self, comment_text, tokenizer, max_length, targets=None,train=False):

        self.comment_text = comment_text

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.targets = targets

        self.train=train



    def __len__(self):

        return len(self.comment_text[0])



    def __getitem__(self, idx):

        input_ids       = self.comment_text[0][idx]

        input_masks     = self.comment_text[1][idx]

        input_segments  = self.comment_text[2][idx]

        

        if self.train: # targets

            labels = self.targets[idx]

            return input_ids, input_masks, input_segments, labels

        

        return input_ids, input_masks, input_segments
def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):

    model.train()

    tk0=tqdm(data_loader, total=len(data_loader))

    

    for bi, d in enumerate(tk0):

        

        ids,mask, token_type_ids, targets= d



        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        targets = targets.to(device, dtype=torch.float)



        optimizer.zero_grad()

        outputs = model(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )



        loss = loss_fn(outputs, targets)

        if bi % 100 == 0:

            print(f'Training -> training_data_{bi}, loss={loss}')



        loss.backward()

        optimizer.step()

        

        if scheduler is not None:

            scheduler.step()
def eval_loop_fn(data_loader, model, device,batch_size,valshape):

    model.eval()

    

    valid_preds = np.zeros((valshape, MAX_SEQUENCE_LENGTH))

    original = np.zeros((valshape, MAX_SEQUENCE_LENGTH))

    

    for bi, d in enumerate(data_loader):

        ids,mask, token_type_ids,targets= d



        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        targets = targets.to(device, dtype=torch.float)



        outputs = model(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )



        valid_preds[bi*batch_size : (bi+1)*batch_size] = outputs.detach().cpu().numpy()

        original[bi*batch_size : (bi+1)*batch_size]    = targets.detach().cpu().numpy()   



    return valid_preds, original





def loss_fn(outputs, targets):

    return nn.BCEWithLogitsLoss()(outputs, targets)
def _convert_to_bert_inputs(text, tokenizer, max_sequence_length):

    """Converts tokenized input to ids, masks and segments for BERT"""

    inputs = tokenizer.encode_plus(

        text,

        None,

        add_special_tokens=True,

        max_length=max_sequence_length,

    )

    ids = inputs["input_ids"]

    token_type_ids = inputs["token_type_ids"]

    mask = inputs["attention_mask"]



    padding_length = max_sequence_length - len(ids)



    ids = ids + ([0] * padding_length)

    mask = mask + ([0] * padding_length)

    token_type_ids = token_type_ids + ([0] * padding_length)



    return [ids,mask,token_type_ids]





def compute_input_arays(df, columns, tokenizer, max_sequence_length):

    

    input_ids, input_masks, input_segments = [], [], []

    

    for _, instance in tqdm(df.iterrows(),total=len(df)):

        

        t = str(instance.text)

        

        ids, masks, segments = _convert_to_bert_inputs(t,tokenizer, max_sequence_length)

        input_ids.append(ids)

        input_masks.append(masks)

        input_segments.append(segments)

        

    return [

        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(), 

        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),

        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),

    ]



def compute_output_arrays(df, col):

    return np.asarray(df[col].values.tolist())
def run():

    

    TRAIN_BATCH_SIZE = 64

    EPOCHS = 5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    

    tokenizer = transformers.BertTokenizer.from_pretrained(BERT_PATH)

    

    

    train_df , valid_df = train_test_split(train,test_size=0.20, random_state=42,shuffle=True) ## Split Labels



    inputs_train = compute_input_arays(train_df, 'text', tokenizer, max_sequence_length=MAX_SEQUENCE_LENGTH)

    outputs_train = compute_output_arrays(train_df,'targets')



    train_dataset = BERTDatasetTraining(

        comment_text=inputs_train,

        targets=outputs_train,

        tokenizer=tokenizer,

        max_length=MAX_SEQUENCE_LENGTH,

        train=True

    )



    train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)



    inputs_valid = compute_input_arays(valid_df, 'text', tokenizer, max_sequence_length=MAX_SEQUENCE_LENGTH)

    outputs_valid = compute_output_arrays(valid_df,'targets')

    

    valid_dataset = BERTDatasetTraining(

        comment_text=inputs_valid,

        targets=outputs_valid,

        tokenizer=tokenizer,

        max_length=MAX_SEQUENCE_LENGTH,

        train=True

    )



    

    valid_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=64,

        num_workers=4

    )



    model = BERTBaseUncased(bert_path=BERT_PATH).to(device)



    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]



    lr = 3e-5

    num_train_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE * EPOCHS)



    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    scheduler = get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=0,

        num_training_steps=num_train_steps

    )



    for epoch in range(EPOCHS):

        print(f'Epoch==>{epoch}')

        train_loop_fn(train_loader, model, optimizer, device, scheduler=scheduler)

        o, t = eval_loop_fn(valid_loader, model, device, batch_size=64,valshape=valid_df.shape[0])

        

        torch.save(model.state_dict(), "model.bin")
run()
device = "cuda"

model = BERTBaseUncased(bert_path=BERT_PATH).to(device)

model.load_state_dict(torch.load("../working/model.bin"))

model.eval()
inputs_test = compute_input_arays(test, 'text', tokenizer, max_sequence_length=MAX_SEQUENCE_LENGTH)



test_dataset = BERTDatasetTraining(

    comment_text=inputs_test,

    tokenizer=tokenizer,

    max_length=MAX_SEQUENCE_LENGTH

)





test_loader = torch.utils.data.DataLoader(

    test_dataset,

    batch_size=64,

    num_workers=4

)
with torch.no_grad():

    test_preds = np.zeros((len(test_dataset), MAX_SEQUENCE_LENGTH))

    

    for bi, d in enumerate(test_loader):

     

        ids,mask, token_type_ids= d



        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)



        outputs = model(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )



        test_preds[bi*64 : (bi+1)*64] = outputs.detach().cpu().numpy()
pred = np.where(test_preds>=0.3,1,0)



temp_output = []

for idx,p in enumerate(pred):

    indexes = np.where(p>=0.3)

    current_text = test['t_text'][idx]

    if len(indexes[0])>0:

        start = indexes[0][0]

        end = indexes[0][-1]

    else:

        start = 0

        end = len(current_text)

    

    temp_output.append(' '.join(current_text[start:end+1]))
test['temp_output'] = temp_output
def correct_op(row):

    placeholder = row['temp_output']

    for original_token in row['text'].split():

        token_str = ' '.join(tokenizer.tokenize(original_token))

        placeholder = placeholder.replace(token_str,original_token,1)

    return placeholder



test['temp_output2'] = test.apply(correct_op,axis=1)
## for Neutral tweets keep things same

def replacer(row):

    if row['sentiment']=='neutral':

        return row['text']

    else:

        return row['temp_output2']



test['temp_output2'] = test.apply(replacer,axis=1)
submission['selected_text']=test['temp_output2'].values

submission.to_csv('submission.csv',index=None)

submission.head()