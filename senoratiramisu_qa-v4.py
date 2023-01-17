# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizer

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torch.nn as nn 

import torch.nn.functional as F

from torch.utils.data import DataLoader

import torch.optim as optim

from scipy.stats import spearmanr

from datetime import datetime

from sklearn.model_selection import GroupKFold,KFold

import gc

import seaborn as sns

import transformers

import re

from collections import Counter as ct

import html



import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))]



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/google-quest-challenge/train.csv')
words = []

for w in train.question_title + train.question_body + train.answer:

    words = words + w.split()
counter = ct(words) 
train.question_title = train.question_title.apply(lambda x: " ".join("[UNK]" if counter[w] < 2 else w for w in x.split()))

train.question_body = train.question_body.apply(lambda x: " ".join("[UNK]" if counter[w] < 2 else w for w in x.split()))

train.answer = train.answer.apply(lambda x: " ".join("[UNK]" if counter[w] < 2 else w for w in x.split()))
train.question_title = train.question_title.apply(html.unescape)

train.question_body = train.question_body.apply(html.unescape)

train.answer = train.answer.apply(html.unescape)
targets = torch.tensor(train[['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']].values,dtype=torch.float32)
class bertdataset:

    def __init__(self, qtitle, qbody, answer, targets,tokenizer, max_length=512):

        self.qtitle = qtitle

        self.qbody = qbody

        self.answer = answer

        self.targets = targets

        self.tokenizer = tokenizer

        self.max_length = max_length



    def __len__(self):

        return len(self.answer)



    def __getitem__(self, item):

        

        question_title = self.qtitle[item]

        question_body = self.qbody[item]

        answer_text = self.answer[item]

        

        question_title = " ".join(question_title.split())

        question_body = " ".join(question_body.split())

        answer_text = " ".join(answer_text.split())



        inputs_q = self.tokenizer.encode_plus("[CLS]" + question_title + "[QBODY]" + question_body + "[SEP]",           

            pad_to_max_length=True,

            max_length=self.max_length,

        )

        ids_q = inputs_q["input_ids"]

        token_type_ids_q = inputs_q["token_type_ids"]

        mask_q = inputs_q["attention_mask"]

        

        inputs_a = self.tokenizer.encode_plus(

            "[CLS]" + answer_text + "[SEP]",

            pad_to_max_length=True,

            max_length=self.max_length,

        )

        ids_a = inputs_a["input_ids"]

        token_type_ids_a = inputs_a["token_type_ids"]

        mask_a = inputs_a["attention_mask"]

        

        return {

        'ids_q': torch.tensor(ids_q, dtype=torch.long),

        'mask_q': torch.tensor(mask_q, dtype=torch.long),

        'token_type_ids_q': torch.tensor(token_type_ids_q, dtype=torch.long),

        'ids_a': torch.tensor(ids_a, dtype=torch.long),

        'mask_a': torch.tensor(mask_a, dtype=torch.long),

        'token_type_ids_a': torch.tensor(token_type_ids_a, dtype=torch.long),

        'targets': self.targets[item]

        }
#model

class nlp(nn.Module):

    def __init__(self,bert_path):

        super(nlp,self).__init__()

        self.bert_path = bert_path

        self.bert_model = BertModel.from_pretrained(self.bert_path, output_hidden_states=True)

        self.drop = nn.Dropout(0.2)

        self.dense = nn.Linear(768*2, 30)



    def forward(self,ids_q,mask_q,token_type_ids_q,ids_a,mask_a,token_type_ids_a):

        hidden_layers_q = self.bert_model(ids_q,attention_mask=mask_q,token_type_ids=token_type_ids_q)[2]

        hidden_layers_a = self.bert_model(ids_a,attention_mask=mask_a,token_type_ids=token_type_ids_a)[2]

        

        

        q12,a12 = hidden_layers_q[-1][:,0].view(-1,1,768),hidden_layers_a[-1][:,0].view(-1,1,768)

        q11,a11 = hidden_layers_q[-2][:,0].view(-1,1,768),hidden_layers_a[-2][:,0].view(-1,1,768)

        q10,a10 = hidden_layers_q[-3][:,0].view(-1,1,768),hidden_layers_a[-3][:,0].view(-1,1,768)

        q9,a9 = hidden_layers_q[-4][:,0].view(-1,1,768),hidden_layers_a[-4][:,0].view(-1,1,768)

        



        q = torch.mean(torch.cat((q12,q11,q10,q9),axis = 1),axis = 1)

        a = torch.mean(torch.cat((a12,a11,a10,a9),axis = 1),axis = 1)

        

        x = torch.cat((q,a),1)

        x = self.dense(self.drop(x))

        

        return x
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True,

#                                          add_specials_tokens = ["[QBODY]","[UNK]"])



# text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"





# inputs_q = tokenizer.encode_plus(

#             text,           

#             pad_to_max_length=True,

#             max_length=512,

#             return_tensors = 'pt'

#         )

# ids_q = inputs_q["input_ids"].type(torch.LongTensor).cuda()

# token_type_ids_q = inputs_q["token_type_ids"].type(torch.LongTensor).cuda()

# mask_q = inputs_q["attention_mask"].type(torch.LongTensor).cuda()



# ids_q = torch.stack((ids_q,ids_q,ids_q)).squeeze(1)

# token_type_ids_q = torch.stack((token_type_ids_q,token_type_ids_q,token_type_ids_q)).squeeze(1)

# mask_q = torch.stack((mask_q,mask_q,mask_q)).squeeze(1)



# inputs_a = tokenizer.encode_plus(

#     text,

#     pad_to_max_length=True,

#     max_length=512,

#     return_tensors = 'pt'

# )

# ids_a = inputs_a["input_ids"].type(torch.LongTensor).cuda()

# token_type_ids_a = inputs_a["token_type_ids"].type(torch.LongTensor).cuda()

# mask_a = inputs_a["attention_mask"].type(torch.LongTensor).cuda()



# ids_a = torch.stack((ids_a,ids_a,ids_a)).squeeze(1)

# token_type_ids_a = torch.stack((token_type_ids_a,token_type_ids_a,token_type_ids_a)).squeeze(1)

# mask_a = torch.stack((mask_a,mask_a,mask_a)).squeeze(1)



# model = nlp('bert-base-uncased').cuda()

# x,y = model(ids_q,mask_q,token_type_ids_q,ids_a,mask_a,token_type_ids_a)
def train_loop(dataset,model,optimizer,batch_size=6,epochs=1):

    

    batches = DataLoader(dataset,shuffle=False,batch_size=batch_size,num_workers=4)

    criterion = nn.BCEWithLogitsLoss(reduce='mean')

    

    model.train()

    for i in range(epochs):

            for j,batch in enumerate(batches):

                ids_q_batch = batch['ids_q'].type(torch.LongTensor).cuda()

                mask_q_batch = batch['mask_q'].type(torch.LongTensor).cuda()

                segments_q_batch = batch['token_type_ids_q'].type(torch.LongTensor).cuda()

                ids_a_batch = batch['ids_a'].type(torch.LongTensor).cuda()

                mask_a_batch = batch['mask_a'].type(torch.LongTensor).cuda()

                segments_a_batch = batch['token_type_ids_a'].type(torch.LongTensor).cuda()

                optimizer.zero_grad()

                output = model(ids_q = ids_q_batch,

                               mask_q = mask_q_batch,

                               token_type_ids_q = segments_q_batch,

                               ids_a = ids_a_batch,

                               mask_a = mask_a_batch,

                               token_type_ids_a = segments_a_batch)

                target_batch = batch['targets'].cuda()

                loss = criterion(output,target_batch)

                loss.backward()

                optimizer.step()

                

    loss = None
def eval_loop(dataset,model):

    batches = DataLoader(dataset,shuffle=False,batch_size=6,num_workers=4)



    model.eval()

    pred_fold = []

    target_fold = []

    score_fold = 0

    

    with torch.no_grad():

        for j,batch in enumerate(batches):



            ids_q_batch = batch['ids_q'].type(torch.LongTensor).cuda()

            mask_q_batch = batch['mask_q'].type(torch.LongTensor).cuda()

            segments_q_batch = batch['token_type_ids_q'].type(torch.LongTensor).cuda()

            ids_a_batch = batch['ids_a'].type(torch.LongTensor).cuda()

            mask_a_batch = batch['mask_a'].type(torch.LongTensor).cuda()

            segments_a_batch = batch['token_type_ids_a'].type(torch.LongTensor).cuda()

            output = model(ids_q = ids_q_batch,

                           mask_q = mask_q_batch,

                           token_type_ids_q = segments_q_batch,

                           ids_a = ids_a_batch,

                           mask_a = mask_a_batch,

                           token_type_ids_a = segments_a_batch)



            out = torch.sigmoid(output).cpu().numpy()

            target_fold.append(batch['targets'].numpy())

            

            pred_fold.append(out)

            

    pred_fold = np.vstack(pred_fold)

    target_fold = np.vstack(target_fold)  

    

    for i in range(30):

        score_fold += spearmanr(target_fold[:,i],pred_fold[:,i]).correlation

        

    return pred_fold,score_fold/30
# def test_loop(dataset,model):

#     batches = DataLoader(dataset,shuffle=False,batch_size=4,num_workers=4)



#     model.eval()

#     pred = []



#     with torch.no_grad():

#         for j,batch in enumerate(batches):



#             ids_batch = batch['ids'].type(torch.LongTensor).cuda()

#             mask_batch = batch['mask'].type(torch.LongTensor).cuda()

#             segments_batch = batch['segments'].type(torch.LongTensor).cuda()

#             out = torch.sigmoid(model(ids = ids_batch,

#                                       mask = mask_batch,

#                                       token_type_ids = segments_batch)).cpu().numpy()

            

#             pred.append(out)



#     return np.vstack(pred)
def cross_val(tokenizer,cv=3):

    oof_predictions = np.zeros((6079,30))

    folds = GroupKFold(n_splits=cv)

    

    

    

    for fold,(train_index,valid_index) in enumerate(folds.split(X=train.question_body, groups=train.question_body)):



        qtitle_train = train.iloc[train_index].question_title.values.astype(str).tolist()

        qbody_train = train.iloc[train_index].question_body.values.astype(str).tolist()

        answer_train = train.iloc[train_index].answer.values.astype(str).tolist()



        train_loader = bertdataset(qtitle=qtitle_train,

                                    qbody=qbody_train,

                                    answer=answer_train,

                                    targets=targets[train_index],

                                    tokenizer=tokenizer)



        qtitle_valid = train.iloc[valid_index].question_title.values.astype(str).tolist()

        qbody_valid = train.iloc[valid_index].question_body.values.astype(str).tolist()

        answer_valid = train.iloc[valid_index].answer.values.astype(str).tolist()



        valid_loader = bertdataset(qtitle=qtitle_valid,

                                    qbody=qbody_valid,

                                    answer=answer_valid,

                                    targets=targets[valid_index],

                                    tokenizer=tokenizer)





        model = nlp('bert-base-uncased').cuda()



        optimizer = optim.AdamW(model.parameters(), lr=3e-5)



        print(f'Fold {fold} started at ' + datetime.now().strftime("%H:%M"))



        train_loop(train_loader,model,optimizer,epochs=4)

        

        print('Training last layers a little bit more ' + datetime.now().strftime("%H:%M"))

        

        for p in model.bert_model.parameters():

            p.requires_grad = False

            

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5)

        train_loop(train_loader,model,optimizer,batch_size=6,epochs=3)



        print(f'Fold {fold} training finished, predicting... ' + datetime.now().strftime("%H:%M"))



        p,s = eval_loop(valid_loader,model) 

        oof_predictions[valid_index] = p

#       test_predictions[fold] = (test_loop(test_loader,model))



        print(f'Fold {fold} finished at ' + datetime.now().strftime("%H:%M") + f' with score: {s}')



        torch.save(model.state_dict(),f"fold_{fold}_r.pt")



        model = None

        optimizer = None

        torch.cuda.empty_cache()



    return oof_predictions
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True,

                                         add_specials_tokens = ["[QBODY]","[UNK]"])

oof_predictions = cross_val(tokenizer)
# n = train['url'].apply(lambda x:(('ell.stackexchange.com' in x) or ('english.stackexchange.com' in x))).tolist()

# spelling=[]



# for x in n:

#     if x:

#         spelling.append(0.5)

#     else:

#         spelling.append(0.)



# spearmanr(train['question_type_spelling'],np.array(spelling)).correlation