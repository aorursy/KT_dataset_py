import os

import torch

import pandas as pd

from scipy import stats

import numpy as np

import pandas as pd



from tqdm import tqdm

from collections import OrderedDict, namedtuple

import torch.nn as nn

from torch.optim import lr_scheduler

import joblib



import logging

import transformers

import sys
class BERTBaseUncased(nn.Module):

    def __init__(self, bert_path):

        super(BERTBaseUncased, self).__init__()

        self.bert_path = bert_path

        self.bert = transformers.BertModel.from_pretrained(self.bert_path)#Taking the pretrained model from TRANSFORMERS and defining the path(which is in config file)

        self.bert_drop = nn.Dropout(0.3)#Has a dropout 0f 0.3, that is 30% of the input tensors are zeroed out

        self.out = nn.Linear(768 * 2, 1)#We get a vector of size 768*2, one 768 for mean pooling, one 768 for max pooling



    def forward(self,ids,mask,token_type_ids):

        o1, o2 = self.bert(ids,attention_mask=mask,token_type_ids=token_type_ids)#Here the underscore represnts that we dont need the second ouotput in the forward propogation step

        apool = torch.mean(o1, 1)#Both of these will be vectors of size 768 as the out layer is a vector of size 768(self.out)

        mpool, _ = torch.max(o1, 1)

        cat = torch.cat((apool, mpool), 1)#We concat both average pooling and max pooling with axis 1



        bo = self.bert_drop(cat)

        p2 = self.out(bo)

        return p2

class BERTDatasetTest:

    def __init__(self, comment_text, tokenizer, max_length):

        self.comment_text = comment_text

        self.tokenizer = tokenizer

        self.max_length = max_length



    def __len__(self):

        return len(self.comment_text)



    def __getitem__(self, item):

        comment_text = str(self.comment_text[item])

        comment_text = " ".join(comment_text.split())



        inputs = self.tokenizer.encode_plus(#from Hugging Face Tokenizers that encodes first and second string, but here there is no second string, so its None

            comment_text,

            None,

            add_special_tokens=True,

            max_length=self.max_length,

        )

        ids = inputs["input_ids"]

        token_type_ids = inputs["token_type_ids"]

        mask = inputs["attention_mask"]

        

        padding_length = self.max_length - len(ids)

        

        ids = ids + ([0] * padding_length)#We pad it on the right for BERT as its a model with absolute position embeddings

        mask = mask + ([0] * padding_length)

        token_type_ids = token_type_ids + ([0] * padding_length)

        

        return {

            'ids': torch.tensor(ids, dtype=torch.long),

            'mask': torch.tensor(mask, dtype=torch.long),

            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)

        }
df = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/test.csv")
tokenizer = transformers.BertTokenizer.from_pretrained("../input/bert-base-multilingual-uncased/", do_lower_case=True)#Takes the tokenizer of the ber base multlingual model
device = "cuda"

model = BERTBaseUncased(bert_path="../input/bert-base-multilingual-uncased/").to(device)

model.load_state_dict(torch.load("../input/modelbin1/model.bin"))#Loads the model saved during TPU Training and uses that for the test dataset

model.eval()
valid_dataset = BERTDatasetTest(#Calls constructor of BERTDatasetTest

        comment_text=df.content.values,

        tokenizer=tokenizer,#Tokenizer is the one got from BERT base multilingual model

        max_length=192

)



valid_data_loader = torch.utils.data.DataLoader(

    valid_dataset,

    batch_size=64,#how many samples per batch to load

    drop_last=False,#the drop_last argument drops the last non-full batch of each worker’s dataset replica.As each core creates a dataset replica for itself, if there are no equal batch sizes, it will crash

    num_workers=4,#how many subprocesses to use for data loading.

    shuffle=False

)
with torch.no_grad():

    fin_outputs = []

    for bi, d in tqdm(enumerate(valid_data_loader)):#Go through all the batches inside Data Loader

        ids = d["ids"]

        mask = d["mask"]

        token_type_ids = d["token_type_ids"]

        

        #Put all the above values to the device you are using

        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)



        outputs = model(ids=ids,mask=mask,token_type_ids=token_type_ids)



        outputs_np = outputs.cpu().detach().numpy().tolist()

        fin_outputs.extend(outputs_np)
sample = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")

sample.loc[:, "toxic"] = fin_outputs

sample.to_csv("submission.csv", index=False)
sample