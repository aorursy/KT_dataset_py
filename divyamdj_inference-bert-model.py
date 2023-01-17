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

        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.bert_drop = nn.Dropout(0.3)

        self.out = nn.Linear(768 * 2, 1)



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



        inputs = self.tokenizer.encode_plus(

            comment_text,

            None,

            add_special_tokens=True,

            max_length=self.max_length,

        )

        ids = inputs["input_ids"]

        token_type_ids = inputs["token_type_ids"]

        mask = inputs["attention_mask"]

        

        padding_length = self.max_length - len(ids)

        

        ids = ids + ([0] * padding_length)

        mask = mask + ([0] * padding_length)

        token_type_ids = token_type_ids + ([0] * padding_length)

        

        return {

            'ids': torch.tensor(ids, dtype=torch.long),

            'mask': torch.tensor(mask, dtype=torch.long),

            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)

        }
df = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/test.csv")
tokenizer = transformers.BertTokenizer.from_pretrained("../input/bert-base-multilingual-uncased/", do_lower_case=True)
device = "cuda"

model = BERTBaseUncased(bert_path="../input/bert-base-multilingual-uncased/").to(device)

model.load_state_dict(torch.load("../input/jmodelval/model.bin"))

model.eval()
valid_dataset = BERTDatasetTest(

        comment_text=df.content.values,

        tokenizer=tokenizer,

        max_length=192

)



valid_data_loader = torch.utils.data.DataLoader(

    valid_dataset,

    batch_size=64,

    drop_last=False,

    num_workers=4,

    shuffle=False

)
with torch.no_grad():

    fin_outputs = []

    for bi, d in tqdm(enumerate(valid_data_loader)):

        ids = d["ids"]

        mask = d["mask"]

        token_type_ids = d["token_type_ids"]



        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)



        outputs = model(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )



        outputs_np = outputs.cpu().detach().numpy().tolist()

        fin_outputs.extend(outputs_np)
# translated english data
df_en = pd.read_csv("../input/test-en-df/test_en.csv")



valid_dataset = BERTDatasetTest(

        comment_text=df_en.content_en.values,

        tokenizer=tokenizer,

        max_length=192

)



valid_data_loader = torch.utils.data.DataLoader(

    valid_dataset,

    batch_size=64,

    drop_last=False,

    num_workers=4,

    shuffle=False

)



with torch.no_grad():

    fin_outputs_en = []

    for bi, d in tqdm(enumerate(valid_data_loader)):

        ids = d["ids"]

        mask = d["mask"]

        token_type_ids = d["token_type_ids"]



        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)



        outputs = model(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )



        outputs_np = outputs.cpu().detach().numpy().tolist()

        fin_outputs_en.extend(outputs_np)
# another translated english data
df_en2 = pd.read_csv("../input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv")



valid_dataset = BERTDatasetTest(

        comment_text=df_en2.translated.values,

        tokenizer=tokenizer,

        max_length=192

)



valid_data_loader = torch.utils.data.DataLoader(

    valid_dataset,

    batch_size=64,

    drop_last=False,

    num_workers=4,

    shuffle=False

)



with torch.no_grad():

    fin_outputs_en2 = []

    for bi, d in tqdm(enumerate(valid_data_loader)):

        ids = d["ids"]

        mask = d["mask"]

        token_type_ids = d["token_type_ids"]



        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)



        outputs = model(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )



        outputs_np = outputs.cpu().detach().numpy().tolist()

        fin_outputs_en2.extend(outputs_np)
fin_outputs_en = [item for sublist in fin_outputs_en for item in sublist]

fin_outputs_en2 = [item for sublist in fin_outputs_en2 for item in sublist]

fin_outputs = [item for sublist in fin_outputs for item in sublist]
sample = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")

sample.loc[:, "toxic"] = (np.array(fin_outputs) + np.array(fin_outputs_en) + np.array(fin_outputs_en2)) / 3.0

sample.to_csv("submission.csv", index=False)
sample.head()