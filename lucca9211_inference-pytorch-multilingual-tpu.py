import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.utils import shuffle

from sklearn import metrics

# imports pytorch

import torch

import torch.nn as nn

from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import DataLoader

# # imports the torch_xla package

# import torch_xla

# import torch_xla.distributed.parallel_loader as pl

# import torch_xla.core.xla_model as xm

# import torch_xla.distributed.xla_multiprocessing as xmp



from tqdm import tqdm



#transformers

import transformers

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings



warnings.filterwarnings("ignore")
BERT_ROOT = "../input/bertbasemultilingualuncased/"

MAX_LEN = 192
DATA_ROOT = Path("..")/"input"/ "jigsaw-multilingual-toxic-comment-classification/"



test,sample = [pd.read_csv(DATA_ROOT / fname) for fname in ["test.csv","sample_submission.csv"]]



test.head()
class DatasetClass:

    def __init__(self, text,tokenizer, max_length):

        self.text = text

        self.tokenizer = tokenizer

        self.max_length = max_length

        



    def __len__(self):

        return len(self.text)



    def __getitem__(self, item):

        text = str(self.text[item])

        text = " ".join(text.split())



        inputs = self.tokenizer.encode_plus(

            text,

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





class Bert_fn(nn.Module):

    def __init__(self, bert_path):

        super(Bert_fn, self).__init__()

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



BERT_MODEL = Bert_fn(bert_path="../input/bertbasemultilingualuncased/")
tokenizer = transformers.BertTokenizer.from_pretrained(BERT_ROOT, do_lower_case=True)
device = "cuda"

model = BERT_MODEL.to(device)

model.load_state_dict(torch.load("../input/pytorch-multilingual-tpu/model.bin"))

model.eval()
test_dataset = DatasetClass(

        text=test.content.values,

        tokenizer=tokenizer,

        max_length=MAX_LEN

    )



test_loader = torch.utils.data.DataLoader(

        test_dataset,

        batch_size=64,

        shuffle = False,

        drop_last=False,

        num_workers=4

    )
with torch.no_grad():

    final_outputs = []

    for bi, d in tqdm(enumerate(test_loader)):

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

        final_outputs.extend(outputs_np)

test1 = pd.read_csv("../input/translated-test-data/test_en.csv")

test1.head()
test1_dataset = DatasetClass(

        text=test1.content_en.values,

        tokenizer=tokenizer,

        max_length=MAX_LEN

    )



test1_loader = torch.utils.data.DataLoader(

        test1_dataset,

        batch_size=64,

        shuffle = False,

        drop_last=False,

        num_workers=4

    )
with torch.no_grad():

    final_outputs1 = []

    for bi, d in tqdm(enumerate(test1_loader)):

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

        final_outputs1.extend(outputs_np)

sample.head()
sample.loc[:, "toxic"] = (np.array(final_outputs)+ np.array(final_outputs)) / 2.0

sample.to_csv("submission.csv", index=False)

sample.head()