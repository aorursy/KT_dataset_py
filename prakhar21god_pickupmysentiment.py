# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/output'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import torch

import pandas as pd

import torch.nn as nn

import numpy as np

import torch.nn.functional as F

from torch.optim import lr_scheduler



from sklearn import model_selection

from sklearn import metrics

import transformers

import tokenizers

from transformers import AdamW,T5Tokenizer,T5ForConditionalGeneration

from transformers import get_linear_schedule_with_warmup

from tqdm.autonotebook import tqdm
tok=T5Tokenizer.from_pretrained('t5-base')
model=T5ForConditionalGeneration.from_pretrained('t5-base')
### Two things to remember

## T5 takes question answer in the format question:<space>positive<space>context:<space>selected_text

## T5 uses sentence piece tokenizer
### first transform the input in the format
train=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
processed_input=[]

labels=[]

for text ,sentiment,selected_text in zip(train['text'],train['sentiment'],train['selected_text']):

    processed_input.append(f"question: {sentiment} context: {text} </s>")

    labels.append(f"{selected_text} </s>")

    
class TweetDataset:

    def __init__(self,original_text,text,labels):

        self.processed_text = text

        self.text=original_text

        self.labels = labels

        self.tokenizer = tok

        self.max_len =128

    

    def __len__(self):

        return len(self.text)



    def __getitem__(self, item):

        review = str(self.processed_text[item])

        review = " ".join(review.split())

        labels= ' '.join(str(self.labels[item]).split())

    

        

        inputs = self.tokenizer.encode_plus(

            review,

            None,

            add_special_tokens=True,

            max_length=self.max_len,

            pad_to_max_length=True

        )

        ids = inputs["input_ids"]

        mask = inputs["attention_mask"]

        outputs_encoded=self.tokenizer.encode(self.labels[item])

        l=len(outputs_encoded)

        output=outputs_encoded+[0]*(128-l)



        return {

            'ids': torch.tensor(ids, dtype=torch.long),

            'mask': torch.tensor(mask, dtype=torch.long),

            'original_text':self.text[item],

            

            'targets': torch.tensor(output, dtype=torch.long)

        }

train['processed']=processed_input

train['target']=labels
from sklearn.model_selection import train_test_split
train_df,val_df=train_test_split(train,random_state=24)
# train_df.head()
class AverageMeter:

    """Computes and stores the average and current value"""

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
train_dataset = TweetDataset(

        original_text=train_df.selected_text.values,

        text=train_df.processed.values,

        labels=train_df.target.values

    )



train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=16,

        num_workers=4

    )
train_dataset = TweetDataset(

        original_text=train_df.selected_text.values,

        text=train_df.processed.values,

        labels=train_df.target.values

    )



train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=16,

        num_workers=4

    )



device = torch.device("cuda")

model.to(device)



param_optimizer = list(model.named_parameters())

no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

optimizer_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

    ]



num_train_steps = int(len(train_df) / 16 * 2 )

optimizer = AdamW(optimizer_parameters, lr=3e-5)

scheduler = get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=0,

        num_training_steps=num_train_steps

    )







def train_fn(data_loader, model, optimizer, device, scheduler):

    model.train()

    losses=AverageMeter()

    tk0=tqdm(data_loader,total=len(data_loader))



    for bi, d in enumerate(tk0):

        

        ids = d["ids"]

        mask = d["mask"]

        targets = d["targets"]

    



        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets = targets.to(device)



        optimizer.zero_grad()

        outputs = model(

            input_ids=ids,

            attention_mask=mask,

            lm_labels=targets

        )



        loss = outputs[0]

        loss.backward()

        optimizer.step()

        scheduler.step()

        losses.update(loss.item(),ids.size(0))

        tk0.set_postfix(loss=losses.avg)

        



for epoch in range(2):

    train_fn(train_data_loader, model, optimizer, device, scheduler)
def jaccard(str1, str2):

    

    



    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
import re

cleanr = re.compile('<.*?>')
val_df.head()
### seems fine lets predict on test and see score

mean_jaccard=[]

for text ,processed in zip(val_df['selected_text'],val_df['processed']):

    encoded=tok.encode(processed, return_tensors="pt").to('cuda')

    output=model.generate(encoded)

    out=output.tolist()

    text_pred=' '.join(tok.decode(i) for i in out)

    text_pred=re.sub(cleanr, '', text_pred)

    print(f"jaccard:{jaccard(text_pred,text)}")

    mean_jaccard.append(jaccard(text_pred,text))
processed_output=[]

for text ,sentiment in zip(test['text'],test['sentiment']):

    processed_output.append(f"question: {sentiment} context: {text} </s>")
predictions=[]

for text in processed_output:

    encoded=tok.encode(text, return_tensors="pt").to('cuda')

    output=model.generate(encoded)

    out=output.tolist()

    text=' '.join(tok.decode(i) for i in out)

    text=re.sub(cleanr, '', text)

    predictions.append(text)

    
sample = pd.read_csv("/kaggle/input/model-pred/submission.csv")

# sample.loc[:, 'selected_text'] = predictions

# sample.to_csv("submission.csv", index=False)
test['selected_text']=predictions
test[['textID','selected_text']].to_csv('output.csv',index=False)