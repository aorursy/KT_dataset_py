# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import transformers

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
true = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")

fake = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")

from transformers import DistilBertModel, DistilBertConfig

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from tqdm import tqdm

import time

from transformers import AdamW,get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from transformers import glue_convert_examples_to_features as convert_examples_to_features
true["Label"] = 1

fake["Label"] = 0
data = pd.concat([true,fake],ignore_index=True)


#data_final = data[["title","Label"]]
data["total_data"] = data["title"] + " [SEP] " + data["text"]
#seq_len = [len(tokenizer.encode(data.iloc[i]["total_data"])) for i in range(len(data))]

#print(np.median(seq_len))
data["Label"].value_counts()
train , test = train_test_split(data,shuffle=True,test_size=0.33, random_state=42)
def create_data_set(df,tokenizer):

    input_ids = []

    token_type_ids = []

    attention_mask = []

    labels = []

    for i in range(len(df)):

        text = df.iloc[i]["total_data"]

        label = df.iloc[i]["Label"]

        out = tokenizer.encode_plus(text,add_special_tokens=True,max_length=256,pad_to_max_length=True)

        input_ids.append(out["input_ids"])

        token_type_ids.append(out["input_ids"])

        attention_mask.append(out["attention_mask"])

        labels.append(label)

    all_input_ids = torch.tensor(input_ids,dtype=torch.long)

    all_token_type_ids = torch.tensor(token_type_ids,dtype=torch.long)

    all_attention_mask = torch.tensor(attention_mask,dtype= torch.long)

    all_labels = torch.tensor(labels,dtype=torch.long)

    tensor_dataset = TensorDataset(all_input_ids,all_token_type_ids,all_attention_mask,all_labels)

    return tensor_dataset
def do_train(train,test,epochs,model,tokenizer,device):

    train_data = create_data_set(train,tokenizer)

    train_data_loader = DataLoader(train_data,batch_size=64)

    loss_per_epoch = []

    test_accuracy_per_epoch = []

    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_data_loader))

    for epoch in range(epochs):

        tr_loss = 0

        for ids,type_ids,att_mask,labels in tqdm(train_data_loader):

            batch = tuple((ids.to(device),att_mask.to(device),type_ids.to(device),labels.to(device)))

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            outputs = model(**inputs)

            loss = outputs[0]

            logits = outputs[1]

            loss.backward()

            optimizer.step()

            scheduler.step()

            model.zero_grad()

            tr_loss = tr_loss + loss.item() 

        loss_per_epoch.append(tr_loss/(epoch+1))

        test_accuracy = do_test(test,model,tokenizer,device)

        test_accuracy_per_epoch.append(test_accuracy)

        print(epoch,test_accuracy)

    

    return loss_per_epoch,test_accuracy_per_epoch



def do_test(data,model,tokenizer,device):

    eval_data = create_data_set(data,tokenizer)

    eval_data_loader = DataLoader(eval_data,batch_size=128)

    #print("Number of Batch:",eval_data_loader.batch_size)

    model.eval()

    accuracy = 0.0

    with torch.no_grad():

        for ids,type_ids,att_mask,labels in tqdm(eval_data_loader):

            batch = tuple((ids.to(device),att_mask.to(device),type_ids.to(device),labels.to(device)))

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            outputs = model(**inputs)

            logits = outputs[1]

            preds = logits.detach().cpu().numpy()

            out_label_ids = inputs["labels"].detach().cpu().numpy()

            preds = np.argmax(preds,axis=1)

            print(out_label_ids)

            print(preds)

            print(accuracy_score(out_label_ids,preds,normalize=False))

            accuracy = accuracy + accuracy_score(out_label_ids,preds,normalize=False)

    

    #print(accuracy,len(data))

    return (accuracy/len(data))
def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #device = "cpu"

    config = DistilBertConfig()

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

    #tokenizer.add_special_tokens

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased')

    model.to(device)

    loss_per_epoch, eval_per_epoch = do_train(train,test,1,model,tokenizer,device)

    #return model

    return (model,loss_per_epoch, eval_per_epoch)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

#model,loss_per_epoch, eval_per_epoch  = main()

#print(do_test(test,model,tokenizer,"cuda"))
print(test.iloc[2]["title"])

print(test.iloc[2]["Label"])
text = test.iloc[923]["total_data"]

#text = "Trump fails in democracy [END] Trump held a meeting today but it failed to produce proper outcome"

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0).to("cuda")  # Batch size 1

outputs = model(input_ids)

outputs = outputs[0].detach().cpu().numpy()

preds = np.argmax(outputs,axis=1)
preds,outputs
ids = 923

print(test.iloc[ids]["total_data"])

print(test.iloc[ids]["Label"])