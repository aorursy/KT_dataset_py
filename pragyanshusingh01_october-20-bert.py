import tensorflow as tf

# Get the GPU device name.
device_name = tf.test.gpu_device_name()

# The device name should look like the following:
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')

import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
import pandas as pd
import transformers
df = pd.read_csv("../input/mbert-5-lang-absa/ABSA_MBERT_21 - English.csv")
def to_sentiment(rating):
  rating = int(rating)
  if rating == -1:
    return 0
  elif rating == 0:
    return 1
  else: 
    return 2

df['sentiment'] = df.sentiment.apply(to_sentiment)
class_names = ['negative', 'neutral', 'positive']
df.head()
from transformers import BertTokenizer, TFBertModel
pre_tr_mdl='distilbert-base-multilingual-cased'
tokenizer = transformers.DistilBertTokenizer.from_pretrained(pre_tr_mdl)
# model = TFBertModel.from_pretrained(pre_tr_mdl)
# l=0
# for i in df['reviews']:
#     maxl=max(l,len(i))
#     l=maxl
# # print(maxl) 370
import torch
input_ids = []
attention_masks = []


for sent,asp in zip(df['reviews'],df['aspect']):
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        asp,
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 370,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
       
    input_ids.append(encoded_dict['input_ids'])
    
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(df['sentiment'])
from torch.utils.data import TensorDataset, random_split

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 16
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
from transformers import DistilBertForSequenceClassification, AdamW, BertConfig

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-multilingual-cased", 
    num_labels = 3,
    output_attentions = False,
    output_hidden_states = False, 
)

# Tell pytorch to run this model on the GPU.

model.to(device)
import numpy as np
def flat_accuracy(preds, labels):
    p=[]
    for i in preds:
        i=i.cpu().detach().numpy()
        p.append(i.argmax())
    labels_flat = labels.flatten().cpu().numpy()
    return np.sum(p == labels_flat) / len(labels_flat)
def flat_accuracy_v2(preds, labels):
    p=[]
    for i in preds:
        i=i.cpu().detach().numpy()
        p.append(i.argmax())
    labels_flat = labels.flatten().cpu().numpy()
    
    return np.sum(p == labels_flat) / len(labels_flat),labels_flat,p
from transformers import BertTokenizer, glue_convert_examples_to_features
import tensorflow as tf
import tensorflow_datasets as tfds

acc=[]
optim = AdamW(model.parameters(), lr=5e-5)
model.eval()
test_res=[]
for batch in validation_dataloader:
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    labels = batch[2].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    test_res.append(flat_accuracy(outputs[1],labels))
print("UNTUNED ACCURACY==>",sum(test_res)/len(test_res))

Epochs=3

for epoch in range(Epochs):
    print("Epoch:",epoch+1," of ",Epochs)
    c=0
    l=len(train_dataloader)
    model.train()
    train_res=[]
    for batch in train_dataloader:
        c+=1
#         print("Epoch:",epoch+1,"Running ",c," of ",l)
        print("Progress {:2.1%}".format(c/ l), end="\r")
        optim.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        train_res.append(flat_accuracy(outputs[1],labels))
        loss.backward()
        optim.step()
    print("TRAIN ACCURACY==>",sum(train_res)/len(train_res))
    model.eval()
    test_res=[]
    for batch in validation_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        test_res.append(flat_accuracy(outputs[1],labels))
    print("TEST ACCURACY==>",sum(test_res)/len(test_res))
    acc.append(sum(test_res)/len(test_res))

# # k=outputs[1].cpu()
# print(k.detach().numpy())
# for batch in validation_dataloader:
#         input_ids = batch[0].to(device)
#         attention_mask = batch[1].to(device)
#         labels = batch[2].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         print(outputs[1])
print(acc)
dfd = pd.read_csv("../input/mbert-5-lang-absa/ABSA_MBERT_21 - Dutch.csv")
dfd['sentiment'] = dfd.sentiment.apply(to_sentiment)
dff = pd.read_csv("../input/mbert-5-lang-absa/ABSA_MBERT_21 - French.csv")
dff['sentiment'] = dff.sentiment.apply(to_sentiment)
dfr = pd.read_csv("../input/mbert-5-lang-absa/ABSA_MBERT_21 - Russian.csv")
dfr['sentiment'] = dfr.sentiment.apply(to_sentiment)
dfs = pd.read_csv("../input/mbert-5-lang-absa/ABSA_MBERT_21 - Spanish.csv")
dfs['sentiment'] = dfs.sentiment.apply(to_sentiment)
dft = pd.read_csv("../input/mbert-5-lang-absa/ABSA_MBERT_21 - Turkish.csv")
dft['sentiment'] = dft.sentiment.apply(to_sentiment)
# l=0
# for i in dfd['reviews']:
#     maxl=max(l,len(i))
#     l=maxl
#     print(len(i))
DU_input_ids = []
DU_attention_masks = []


for sent,asp in zip(dfd['reviews'],dfd['aspect']):
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        asp,
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 370,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
       
    DU_input_ids.append(encoded_dict['input_ids'])
    
    DU_attention_masks.append(encoded_dict['attention_mask'])

DU_input_ids = torch.cat(DU_input_ids, dim=0)
DU_attention_masks = torch.cat(DU_attention_masks, dim=0)
DU_labels = torch.tensor(dfd['sentiment'])
DU_dataset = TensorDataset(DU_input_ids, DU_attention_masks, DU_labels)
# print(len(DU_attention_masks[0]))

# l=0
# for i in dff['reviews']:
#     maxl=max(l,len(i))
#     l=maxl
#     print(len(i))
FR_input_ids = []
FR_attention_masks = []


for sent,asp in zip(dff['reviews'],dff['aspect']):
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        asp,
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 370,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
       
    FR_input_ids.append(encoded_dict['input_ids'])
    
    FR_attention_masks.append(encoded_dict['attention_mask'])

FR_input_ids = torch.cat(FR_input_ids, dim=0)
FR_attention_masks = torch.cat(FR_attention_masks, dim=0)
FR_labels = torch.tensor(dff['sentiment'])
FR_dataset = TensorDataset(FR_input_ids, FR_attention_masks, FR_labels)
# print(len(FR_attention_masks[0]),l)
# l=0
# for i in dfs['reviews']:
#     maxl=max(l,len(i))
#     l=maxl
#     print(len(i))
SP_input_ids = []
SP_attention_masks = []


for sent,asp in zip(dfs['reviews'],dfs['aspect']):
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        asp,
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 370,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
       
    SP_input_ids.append(encoded_dict['input_ids'])
    
    SP_attention_masks.append(encoded_dict['attention_mask'])

SP_input_ids = torch.cat(SP_input_ids, dim=0)
SP_attention_masks = torch.cat(SP_attention_masks, dim=0)
SP_labels = torch.tensor(dfs['sentiment'])
SP_dataset = TensorDataset(SP_input_ids, SP_attention_masks, SP_labels)
# l=0
# for i in dfr['reviews']:
#     maxl=max(l,len(i))
#     l=maxl
#     print(len(i))
RU_input_ids = []
RU_attention_masks = []


for sent,asp in zip(dfr['reviews'],dfr['aspect']):
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        asp,
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 370,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
       
    RU_input_ids.append(encoded_dict['input_ids'])
    
    RU_attention_masks.append(encoded_dict['attention_mask'])

RU_input_ids = torch.cat(RU_input_ids, dim=0)
RU_attention_masks = torch.cat(RU_attention_masks, dim=0)
RU_labels = torch.tensor(dfr['sentiment'])
RU_dataset = TensorDataset(RU_input_ids, RU_attention_masks, RU_labels)
# l=0
# for i in dfr['reviews']:
#     maxl=max(l,len(i))
#     l=maxl
#     print(len(i))
TU_input_ids = []
TU_attention_masks = []


for sent,asp in zip(dft['reviews'],dft['aspect']):
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        asp,
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 370,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
       
    TU_input_ids.append(encoded_dict['input_ids'])
    
    TU_attention_masks.append(encoded_dict['attention_mask'])

TU_input_ids = torch.cat(TU_input_ids, dim=0)
TU_attention_masks = torch.cat(TU_attention_masks, dim=0)
TU_labels = torch.tensor(dft['sentiment'])
TU_dataset = TensorDataset(TU_input_ids, TU_attention_masks, TU_labels)
batch_size = 16
DU_dataloader = DataLoader(
            DU_dataset,  
            sampler = RandomSampler(DU_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

model.eval()
test_res=[]
for batch in DU_dataloader:
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    labels = batch[2].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    test_res.append(flat_accuracy_v2(outputs[1],labels)[0])
print("DU ACCURACY==>",sum(test_res)/len(test_res))


FR_dataloader = DataLoader(
            FR_dataset,  
            sampler = RandomSampler(FR_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
model.eval()
for batch in FR_dataloader:
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    labels = batch[2].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    test_res.append(flat_accuracy_v2(outputs[1],labels)[0])
print("FR ACCURACY==>",sum(test_res)/len(test_res))


SP_dataloader = DataLoader(
            SP_dataset,  
            sampler = RandomSampler(SP_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
model.eval()
for batch in SP_dataloader:
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    labels = batch[2].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    test_res.append(flat_accuracy_v2(outputs[1],labels)[0])
print("SP ACCURACY==>",sum(test_res)/len(test_res))


RU_dataloader = DataLoader(
            RU_dataset,  
            sampler = RandomSampler(RU_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
model.eval()
for batch in RU_dataloader:
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    labels = batch[2].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    test_res.append(flat_accuracy_v2(outputs[1],labels)[0])
print("RU ACCURACY==>",sum(test_res)/len(test_res))

TU_dataloader = DataLoader(
            TU_dataset,  
            sampler = RandomSampler(TU_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
ot,lt=[],[]

model.eval()
for batch in TU_dataloader:
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    labels = batch[2].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    test_res.append(flat_accuracy_v2(outputs[1],labels)[0])
print("TU ACCURACY==>",sum(test_res)/len(test_res))

import matplotlib.pyplot as plt
p,n,ng=0,0,0
for i in dfd.sentiment:
    if i==0:
        ng+=1
    elif i==1:
        n+=1
    else:
        p+=1
d=[ng,n,p]

names = ['negative', 'neutral', 'positive']
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar(names, d)
plt.title('Dutch Sentiment Distribution')
plt.show()


p,n,ng=0,0,0
for i in dff.sentiment:
    if i==0:
        ng+=1
    elif i==1:
        n+=1
    else:
        p+=1
d=[ng,n,p]

names = ['negative', 'neutral', 'positive']
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar(names, d)
plt.title('French Sentiment Distribution')
plt.show()

p,n,ng=0,0,0
for i in dfs.sentiment:
    if i==0:
        ng+=1
    elif i==1:
        n+=1
    else:
        p+=1
d=[ng,n,p]

names = ['negative', 'neutral', 'positive']
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar(names, d)
plt.title('Spanish Sentiment Distribution')
plt.show()

p,n,ng=0,0,0
for i in dfr.sentiment:
    if i==0:
        ng+=1
    elif i==1:
        n+=1
    else:
        p+=1
d=[ng,n,p]

names = ['negative', 'neutral', 'positive']
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar(names, d)
plt.title('Russian Sentiment Distribution')
plt.show()

p,n,ng=0,0,0
for i in dft.sentiment:
    if i==0:
        ng+=1
    elif i==1:
        n+=1
    else:
        p+=1
d=[ng,n,p]

names = ['negative', 'neutral', 'positive']
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar(names, d)
plt.title('Turkish Sentiment Distribution')
plt.show()

p,n,ng=0,0,0
for i in df.sentiment:
    if i==0:
        ng+=1
    elif i==1:
        n+=1
    else:
        p+=1
d=[ng,n,p]

names = ['negative', 'neutral', 'positive']
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar(names, d)
plt.title('English Sentiment Distribution')
plt.show()
