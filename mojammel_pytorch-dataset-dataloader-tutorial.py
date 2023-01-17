import torch

import pandas as pd

import numpy as np

from torch.utils.data import Dataset, DataLoader

from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")

df = pd.read_csv('../input/sentiment-analysis-subset/sentiment.csv')

print("Number of example: {}".format(len(df)))
df.head(5)
input_pair = [df.loc[0,['premise', 'hypothesis']].values.tolist()]

tokenizer(input_pair, add_special_tokens=True)
class MyDataset(Dataset):



    def __init__(self, data, labels):

        self.inputs = np.array(data['input_ids'])

        self.mask = np.array(data['attention_mask'])

        self.labels = np.array(labels)



    def __len__(self):

        return len(self.inputs)



    def __getitem__(self, idx):

        print(idx)

        inputs = self.inputs[idx]

        mask = self.mask[idx]

        label = self.labels[idx]

        return inputs, mask, label
%%time

# prepare data for pass in tokenizer

features = df.loc[:,['premise', 'hypothesis']].values.tolist()



# tokenize and padding data

inputs = tokenizer(features, add_special_tokens=True, padding=True)



# create dataset and dataloader object

dataset = MyDataset(inputs, df['label'].values.tolist())

dataloader = DataLoader(dataset, batch_size=30, drop_last=False, shuffle=False)
%%time

first_batch = None

for i, batch in enumerate(dataloader):

    print("Batch Shape: {}\n".format(batch[0].shape))

    if i==0:

        first_batch=batch

print("Input_ids: {}\nAttention_mask: {}\nLabels: {}".format(first_batch[0], first_batch[1], first_batch[2]))
class MyDataset2(Dataset):

    

    def __init__(self, data, tokenizer):

        self.data = data.loc[:,['premise', 'hypothesis']].values

        self.labels = data['label'].values

        self.tokenizer = tokenizer



    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        print(idx)

        input_pairs = self.data[idx].tolist()

        inputs = self.tokenizer(input_pairs, add_special_tokens=True, padding=True, return_tensors='pt')

        inputs['labels'] = self.labels[idx]

        return inputs
%%time

# create dataset object

dataset2 = MyDataset2(df, tokenizer)



# create dataloader object with batch sampler

dataloader2 = DataLoader(dataset2, sampler=BatchSampler(

                                    SequentialSampler(dataset2), 

                                    batch_size=30, drop_last=False))
%%time

first_batch = None

for i, batch in enumerate(dataloader2):

    print("Batch Shape: {}\n".format(batch['input_ids'].shape))

    if i==0:

        first_batch=batch

print("Input_ids: {}\nAttention_mask: {}\nLabels: {}".format(first_batch['input_ids'], 

                                                             first_batch['attention_mask'],

                                                             first_batch['labels']))
%%time

# create dataset object

dataset2 = MyDataset2(df, tokenizer)



# create dataloader object with batch sampler

dataloader3 = DataLoader(dataset2, sampler=BatchSampler(

                                    RandomSampler(dataset2), 

                                    batch_size=30, drop_last=False))
%%time

first_batch = None

for i, batch in enumerate(dataloader3):

    print("Batch Shape: {}".format(batch['input_ids'].shape))

    if i==0:

        first_batch=batch

print("Input_ids: {}\nAttention_mask: {}\nLabels: {}".format(first_batch['input_ids'], 

                                                             first_batch['attention_mask'],

                                                             first_batch['labels']))