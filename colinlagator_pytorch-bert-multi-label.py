import numpy as np

import pandas as pd

import json

import re

import random

from nltk.corpus import stopwords



from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, BertForSequenceClassification

from torch import nn

import torch
torch.cuda.is_available()
def get_metadata():

    with open('../input/arxiv/arxiv-metadata-oai-snapshot.json', 'r') as f:

        for line in f:

            yield line
metadata = get_metadata()
text_tags_dict = {"text":[], "tags":[]}

for paper in metadata:

    parsed = json.loads(paper)

    text = parsed['title'] + ' ' + parsed['abstract']

    text_tags_dict["text"].append(text)

    text_tags_dict["tags"].append(parsed['categories'])
text_tags_df = pd.DataFrame.from_records(text_tags_dict)
text_tags_df = text_tags_df.sample(n=500000, random_state=33)
len(text_tags_df)
categories = text_tags_df['tags'].apply(lambda x: x.split(' ')).explode().unique()
label_to_int_dict = {}

for i, key in enumerate(categories):

    label_to_int_dict[key] = i
int_to_label_dict = {}

for key, val in label_to_int_dict.items():

    int_to_label_dict[val] = key
def generate_label_array(label):

    result = np.zeros(len(label_to_int_dict))

    labels = label.split(' ')

    for l in labels:

        result[label_to_int_dict[l]] = 1

    return np.expand_dims(result, 0)
tag_labels = [generate_label_array(tag) for tag in text_tags_df["tags"]]
tag_labels = np.concatenate(tag_labels, axis = 0)
tag_labels[1].shape
stop = stopwords.words('english')



text = text_tags_df['text'].apply(lambda x : x.lower())

text = text.apply(lambda x: [item for item in x if item not in stop])

text = text.apply(lambda x: re.sub('[^A-Za-z\s]+', ' ', x))

text = text.apply(lambda x: re.sub('\n', ' ', x))

text = text.apply(lambda x: re.sub(r'\s+', ' ', x))

text = text.apply(lambda x: re.sub(r'^\s', '', x))

text = text.apply(lambda x: re.sub(r'\s$', '', x))
titles = list(titles)
tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
title_tokens = tokenizer.batch_encode_plus(titles, pad_to_max_length=True, max_length=20, return_tensors='pt')
title_tokens['input_ids'].shape
random.seed(33)

sample_indices = random.sample(range(title_tokens['input_ids'].shape[0]), title_tokens["input_ids"].shape[0])
x_train, x_test, y_train, y_test = train_test_split(title_tokens["input_ids"][sample_indices,:], tag_labels[sample_indices, :], test_size = 0.2)
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
class arxiv_dataset(torch.utils.data.Dataset):

    def __init__(self, titles, labels):

        self.titles = titles

        self.labels = labels

        

    def __len__(self):

        return self.labels.shape[0]

    

    def __getitem__(self, index):

        x = self.titles[index, :]

        y = self.labels[index, :]

        return x, y
train_data = arxiv_dataset(x_train, y_train)
train_gen = torch.utils.data.DataLoader(train_data, batch_size=128)
test_data = arxiv_dataset(x_test, y_test)
test_gen = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)
class BERT(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = BertForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased",

                                                                    output_hidden_states=True)

        for param in self.encoder.parameters():

            param.requires_grad = False

            

        self.dense_1 = nn.Linear(768, 384)

        self.dense_2 = nn.Linear(384, 176)

        

    def forward(self, tokens):

        hidden_states = self.encoder(tokens)[1][-1][:, 0]

        x = self.dense_1(hidden_states)

        x = self.dense_2(x)

        return x
model = BERT()

model = model.cuda()
for toks, _ in train_gen:

    print(model(toks.cuda()).shape)

    break
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 10
train_loss = []

for epoch in range(EPOCHS):

    running_loss = 0.0

    num_batches = 0

    for data in train_gen:

        inputs, labels = data

        inputs = inputs.cuda()

        labels = labels.cuda()

        

        #Zero the gradients from last step

        optimizer.zero_grad()

        logits = model(inputs)

        #Calculate BCE with logits

        loss = criterion(logits, labels)

        #Back prop and optimizer step

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        num_batches += 1

        

        #Removing tensors from GPU to free up memory

        del inputs

        del labels

        del logits

        torch.cuda.empty_cache()

        

    train_loss.append(running_loss / num_batches)

        
print(train_loss)
torch.save(model.state_dict(), '10_21_1937.pt')