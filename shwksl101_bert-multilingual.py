# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import csv



from tqdm import tqdm



import torch

from torch.utils.data import TensorDataset, DataLoader



import transformers

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction

from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed
config = AutoConfig.from_pretrained('bert-base-multilingual-cased', num_labels=2)

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

model = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-cased', config=config)
data = {'id': [],

        'sentence': [],

        'label': []}



# 파일 불러오기

with open("/kaggle/input/2020-lg-ai-camp-nlp-final-project-2/ratings_train.csv", 'r') as f:

    train = csv.reader(f)

    for i, example in enumerate(train):

        if i == 0:

            continue

        data['id'].append(example[0])

        data['sentence'].append(example[1])

        data['label'].append(int(example[2]))
tokenized_output = tokenizer(data['sentence'], padding=True, return_tensors='pt')
all_input_ids = tokenized_output.data['input_ids']

all_token_type_ids = tokenized_output.data['token_type_ids']

attention_mask = tokenized_output.data['attention_mask']

label = torch.tensor(data['label'], dtype=torch.long)
train_dataset = TensorDataset(all_input_ids, all_token_type_ids, attention_mask, label)
BATCH_SIZE = 50

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
EPOCHS = 1

model.cuda()



for epoch in range(EPOCHS):

    losses = []

    for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

        batch = tuple(t.cuda() for t in batch)

        inputs = {'input_ids': batch[0],

                  'token_type_ids': batch[1],

                  'attention_mask': batch[2],

                  'labels': batch[3]}



        output = model(**inputs)

        

        loss = output[0]

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        losses.append(np.array(loss.cpu().detach()))

        if batch_idx % 100 == 0:

            print(batch_idx, "Current loss:", np.mean(losses))

            losses = []
# Evaluate

data = {'id': [],

        'sentence': [],

        'label': []}



# 파일 불러오기

with open("/kaggle/input/2020-lg-ai-camp-nlp-final-project-2/ratings_test.csv", 'r') as f:

    train = csv.reader(f)

    for i, example in enumerate(train):

        if i == 0:

            continue

        data['id'].append(example[0])

        data['sentence'].append(example[1])
tokenized_output = tokenizer(data['sentence'], padding=True, return_tensors='pt')



all_input_ids = tokenized_output.data['input_ids']

all_token_type_ids = tokenized_output.data['token_type_ids']

attention_mask = tokenized_output.data['attention_mask']



test_dataset = TensorDataset(all_input_ids, all_token_type_ids, attention_mask)



BATCH_SIZE = 100

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
predictions = []



with torch.no_grad():

    for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):

            batch = tuple(t.cuda() for t in batch)

            inputs = {'input_ids': batch[0],

                      'token_type_ids': batch[1],

                      'attention_mask': batch[2]}



            output = model(**inputs)

            

            logits = output[0]

            predictions += torch.argmax(logits, dim=1)
predictions = pd.DataFrame(predictions)

predictions.to_csv('sampleSubmission.csv', index=False)