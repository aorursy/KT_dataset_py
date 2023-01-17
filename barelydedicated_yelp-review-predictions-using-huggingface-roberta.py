!pip install transformers
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from transformers import *

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

from transformers.data.processors.utils import InputExample

from torch import nn

from torch.autograd import Variable

from torch.utils.data import TensorDataset, DataLoader

from torch.optim import Adam

from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm, tnrange 

import time 
reader = pd.read_json('/kaggle/input/yelp-dataset/yelp_academic_dataset_review.json', lines=True, chunksize=10000)
def generate_dataset(json_reader, nrows=10000, min_length=100, max_length = 128):

    df = None

    while True:

        df_candidate = next(json_reader)

        df_candidate = df_candidate.loc[(df_candidate['text'].str.len() > min_length) & (df_candidate['text'].str.len() <= max_length), ['text', 'stars']]

        if df is None:

            df = df_candidate

        else:

            df = df.append(df_candidate)

        for rating in range(1, 6, 1):

            df_rating = df[df['stars'] == rating]

            if len(df_rating) > nrows//5:

                df_rating = df_rating.iloc[:nrows//5, :]

                df = df.loc[~(df['stars'] == rating), :]

                df = df.append(df_rating)

        if len(df) == nrows:

            return df
train_df = generate_dataset(reader)

test_df = generate_dataset(reader)
train_df['stars'].value_counts()
test_df['stars'].value_counts()
kwargs = {'cumulative': True}

plt.hist(train_df['text'].str.len().tolist() + test_df['text'].str.len().tolist(),cumulative=True, density=True, bins=40)

plt.xlim(left=90, right=130) 

plt.show()
train_df.head()
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
hidden_size = 200
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5)





FINE_TUNE = True

print(f'Total model trainable parameters {count_parameters(model)}')

if FINE_TUNE:

    for param in model.roberta.parameters():

        param.requires_grad = False



    for param in model.classifier.parameters():

        param.requires_grad = True

    print(f'Total head trainable parameters {count_parameters(model)}')

model.cuda();
model.classifier
tokenized = tokenizer.tokenize(' I am parachuting with you')

print(tokenized)

print(tokenizer.encode(tokenized, add_special_tokens=False))

print(tokenizer.encode(tokenized, add_special_tokens=True))
# https://huggingface.co/transformers/main_classes/processors.html

def get_features(df, text_col, label_col):

    l = [InputExample(guid=idx, text_a=df.loc[idx, text_col], label=df.loc[idx, label_col]) for 

       idx, row in tqdm(df.iterrows(), total=df.shape[0])]

    features = glue_convert_examples_to_features(examples=l, 

                                    tokenizer=tokenizer,

                                    max_length=300,

                                    label_list = df[label_col].values,

                                    output_mode='regression')



    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)

    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    all_labels = torch.tensor([f.label-1 for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)

    return dataset
train_dataset = get_features(train_df, 'text', 'stars')

test_dataset = get_features(test_df, 'text', 'stars')
val_idx, train_idx = train_test_split(np.arange(len(train_dataset)), random_state=4, train_size=0.1)

total_size = len(train_dataset)

val_dataset = TensorDataset(*train_dataset[val_idx])

train_dataset = TensorDataset(*train_dataset[train_idx])

assert total_size == len(val_dataset) + len(train_dataset)
# works

model(input_ids=train_dataset[:2][0].cuda(), 

      attention_mask=train_dataset[:2][1].cuda(), 

      labels=train_dataset[:2][2].cuda());
batch_size = 16

gradient_every = 32

assert batch_size <= gradient_every and gradient_every % batch_size == 0



accumulation_steps = gradient_every//batch_size



train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)



epochs = 25



lr = 0.002

optimizer = AdamW(model.classifier.parameters(), lr=lr)  
tr_losses = []

v_losses = []
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_data_loader)*epochs)

for epoch in tnrange(epochs, desc='epoch'):

    """ Training stage """

    epoch_tr_losses = []

    print(f'epoch {epoch+1}')

    for k, (input_ids, attention_mask, labels) in enumerate(tqdm(train_dataloader, total=len(train_dataloader), desc='batch')):

        feed_dict = {'input_ids': input_ids.cuda(),

                     'attention_mask': attention_mask.cuda(),

                     'labels': labels.cuda()}

        

        loss, _ = model(**feed_dict)



        # gradient accumulation

        epoch_tr_losses.append(loss.item())

        loss = loss/accumulation_steps

        loss.backward()

        if (k + 1) % accumulation_steps == 0:

            optimizer.step()

            model.zero_grad()



    tr_losses.append(np.mean(epoch_tr_losses))

    print(f'train NLL loss: {np.mean(epoch_tr_losses)}')

  

    """ Validation stage """

    epoch_v_losses = [] 

    with torch.no_grad():

        for k, (input_ids, attention_mask, labels) in enumerate(tqdm(val_dataloader, total=len(val_dataloader), desc='val batch')):

            feed_dict = {'input_ids': input_ids.cuda(),

                         'attention_mask': attention_mask.cuda(),

                         'labels': labels.cuda()} 



            loss, pred = model(**feed_dict)

            epoch_v_losses.append(loss.item())

        v_losses.append(np.mean(epoch_v_losses))

    print(f'validation BCE loss: {np.mean(epoch_v_losses)}')

    torch.save(model.classifier.state_dict(), f'/kaggle/working/yelp-head{epoch}.pt')
plt.plot(tr_losses)

plt.plot(v_losses)
batch_predictions, batch_actual = [], []

with torch.no_grad():

    for k, (input_ids, attention_mask, labels) in enumerate(tqdm(test_dataloader, total=len(test_dataloader), desc='val batch')):

        feed_dict = {'input_ids': input_ids.cuda(),

                     'attention_mask': attention_mask.cuda()} 

        

        pred = model(**feed_dict)[0].cpu()

        batch_predictions.append(pred.numpy())

        batch_actual.append(labels)
predictions = np.array([i for k in batch_predictions for i in k ])



predictions = np.argmax(predictions, axis=1)

actual = np.array([i for k in batch_actual for i in k ])
from sklearn.metrics import f1_score

f1_score(actual, predictions, average='micro')
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')
# compute the confusion matrix

from sklearn.metrics import confusion_matrix

import itertools

confusion_mtx = confusion_matrix(actual, predictions) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(1,6))

plt.show()