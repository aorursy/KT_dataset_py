# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

dataPath = '/kaggle/input/codemlsentimentanalysis/'

# Any results you write to the current directory are saved as output.
train_data_input = pd.read_csv(dataPath+'train.csv')

test_data_input = pd.read_csv(dataPath+'test_nolabel.csv')
training_device = 'GPU'

model_size = 'base'

need_training = True
!pip install transformers
import os

import re

import io

import collections

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn.metrics as metrics

import torch

import torch.nn as nn

from transformers import XLNetModel, XLNetConfig, XLNetTokenizer, XLNetForSequenceClassification, AdamW, XLNetPreTrainedModel, modeling_utils



from IPython.display import FileLink 

from sklearn.model_selection import train_test_split

from multiprocessing import Pool

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split

# from pytorch_transformers import XLNetModel, XLNetConfig, XLNetTokenizer, XLNetForSequenceClassification, AdamW, XLNetPreTrainedModel, modeling_utils

from tqdm import tqdm, trange
if training_device == 'CPU':

  # Setup GPU

  device = torch.device("cpu")

#   n_gpu = torch.cuda.device_count()

#   torch.cuda.get_device_name(0)

elif training_device == 'TPU':

  assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'
if training_device == 'GPU':

  # Setup GPU

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  n_gpu = torch.cuda.device_count()

  torch.cuda.get_device_name(0)

elif training_device == 'TPU':

  assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'
# Addition of special tokens



# On train

train_data_input['twit'] = train_data_input['twit'] + ' <sep> <cls>'

# train_data.head()



# On test

test_data_input['twit'] = test_data_input['twit'] +' <sep> <cls>'

test_data.head()
train_data_input.head()
# Tokenize with XLNet tokenizer

tokenizer = XLNetTokenizer.from_pretrained('xlnet-'+model_size+'-cased', do_lower_case=False) # lower case false, because it's cased XLnet (let's try with True also)



train_sentences_tokenized = train_data_input['twit'].apply(lambda sentence: tokenizer.tokenize(sentence))

test_sentences_tokenized = test_data_input['twit'].apply(lambda sentence: tokenizer.tokenize(sentence))

# Il y a des '.' seulemt dans certaines phrases, on les a donc retires
print(max([len(sentence) for sentence in train_sentences_tokenized]))

print(max([len(sentence) for sentence in test_sentences_tokenized]))
train_data_labels_num = train_data_input.target.values
MAX_LEN = 128
train_attention_masks = [[1]*len(s) + [0]*(MAX_LEN-len(s)) for s in train_sentences_tokenized]

test_attention_masks = [[1]*len(s) + [0]*(MAX_LEN-len(s)) for s in test_sentences_tokenized]
segments = { 'twit' : 0, 'sep': 1, 'cls': 2, 'pad': 3 }



train_SEP_index = [ sentence.index('<sep>') for sentence in train_sentences_tokenized ]

train_token_type_ids =  [ [segments['twit']]*(train_SEP_index[i])

                        + [segments['sep']]

                        + [segments['cls']]

                        + [segments['pad']]*(MAX_LEN-len(train_sentences_tokenized[i]))

                        for i in range(len(train_sentences_tokenized)) ]



test_SEP_index = [ sentence.index('<sep>') for sentence in test_sentences_tokenized ]

test_token_type_ids = [ [segments['twit']]*(test_SEP_index[i])

                      + [segments['sep']]

                      + [segments['cls']]

                      + [segments['pad']]*(MAX_LEN-len(test_sentences_tokenized[i]))

                      for i in range(len(test_sentences_tokenized)) ]
train_sentences_tokenized = [ sentence + ['<pad>']*(MAX_LEN - len(sentence)) for sentence in train_sentences_tokenized ]

test_sentences_tokenized = [ sentence + ['<pad>']*(MAX_LEN - len(sentence)) for sentence in test_sentences_tokenized ]
train_sentences_tokenized_ids = [tokenizer.convert_tokens_to_ids(tokenized_sentence) for tokenized_sentence in train_sentences_tokenized]

test_sentences_tokenized_ids = [tokenizer.convert_tokens_to_ids(tokenized_sentence) for tokenized_sentence in test_sentences_tokenized]
# Split into train and validation data

validation_size = 0



# # Don't use train test split because sentences are grouped by same sentence1, and we don't want to validate on sentences that already are in the train

# train_inputs, validation_inputs = train_sentences_tokenized_ids[:-validation_size], train_sentences_tokenized_ids[-validation_size:]

# train_labels, validation_labels = train_data_labels_num[:-validation_size], train_data_labels_num[-validation_size:]

# train_masks, validation_masks = train_attention_masks[:-validation_size], train_attention_masks[-validation_size:]

# train_token_type_ids, validation_token_type_ids = train_token_type_ids[:-validation_size], train_token_type_ids[-validation_size:]



train_inputs = train_sentences_tokenized_ids

train_labels = train_data_labels_num

train_masks = train_attention_masks

train_token_type_ids = train_token_type_ids
# Transform to tensor



train_inputs = torch.tensor(train_inputs)

# validation_inputs = torch.tensor(validation_inputs)

test_inputs = torch.tensor(test_sentences_tokenized_ids)





train_labels = torch.tensor(train_labels)

# validation_labels = torch.tensor(validation_labels)



train_masks = torch.tensor(train_masks)

# validation_masks = torch.tensor(validation_masks)

test_masks = torch.tensor(test_attention_masks)



train_token_type_ids = torch.tensor(train_token_type_ids)

# validation_token_type_ids = torch.tensor(validation_token_type_ids)

test_token_type_ids = torch.tensor(test_token_type_ids)
batch_size = 32
# In order not to use too much memory, use a DataLoader



train_data = TensorDataset(train_inputs, train_masks, train_labels, train_token_type_ids)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)



# validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, validation_token_type_ids)

# validation_sampler = SequentialSampler(validation_data)

# validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)



test_data = TensorDataset(test_inputs, test_masks, test_token_type_ids)

test_sampler = SequentialSampler(test_data)

test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
# Retrieve pre-trained model

model = XLNetForSequenceClassification.from_pretrained('xlnet-'+model_size+'-cased')

model.cuda()
param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'gamma', 'beta']

optimizer_grouped_parameters = [

  {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],

    'weight_decay_rate': 0.01},

  {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],

    'weight_decay_rate': 0.0}

]

optimizer = AdamW(optimizer_grouped_parameters,

                     lr=2e-5)
def flat_accuracy(preds, labels):

  pred_flat = np.argmax(preds, axis=1).flatten()

  labels_flat = labels.flatten()

  return np.sum(pred_flat == labels_flat) / len(labels_flat)
if need_training:

  # Number of training epochs (authors recommend between 2 and 4)

  epochs = 3



  # Store our loss and accuracy for plotting

  train_loss_set = [[] for i in range(epochs)]



  # trange is a tqdm wrapper around the normal python range

  for e in trange(epochs, desc="Epoch"):



    # Training



    # Set our model to training mode (as opposed to evaluation mode)

    model.train()



    # Tracking variables

    tr_loss = 0

    nb_tr_examples, nb_tr_steps = 0, 0



    # Train the data for one epoch

    for step, batch in enumerate(train_dataloader):

      

      # Add batch to GPU

      batch = tuple(t.to(device) for t in batch)

      

      # Unpack the inputs from our dataloader

      b_input_ids, b_input_mask, b_labels, b_token_type_ids = batch

      # Clear out the gradients (by default they accumulate)

      optimizer.zero_grad()

      # Forward pass

      outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask, labels=b_labels)

      loss = outputs[0]

      logits = outputs[1]

      print(step, loss.item())

      train_loss_set[e].append(loss.item())

      # Backward pass

      loss.backward()

      # Update parameters and take a step using the computed gradient

      optimizer.step()





      # Update tracking variables

      tr_loss += loss.item()

      nb_tr_examples += b_input_ids.size(0)

      nb_tr_steps += 1

      # break



    print("Train loss: {}".format(tr_loss/nb_tr_steps))





    # Validation



    # Put model in evaluation mode to evaluate loss on the validation set

#     model.eval()



#     # Tracking variables 

#     eval_loss, eval_accuracy = 0, 0

#     nb_eval_steps, nb_eval_examples = 0, 0



#     # Evaluate data for one epoch

#     for batch in validation_dataloader:

#       # Add batch to GPU

#       batch = tuple(t.to(device) for t in batch)

#       # Unpack the inputs from our dataloader

#       b_input_ids, b_input_mask, b_labels, b_token_type_ids = batch

#       # Telling the model not to compute or store gradients, saving memory and speeding up validation

#       with torch.no_grad():

#         # Forward pass, calculate logit predictions

#         outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)

#         logits = outputs[0]



#       # Move logits and labels to CPU

#       logits = logits.detach().cpu().numpy()

#       label_ids = b_labels.to('cpu').numpy()



#       tmp_eval_accuracy = flat_accuracy(logits, label_ids)



#       eval_accuracy += tmp_eval_accuracy

#       nb_eval_steps += 1



#     print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

  

#     torch.save(model.state_dict(), gdrive_path + model_name+str(e)+'.pt')


# Prediction on test set



# Put model in evaluation mode

model.eval()



# Tracking variables 

predictions = []



# Predict 

for batch in test_dataloader:

  # Add batch to GPU

  batch = tuple(t.to(device) for t in batch)

  # Unpack the inputs from our dataloader

  b_input_ids, b_input_mask, b_token_type_ids = batch

  # Telling the model not to compute or store gradients, saving memory and speeding up prediction

  with torch.no_grad():

    # Forward pass, calculate logit predictions

    outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)

    logits = outputs[0]



  # Move logits and labels to CPU

  logits = logits.detach().cpu().numpy()



  # Store predictions and true labels

  predictions.append(logits)
test_predictions_labels_num_tri = [list(item) for sublist in predictions for item in sublist]

test_predictions_labels_num = np.argmax(test_predictions_labels_num_tri, axis=1).flatten()

# test_predictions_labels = [labels[i] for i in test_predictions_labels_num]
test_data = pd.read_csv(dataPath+'test_nolabel.csv')

test_data['Target'] = test_predictions_labels_num+0.0

submission = test_data[["Target","ID"]]

submission.head()
submission.to_csv("submission.csv",index=False)