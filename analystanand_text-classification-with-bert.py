!pip install transformers
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from transformers import BertTokenizer, BertModel, BertForMaskedLM





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import torch

from transformers import *

from transformers import BertTokenizer, BertModel,BertForSequenceClassification,AdamW

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from tqdm import tqdm, trange

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':

  raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gpu = torch.cuda.device_count()

torch.cuda.get_device_name(0)
# Set the maximum sequence length.

MAX_LEN = 128
sample_submission = pd.read_csv("../input/quora-insincere-questions-classification/sample_submission.csv")

all_test = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")

all_train = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")
sample_train =  all_train.loc[:10000,:]
#creates list of sentences and labels

sentences = sample_train.question_text.values

labels = sample_train.target.values



#initialize BERT tokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)



#user tokenizer to convert sentences into tokenizer

input_ids  = [tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN) for sent in sentences]



# Pad our input tokens

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")



# Create attention masks

attention_masks = []



# Create a mask of 1s for each token followed by 0s for padding

for seq in input_ids:

  seq_mask = [float(i>0) for i in seq]

  attention_masks.append(seq_mask)
# Use train_test_split to split our data into train and validation sets for training



train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 

                                                            random_state=2018, test_size=0.1)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,

                                             random_state=2018, test_size=0.1)
# Convert all of our data into torch tensors, the required datatype for our model



train_inputs = torch.tensor(train_inputs)

validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)

validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)

validation_masks = torch.tensor(validation_masks)


# Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32

batch_size = 16



# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 

# with an iterator the entire dataset does not need to be loaded into memory



train_data = TensorDataset(train_inputs, train_masks, train_labels)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)



validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)

validation_sampler = SequentialSampler(validation_data)

validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 



model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

model.cuda()
param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'gamma', 'beta']

optimizer_grouped_parameters = [

    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],

     'weight_decay_rate': 0.01},

    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],

     'weight_decay_rate': 0.0}

]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=10e-8)


# Function to calculate the accuracy of our predictions vs labels

def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()

    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# # Store our loss and accuracy for plotting

train_loss_set = []



# # Number of training epochs (authors recommend between 2 and 4)

epochs = 5



# trange is a tqdm wrapper around the normal python range

for _ in trange(epochs, desc="Epoch"):

  

  

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

    b_input_ids, b_input_mask, b_labels = batch

    # Clear out the gradients (by default they accumulate)

    optimizer.zero_grad()

    # Forward pass

    # loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

    outputs = model(b_input_ids,token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

    loss = outputs[0]

    train_loss_set.append(loss.item())    

    # Backward pass

    loss.backward()

    # Update parameters and take a step using the computed gradient

    optimizer.step()

    

    

    # Update tracking variables

    tr_loss += loss.item()

    nb_tr_examples += b_input_ids.size(0)

    nb_tr_steps += 1



  print("Train loss: {}".format(tr_loss/nb_tr_steps))

    

    

  # Validation



  # Put model in evaluation mode to evaluate loss on the validation set

  model.eval()



  # Tracking variables 

  eval_loss, eval_accuracy = 0, 0

  nb_eval_steps, nb_eval_examples = 0, 0



  # Evaluate data for one epoch

  for batch in validation_dataloader:

    # Add batch to GPU

    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader

    b_input_ids, b_input_mask, b_labels = batch

    # Telling the model not to compute or store gradients, saving memory and speeding up validation

    with torch.no_grad():

      # Forward pass, calculate logit predictions

      outputs =  model(b_input_ids,token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

      loss, logits = outputs[:2]

    

    # Move logits and labels to CPU

    logits = logits.detach().cpu().numpy()

    label_ids = b_labels.to('cpu').numpy()



    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    

    eval_accuracy += tmp_eval_accuracy

    nb_eval_steps += 1



  print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
plt.figure(figsize=(15,8))

plt.title("Training loss")

plt.xlabel("Batch")

plt.ylabel("Loss")

plt.plot(train_loss_set)

plt.show()
#creates list of sentences and labels

sentences = all_test.question_text.values

qids = all_test.qid.values



#user tokenizer to convert sentences into tokenizer

input_ids  = [tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN) for sent in sentences]



# Pad our input tokens

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")



# Create attention masks

attention_masks = []



# Create a mask of 1s for each token followed by 0s for padding

for seq in input_ids:

  seq_mask = [float(i>0) for i in seq]

  attention_masks.append(seq_mask)



prediction_inputs = torch.tensor(input_ids)

prediction_masks = torch.tensor(attention_masks)



batch_size = 16 





prediction_data = TensorDataset(prediction_inputs, prediction_masks,)

prediction_sampler = SequentialSampler(prediction_data)

prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
# Prediction on testing set

model.eval()

flat_pred = []

for batch in prediction_dataloader:

  # Add batch to GPU

  batch = tuple(t.to(device) for t in batch)

  # Unpack the inputs from our dataloader

  b_input_ids, b_input_mask = batch

  # Telling the model not to compute or store gradients, saving memory and speeding up prediction

  with torch.no_grad():

    # Forward pass, calculate logit predictions

    outputs =  model(b_input_ids,token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs[0]

    logits = logits.detach().cpu().numpy() 

    flat_pred.extend(np.argmax(logits, axis=1).flatten())
pred_df = pd.DataFrame(columns=["qid","prediction"])

pred_df["qid"] = qids

pred_df["prediction"]  = flat_pred

pred_df.to_csv("submission.csv",index=False)