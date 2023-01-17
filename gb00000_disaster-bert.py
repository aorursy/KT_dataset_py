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
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

samp_sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
test
samp_sub


import tensorflow as tf



device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':

  raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))


!pip install pytorch-pretrained-bert pytorch-nlp





import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from pytorch_pretrained_bert import BertTokenizer, BertConfig

from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification

from tqdm import tqdm, trange

import pandas as pd

import io

import numpy as np

import matplotlib.pyplot as plt







device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gpu = torch.cuda.device_count()

torch.cuda.get_device_name(0)
train
tweet_lens = [len(i) for i in train['text']]
MAX_LEN = max(tweet_lens)
labels = train['target'].values
nb_labels = len(labels)
sentences = ["[CLS] " + query + " [SEP]" for query in train['text']]

sentences_test = ["[CLS] " + query + " [SEP]" for query in test['text']]

print(sentences[0])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

tokenized_texts_test = [tokenizer.tokenize(sent) for sent in sentences_test]

print ("Tokenize the first sentence:")

print (tokenized_texts[0])
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],

                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

input_ids_test = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts_test],

                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

input_ids_test = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts_test]

input_ids_test = pad_sequences(input_ids_test, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
attention_masks = []

attention_masks_test = []

for seq in input_ids:

  seq_mask = [float(i>0) for i in seq]

  attention_masks.append(seq_mask)

for seq in input_ids_test:

  seq_mask = [float(i>0) for i in seq]

  attention_masks_test.append(seq_mask)

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 

                                                            random_state=2018, test_size=0.1)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,

                                             random_state=2018, test_size=0.1)
train_inputs = torch.tensor(train_inputs)

validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)

validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)

validation_masks = torch.tensor(validation_masks)

test_masks_sub = torch.tensor(attention_masks_test)

test_inputs_sub = torch.tensor(input_ids_test)

train_masks_sub = torch.tensor(attention_masks)

train_inputs_sub = torch.tensor(input_ids)

train_labels_sub = torch.tensor(labels)
batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)

validation_sampler = SequentialSampler(validation_data)

validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=nb_labels)

model.cuda()

param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'gamma', 'beta']

optimizer_grouped_parameters = [

    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],

     'weight_decay_rate': 0.01},

    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],

     'weight_decay_rate': 0.0}

]
optimizer = BertAdam(optimizer_grouped_parameters,

                     lr=2e-5,

                     warmup=.1)
def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()

    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)
train_loss_set = []



epochs = 4





for _ in trange(epochs, desc="Epoch"):  

  



  



    model.train()  



    tr_loss = 0

    nb_tr_examples, nb_tr_steps = 0, 0



    for step, batch in enumerate(train_dataloader):



        batch = tuple(t.to(device) for t in batch)



        b_input_ids, b_input_mask, b_labels = batch



        optimizer.zero_grad()



        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        train_loss_set.append(loss.item())    



        loss.backward()



        optimizer.step()



        tr_loss += loss.item()

        nb_tr_examples += b_input_ids.size(0)

        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss/nb_tr_steps))

    model.eval()



    eval_loss, eval_accuracy = 0, 0

    nb_eval_steps, nb_eval_examples = 0, 0



    for batch in validation_dataloader:



        batch = tuple(t.to(device) for t in batch)



        b_input_ids, b_input_mask, b_labels = batch



        with torch.no_grad():



            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)    



        logits = logits.detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)    

        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))    
#results would imply that training the full model shouldn't be done for more than 1 epoch, but we'll train for longer anyways and the final results can be a test of how many epochs to use
'''model.eval()



eval_loss, eval_accuracy = 0, 0

nb_eval_steps, nb_eval_examples = 0, 0



for batch in validation_dataloader:



    batch = tuple(t.to(device) for t in batch)



    b_input_ids, b_input_mask, b_labels = batch



    with torch.no_grad():



        logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)    

 

    logits = logits.detach().cpu().numpy()

    label_ids = b_labels.to('cpu').numpy()

    tmp_eval_accuracy = flat_accuracy(logits, label_ids)    

    eval_accuracy += tmp_eval_accuracy

    nb_eval_steps += 1

print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))'''

'''plt.figure(figsize=(15,8))

plt.title("Training loss")

plt.xlabel("Batch")

plt.ylabel("Loss")

plt.plot(train_loss_set)

plt.show()'''
train_data_sub = TensorDataset(train_inputs_sub, train_masks_sub, train_labels_sub)



train_dataloader_sub = DataLoader(train_data_sub, batch_size=batch_size)



test_data_sub = TensorDataset(test_inputs_sub, test_masks_sub)



test_dataloader_sub = DataLoader(test_data_sub, batch_size=batch_size)

train_loss_set = []



epochs = 10





for _ in trange(epochs, desc="Epoch"):  

  



  



    model.train()  



    tr_loss = 0

    nb_tr_examples, nb_tr_steps = 0, 0



    for step, batch in enumerate(train_dataloader_sub):



        batch = tuple(t.to(device) for t in batch)



        b_input_ids, b_input_mask, b_labels = batch



        optimizer.zero_grad()



        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        train_loss_set.append(loss.item())    



        loss.backward()



        optimizer.step()



        tr_loss += loss.item()

        nb_tr_examples += b_input_ids.size(0)

        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss/nb_tr_steps))
model.eval()



targs = []



for batch in test_dataloader_sub:



    batch = tuple(t.to(device) for t in batch)



    b_input_ids, b_input_mask = batch



    with torch.no_grad():



        logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)    

 

    logits = logits.detach().cpu().numpy()

    

    targs.append(logits)



pred_flat = [np.argmax(i, axis=1).flatten().tolist() for i in targs]
#pred_flat
from functools import reduce

topc_list1 = reduce(lambda x,y: x+y, pred_flat)
#topc_list1
sub=pd.DataFrame({'id':samp_sub['id'].values.tolist(),'target':topc_list1})
sub.to_csv('submission.csv',index=False)