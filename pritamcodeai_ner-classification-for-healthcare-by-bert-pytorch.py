import pandas as pd

import dask.dataframe as dd

import time

import numpy as np 

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from keras.models import Model

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

from keras.preprocessing import text, sequence

from keras.callbacks import Callback

import warnings

warnings.filterwarnings('ignore')



import numpy as np 

import pandas as pd 



import os



from tqdm import tqdm, trange
!pip install pytorch-pretrained-bert==0.6.1  
!pip install seqeval==0.0.6  
print(os.listdir("../input"))
train_file_path = os.listdir("../input/ner-train-dataset")
t=time.clock()

train = dd.read_csv("../input/ner-train-dataset/train.csv", usecols=['Sent_ID', 'Word', 'tag', ])

train= train.compute()

print("load train: " , time.clock()-t)
df1 = train

df1.head(5)
df1.plot(kind='hist', bins=600, figsize=[12,10], alpha=.4, legend=True)  
df1.describe()
volume_tag = df1['tag'].value_counts()

volume_tag
df1[0:410].plot.bar(rot=0, subplots=True)
input_data = train.fillna(method="ffill")

input_data.tail(10)
words_list = list(set(input_data["Word"].values))

words_list[:10]
number_words = len(words_list); 

number_words 
class RetrieveSentance(object):

    

    def __init__(self, data):

        self.n_sent = 1

        self.data = data

        self.empty = False

        function = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),s["tag"].values.tolist())]

        self.grouped = self.data.groupby("Sent_ID").apply(function)

        self.sentences = [s for s in self.grouped]

    

    def retrieve(self):

        try:

            s = self.grouped["Sentence: {}".format(self.n_sent)]

            self.n_sent += 1

            return s

        except:

            return None
Sentences = RetrieveSentance(input_data)

Sentences
Sentences_list = [" ".join([s[0] for s in sent]) for sent in Sentences.sentences]

Sentences_list[0]
len(Sentences_list)
labels = [[s[1] for s in sent] for sent in Sentences.sentences]

print(labels[0])
labels [0]
tags2vals = list(set(input_data["tag"].values))

tag2idx = {t: i for i, t in enumerate(tags2vals)}
tags2vals
tag2idx
import torch

from torch.optim import Adam

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from pytorch_pretrained_bert import BertTokenizer, BertConfig

from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
max_seq_len = 36

batch_s = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gpu = torch.cuda.device_count()

torch.cuda.get_device_name(0) 
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in Sentences_list]

print(tokenized_texts[0])
len(tokenized_texts)
print(tokenized_texts[1])
X = pad_sequences([tokenizer.convert_tokens_to_ids(txt[0:36]) for txt in tokenized_texts], maxlen=36, dtype="long", truncating="post", padding="post")
Y = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels], maxlen=max_seq_len, value=tag2idx["O"], padding="post", dtype="long", truncating="post")
X.shape
Y.shape
X
Y
attention_masks = [[float(i>0) for i in ii] for ii in X]
len(attention_masks) 
attention_masks[0]
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, random_state=20, test_size=0.33) 
Mask_train, Mask_valid, _, _ = train_test_split(attention_masks, X, random_state=20, test_size=0.33)
X_train = torch.tensor(X_train)

X_train = X_train.long()

X_valid = torch.tensor(X_valid)

X_valid = X_valid.long()

Y_train = torch.tensor(Y_train)

Y_train = Y_train.long()

Y_valid = torch.tensor(Y_valid)

Y_valid = Y_valid.long()
Mask_train = torch.tensor(Mask_train)

Mask_train = Mask_train.long()

Mask_valid = torch.tensor(Mask_valid)

Mask_valid = Mask_valid.long()
data_train = TensorDataset(X_train, Mask_train, Y_train)

data_train_sampler = RandomSampler(data_train)

DL_train = DataLoader(data_train, sampler=data_train_sampler, batch_size=batch_s)
data_valid = TensorDataset(X_valid, Mask_valid, Y_valid)

data_valid_sampler = SequentialSampler(data_valid)

DL_valid = DataLoader(data_valid, sampler=data_valid_sampler, batch_size=batch_s)
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(tag2idx))
model.cuda();
FULL_FINETUNING = True

if FULL_FINETUNING:

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],

         'weight_decay_rate': 0.01},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],

         'weight_decay_rate': 0.0}

    ]

else:

    param_optimizer = list(model.classifier.named_parameters()) 

    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
from seqeval.metrics import f1_score



def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=2).flatten()

    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)
epochs = 5

max_grad_norm = 1.0



for _ in trange(epochs, desc="Epoch"):

    # TRAIN loop

    model.train()

    tr_loss = 0

    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(DL_train):

        # add batch to gpu

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        # forward pass

        loss = model(b_input_ids, token_type_ids=None,

                     attention_mask=b_input_mask, 

                     labels=b_labels)

        # backward pass

        loss.backward()

        # track train loss

        tr_loss += loss.item()

        nb_tr_examples += b_input_ids.size(0)

        nb_tr_steps += 1

        # gradient clipping

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)

        # update parameters

        optimizer.step()

        model.zero_grad()

    # print train loss per epoch

    print("Train loss: {}".format(tr_loss/nb_tr_steps))

    # VALIDATION on validation set

    model.eval()

    eval_loss, eval_accuracy = 0, 0

    nb_eval_steps, nb_eval_examples = 0, 0

    predictions , true_labels = [], []

    for batch in DL_valid:

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        

        with torch.no_grad():

            tmp_eval_loss = model(b_input_ids, token_type_ids=None,

                                  attention_mask=b_input_mask, 

                                  labels=b_labels)

            logits = model(b_input_ids, token_type_ids=None,

                           attention_mask=b_input_mask)

        logits = logits.detach().cpu().numpy()

        label_ids = b_labels.to("cpu").numpy()

        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

        true_labels.append(label_ids)

        

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        

        eval_loss += tmp_eval_loss.mean().item()

        eval_accuracy += tmp_eval_accuracy

        

        nb_eval_examples += b_input_ids.size(0)

        nb_eval_steps += 1

    eval_loss = eval_loss/nb_eval_steps

    print("Validation loss: {}".format(eval_loss))

    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

    pred_tags = [tags2vals[p_i] for p in predictions for p_i in p]

    valid_tags = [tags2vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]

    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
model.eval()

predictions = []

true_labels = []

eval_loss, eval_accuracy = 0, 0

nb_eval_steps, nb_eval_examples = 0, 0

for batch in DL_valid:

    batch = tuple(t.to(device) for t in batch)

    b_input_ids, b_input_mask, b_labels = batch



    with torch.no_grad():

        tmp_eval_loss = model(b_input_ids, token_type_ids=None,

                              attention_mask=b_input_mask, labels=b_labels)

        logits = model(b_input_ids, token_type_ids=None,

                       attention_mask=b_input_mask)

        

    logits = logits.detach().cpu().numpy()

    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

    label_ids = b_labels.to('cpu').numpy()

    true_labels.append(label_ids)

    tmp_eval_accuracy = flat_accuracy(logits, label_ids)



    eval_loss += tmp_eval_loss.mean().item()

    eval_accuracy += tmp_eval_accuracy



    nb_eval_examples += b_input_ids.size(0)

    nb_eval_steps += 1



pred_tags = [[tags2vals[p_i] for p_i in p] for p in predictions]

valid_tags = [[tags2vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l ]

print("Validation loss: {}".format(eval_loss/nb_eval_steps))

print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))