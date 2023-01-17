!pip install pytorch-pretrained-bert==0.4.0

!pip install transformers

!pip install seqeval
import pandas as pd

import numpy as np

import re

import string

male_name=pd.read_csv("../input/indian-names/Indian-Male-Names.csv")

female_name=pd.read_csv("../input/indian-names/Indian-Female-Names.csv")

data = pd.read_csv("../input/entity-annotated-corpus/ner_dataset.csv", encoding="latin1")
data.tail()
male_name.head()
male_list=[y for x in np.array(male_name.name) for y in str(x).split()]

female_list=[y for x in np.array(female_name.name) for y in str(x).split()]

(len(male_list),len(female_list))
# removing dublicate names

name_list= list(set((male_list+female_list)))

# removing invailid names

name_list= [re.sub('[^a-zA-Z]+', '', x) for x in name_list if len(x) > 1]

name_list= list(set([x for x in name_list if len(x) > 4]))

# name_list=[ for x in name_list]

len(name_list)
# check if name is present in the list

name_list.index("kumar")
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

print(len(tokenizer))

tokenizer.tokenize("my name is abhishek kumar.")
# Let's see how to increase the vocabulary of Bert model and tokenizer

num_added_toks = tokenizer.add_tokens(name_list)

# some names are already present in the dictionary

print('We have added', num_added_toks, 'tokens')

print(len(tokenizer))

tokenizer.tokenize("my name is abhishek kumar.")


df1 = pd.DataFrame({ "Sentence #":['Sentence: 47960']*6, 

                    "Word":['my', 'name', 'is', 'abhishek', 'kumar','.'],  

                    "POS":[None]*6,

                    "Tag":['O','O','O','B-per','I-per','O']})

df2 = pd.DataFrame({ "Sentence #":['Sentence: 47961']*7, 

                    "Word":['my', 'name', 'is', 'ritik', 'kumar','gupta','.'],  

                    "POS":[None]*7,

                    "Tag":['O','O','O','B-per','I-per','I-per','O']})

df3 = pd.DataFrame({ "Sentence #":['Sentence: 47962']*6, 

                    "Word":['I', 'am', 'pranav', 'singh', 'murari','.'],  

                    "POS":[None]*6,

                    "Tag":['O','O','B-per','I-per','I-per','O']})
# adding custom data in ner_dataset 

data=data.append([df1,df2,df3]).reset_index(drop=True)

data = data.fillna(method="ffill")
data = data.fillna(method="ffill")
data.tail()
class SentenceGetter(object):

    

    def __init__(self, data):

        self.n_sent = 1

        self.data = data

        self.empty = False

        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),

                                                           s["POS"].values.tolist(),

                                                           s["Tag"].values.tolist())]

        self.grouped = self.data.groupby("Sentence #").apply(agg_func)

        self.sentences = [s for s in self.grouped]

    

    def get_next(self):

        try:

            s = self.grouped["Sentence: {}".format(self.n_sent)]

            self.n_sent += 1

            return s

        except:

            return None
getter = SentenceGetter(data)
sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]

sentences[0]
labels = [[s[2] for s in sent] for sent in getter.sentences]

print(labels[0])
tags_vals = list(set(data["Tag"].values))

tag2idx = {t: i for i, t in enumerate(tags_vals)}
import torch

from torch.optim import Adam

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from pytorch_pretrained_bert import BertTokenizer, BertConfig

from pytorch_pretrained_bert import BertForTokenClassification
MAX_LEN = 75

bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0) 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

print(tokenized_texts[0])
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],

                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],

                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",

                     dtype="long", truncating="post")
attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, 

                                                            random_state=2018, test_size=0.1)

tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,

                                             random_state=2018, test_size=0.1)
tr_inputs = torch.tensor(tr_inputs)

val_inputs = torch.tensor(val_inputs)

tr_tags = torch.tensor(tr_tags)

val_tags = torch.tensor(val_tags)

tr_masks = torch.tensor(tr_masks)

val_masks = torch.tensor(val_masks)
train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)



valid_data = TensorDataset(val_inputs, val_masks, val_tags)

valid_sampler = SequentialSampler(valid_data)

valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))

model.cuda()



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

# model.resize_token_embeddings(len(tokenizer))  
from seqeval.metrics import f1_score

from tqdm import trange



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

    for step, batch in enumerate(train_dataloader):

        # add batch to gpu

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        # forward pass

        loss = model(b_input_ids, token_type_ids=None,

                     attention_mask=b_input_mask, labels=b_labels)

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

    for batch in valid_dataloader:

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        

        with torch.no_grad():

            tmp_eval_loss = model(b_input_ids, token_type_ids=None,

                                  attention_mask=b_input_mask, labels=b_labels)

            logits = model(b_input_ids, token_type_ids=None,

                           attention_mask=b_input_mask)

        logits = logits.detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()

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

    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]

    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]

    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
model.eval()

predictions = []

true_labels = []

eval_loss, eval_accuracy = 0, 0

nb_eval_steps, nb_eval_examples = 0, 0

for batch in valid_dataloader:

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



pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]

valid_tags = [[tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l ]

print("Validation loss: {}".format(eval_loss/nb_eval_steps))

print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
# test

sent="my name is ajay parker singh."

a=[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))]

input_ids=pad_sequences(a,maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

attention_masks=[[float(i>0) for i in ii] for ii in input_ids]

tr_inputs = torch.tensor(input_ids)

tr_masks = torch.tensor(attention_masks)

valid_data = TensorDataset(tr_inputs, tr_masks)

valid_sampler = SequentialSampler(valid_data)

valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)
predictions = []

for batch in valid_dataloader:

    batch = tuple(t.to(device) for t in batch)

    b_input_ids, b_input_mask = batch

    with torch.no_grad():

#         tmp_eval_loss = model(b_input_ids, token_type_ids=None,

#                               attention_mask=b_input_mask, labels=b_labels)

        logits = model(b_input_ids, token_type_ids=None,

                       attention_mask=b_input_mask)

        

    logits = logits.detach().cpu().numpy()

    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

tags=[[p_i for p_i in p] for p in predictions]
# Custom Tokenizer

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r' \1 ', s).split()
print("{:15}||{}".format("Word", "Prediction"))

print(30 * "=")

for w, pred in zip(tokenize(sent), tags[0]):

    print("{:15}: {:5}".format(w, tags_vals[pred]))