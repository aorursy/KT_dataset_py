!curl -q https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev

!pip -q install seqeval
# Importing pytorch and the library for TPU execution



import torch

import torch_xla

import torch_xla.core.xla_model as xm
# Importing stock ml libraries



import numpy as np

import pandas as pd

import transformers

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertForTokenClassification, BertTokenizer, BertConfig, BertModel



# Preparing for TPU usage

dev = xm.xla_device()
df = pd.read_csv("../input/entity-annotated-corpus/ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)

dataset=df.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',

       'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',

       'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',

       'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',

       'prev-prev-word', 'prev-shape', 'prev-word','shape'],axis=1)

dataset.head()
# Creating a class to pull the words from the columns and create them into sentences



class SentenceGetter(object):

    

    def __init__(self, dataset):

        self.n_sent = 1

        self.dataset = dataset

        self.empty = False

        agg_func = lambda s: [(w,p, t) for w,p, t in zip(s["word"].values.tolist(),

                                                       s['pos'].values.tolist(),

                                                        s["tag"].values.tolist())]

        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)

        self.sentences = [s for s in self.grouped]

    

    def get_next(self):

        try:

            s = self.grouped["Sentence: {}".format(self.n_sent)]

            self.n_sent += 1

            return s

        except:

            return None



getter = SentenceGetter(dataset)
# Creating new lists and dicts that will be used at a later stage for reference and processing



tags_vals = list(set(dataset["tag"].values))

tag2idx = {t: i for i, t in enumerate(tags_vals)}

sentences = [' '.join([s[0] for s in sent]) for sent in getter.sentences]

labels = [[s[2] for s in sent] for sent in getter.sentences]

labels = [[tag2idx.get(l) for l in lab] for lab in labels]
# Defining some key variables that will be used later on in the training



MAX_LEN = 200

TRAIN_BATCH_SIZE = 32

VALID_BATCH_SIZE = 16

EPOCHS = 5

LEARNING_RATE = 2e-05

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
class CustomDataset(Dataset):

    def __init__(self, tokenizer, sentences, labels, max_len):

        self.len = len(sentences)

        self.sentences = sentences

        self.labels = labels

        self.tokenizer = tokenizer

        self.max_len = max_len

        

    def __getitem__(self, index):

        sentence = str(self.sentences[index])

        inputs = self.tokenizer.encode_plus(

            sentence,

            None,

            add_special_tokens=True,

            max_length=self.max_len,

            pad_to_max_length=True,

            return_token_type_ids=True

        )

        ids = inputs['input_ids']

        mask = inputs['attention_mask']

        label = self.labels[index]

        label.extend([4]*200)

        label=label[:200]



        return {

            'ids': torch.tensor(ids, dtype=torch.long),

            'mask': torch.tensor(mask, dtype=torch.long),

            'tags': torch.tensor(label, dtype=torch.long)

        } 

    

    def __len__(self):

        return self.len
# Creating the dataset and dataloader for the neural network



train_percent = 0.8

train_size = int(train_percent*len(sentences))

# train_dataset=df.sample(frac=train_size,random_state=200).reset_index(drop=True)

# test_dataset=df.drop(train_dataset.index).reset_index(drop=True)

train_sentences = sentences[0:train_size]

train_labels = labels[0:train_size]



test_sentences = sentences[train_size:]

test_labels = labels[train_size:]



print("FULL Dataset: {}".format(len(sentences)))

print("TRAIN Dataset: {}".format(len(train_sentences)))

print("TEST Dataset: {}".format(len(test_sentences)))



training_set = CustomDataset(tokenizer, train_sentences, train_labels, MAX_LEN)

testing_set = CustomDataset(tokenizer, test_sentences, test_labels, MAX_LEN)
train_params = {'batch_size': TRAIN_BATCH_SIZE,

                'shuffle': True,

                'num_workers': 0

                }



test_params = {'batch_size': VALID_BATCH_SIZE,

                'shuffle': True,

                'num_workers': 0

                }



training_loader = DataLoader(training_set, **train_params)

testing_loader = DataLoader(testing_set, **test_params)
# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 



class BERTClass(torch.nn.Module):

    def __init__(self):

        super(BERTClass, self).__init__()

        self.l1 = transformers.BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=18)

        # self.l2 = torch.nn.Dropout(0.3)

        # self.l3 = torch.nn.Linear(768, 200)

    

    def forward(self, ids, mask, labels):

        output_1= self.l1(ids, mask, labels = labels)

        # output_2 = self.l2(output_1[0])

        # output = self.l3(output_2)

        return output_1
model = BERTClass()

model.to(dev)
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
def train(epoch):

    model.train()

    for _,data in enumerate(training_loader, 0):

        ids = data['ids'].to(dev, dtype = torch.long)

        mask = data['mask'].to(dev, dtype = torch.long)

        targets = data['tags'].to(dev, dtype = torch.long)



        loss = model(ids, mask, labels = targets)[0]



        # optimizer.zero_grad()

        if _%500==0:

            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        

        optimizer.zero_grad()

        loss.backward()

        xm.optimizer_step(optimizer)

        xm.mark_step() 
for epoch in range(5):

    train(epoch)
from seqeval.metrics import f1_score



def flat_accuracy(preds, labels):

    flat_preds = np.argmax(preds, axis=2).flatten()

    flat_labels = labels.flatten()

    return np.sum(flat_preds == flat_labels)/len(flat_labels)
def valid(model, testing_loader):

    model.eval()

    eval_loss = 0; eval_accuracy = 0

    n_correct = 0; n_wrong = 0; total = 0

    predictions , true_labels = [], []

    nb_eval_steps, nb_eval_examples = 0, 0

    with torch.no_grad():

        for _, data in enumerate(testing_loader, 0):

            ids = data['ids'].to(dev, dtype = torch.long)

            mask = data['mask'].to(dev, dtype = torch.long)

            targets = data['tags'].to(dev, dtype = torch.long)



            output = model(ids, mask, labels=targets)

            loss, logits = output[:2]

            logits = logits.detach().cpu().numpy()

            label_ids = targets.to('cpu').numpy()

            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

            true_labels.append(label_ids)

            accuracy = flat_accuracy(logits, label_ids)

            eval_loss += loss.mean().item()

            eval_accuracy += accuracy

            nb_eval_examples += ids.size(0)

            nb_eval_steps += 1

        eval_loss = eval_loss/nb_eval_steps

        print("Validation loss: {}".format(eval_loss))

        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

        pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]

        valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]

        print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
valid(model, testing_loader)