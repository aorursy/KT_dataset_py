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
! pip install jsonlines
import json
import jsonlines
import random

df = pd.DataFrame(json.load(open('../input/sponsor-block-subtitles-80k/full-sponsor-block.json','r')))
dicts = df.to_dict(orient='records')

dictsProcessed = []
    
for d in dicts:
    if d['quote'] != []:
        dictsProcessed.append({'quote':d['quote'],'label':d['label']})

split1 = round(len(dictsProcessed) * 0.7)
split2 = round(len(dictsProcessed) * 0.15)

random.shuffle(dictsProcessed)

with jsonlines.open('train_data.json', mode='w') as writer:
    writer.write_all(dictsProcessed[:split1])
with jsonlines.open('validation_data.json', mode='w') as writer:
    writer.write_all(dictsProcessed[split1 + 1: split1 + 1 + split2])
with jsonlines.open('test_data.json', mode='w') as writer:
    writer.write_all(dictsProcessed[-split2:])
import torch
import random
import numpy as np
from transformers import BertTokenizer

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id
max_input_length = tokenizer.max_model_input_sizes['bert-base-cased']

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens
from torchtext import data

TEXT = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

LABEL = data.LabelField(dtype = torch.float)
fields = {'quote': ('text', TEXT), 'label': ('label', LABEL)}
train_data, validation_data, test_data = data.TabularDataset.splits(
                            path = './',
                            train = 'train_data.json',
                            validation= 'validation_data.json',
                            test = 'test_data.json',
                            format = 'json',
                            fields = fields
)
LABEL.build_vocab(train_data)
BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, validation_data, test_data), 
    batch_size = BATCH_SIZE, 
    sort_key = lambda x: x.text,
    sort_within_batch= False,
    device = device)
from transformers import BertTokenizer, BertModel
import torch.nn as nn

bert = BertModel.from_pretrained('bert-base-cased')

class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = self.bert(text)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output


HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
for name, param in model.named_parameters():                
    if name.startswith('bert'):
        param.requires_grad = False
print(f'The model has {count_parameters(model):,} trainable parameters')
import torch.optim as optim
import time
from sklearn import metrics
import pandas as pd

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc  

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    accumulated_preds = []
    accumulated_test = []
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            accumulated_preds.append(predictions)
            accumulated_test.append(batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), accumulated_preds, accumulated_test

def test_results(accumulated_preds,accumulated_test):
    return torch.round(torch.sigmoid(torch.cat((accumulated_preds), 0))).cpu().numpy(),torch.cat((accumulated_test), 0).cpu().numpy()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

import os

N_EPOCHS = 100

best_valid_loss = float('inf')
model_path = 'full-model.pt'

if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))

i = 0

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc, accumulated_preds, accumulated_test = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
        
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
    if valid_loss < best_valid_loss:
        i = 0
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_path)
    else:
        i +=1
        
    if i == 10:
        print("10 epochs without valid loss improvement. Stopping training.")
        break

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
model.load_state_dict(torch.load('../input/latestmodelsponsorblock/full-model.pt'))

test_loss, test_acc, accumulated_preds, accumulated_test = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

p, t = test_results(accumulated_preds, accumulated_test)
dfR = pd.DataFrame({"preds":p,"test":t})
dfR['diff'] = dfR['preds'] - dfR['test']
dfR['false_positive'] = dfR['diff'].map(lambda x: 1 if x > 0 else 0)
dfR['false_negative'] = dfR['diff'].map(lambda x: 1 if x < 0 else 0)
dfR
cf = metrics.confusion_matrix(torch.round(torch.sigmoid(torch.cat((accumulated_preds), 0))).cpu().numpy(), torch.cat((accumulated_test), 0).cpu().numpy())
cf
dfTest = pd.DataFrame(dictsProcessed[-split2:])
falsePosDf = dfTest[dfTest.index.isin(dfR[dfR['false_positive']==1].index)].reset_index(drop=True)
falseNegDf = dfTest[dfTest.index.isin(dfR[dfR['false_negative']==1].index)].reset_index(drop=True)
def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()
predict_sentiment(model,tokenizer, 'this video was made possible thanks to our sponsor : Adem.sh')
predict_sentiment(model,tokenizer, 'we can clearly see there is a problem in this video')
print(LABEL.vocab.stoi)
falsePosDf.quote.iloc[0]
falseNegDf.quote.iloc[0]
