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



# Any results you write to the current directory are saved as output.
import re

from string import punctuation



def preprocess(phrase):

    phrase = re.sub(r'#', '', phrase)

    phrase = re.sub(r'@\w+', '', phrase)

    phrase = re.sub(r'(http|https)://[=a-zA-Z0-9_/?&.-]+', '', phrase)

    phrase = re.sub(r'[^\w\s]', ' ', phrase)

    phrase = phrase.lower()

    return phrase
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
train_data = train['text'].apply(preprocess)
train['text'] = train_data

train.head()
vocab_file = '/kaggle/input/pretrained-bert-models-for-pytorch/bert-base-uncased-vocab.txt'

bert_dir = '/kaggle/input/pretrained-bert-models-for-pytorch/bert-base-uncased/'

bert_config = '/kaggle/input/pretrained-bert-models-for-pytorch/bert-base-uncased/bert_config.json'
from transformers import BertTokenizer, BertModel

import torch

from torch.utils.data import DataLoader, Dataset
def bert_tokenizer(vocab_file):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True)

    return tokenizer



tokenizer = bert_tokenizer(vocab_file)
class TextDataset(Dataset):

    def __init__(self, df, tokenizer, max_len):

        super(TextDataset, self).__init__()

        

        self.bert_encode = tokenizer

        self.text = df.text.values

        self.label = df.target.values

        self.max_len = max_len

    

    def __len__(self):

        return len(self.text)

    

    def __getitem__(self, idx):

        tokens, mask = self.get_token_mask(self.text[idx], self.max_len)

        label = self.label[idx]

        if torch.cuda.is_available():

          return [torch.tensor(tokens).cuda(), torch.tensor(mask).cuda()], torch.tensor(label).unsqueeze(0).cuda()

        else:

          return [torch.tensor(tokens), torch.tensor(mask)], torch.tensor(label).unsqueeze(0)



    def get_token_mask(self, text, max_len):

        tokens = []

        mask = []

        #text = self.bert_encode.encode(text, add_special_tokens=True, max_length=20, pad_to_max_length=True, padding_side='right')

        text = self.bert_encode.encode(text, add_special_tokens=True, max_length=20)

        size = len(text)

        if size < max_len:

            pads = self.bert_encode.encode(['PAD'] * (max_len - size), add_special_tokens=False)

            tokens = text + pads

            mask = [1] * size + [0] * (max_len - size)

        else:

            tokens = text

            mask = [1] * max_len

        return tokens, mask
#from sklearn.model_selection import train_test_split

#X_train, X_valid = train_test_split(train, test_size=0.1, random_state=143)



#train_dataset = TextDataset(X_train, tokenizer=tokenizer, max_len=20)

train_dataset = TextDataset(train, tokenizer=tokenizer, max_len=20)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)



#valid_dataset = TextDataset(X_valid, tokenizer=tokenizer, max_len=20)

#valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=True)
import torch.nn as nn

import torch.functional as F

import torch.optim as optim
class LSTMnet(nn.Module):

    def __init__(self, output_size, path_config = bert_config ,path_bert = bert_dir, n_layers=1):

        super(LSTMnet, self).__init__()

        self.n_layers = n_layers

        

        self.bert = BertModel.from_pretrained(path_bert, config=path_config)

        self.hidden_size = self.bert.config.hidden_size

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, num_layers=n_layers, bidirectional=True)

        self.fc1 = nn.Linear(2 * self.hidden_size, self.hidden_size) #2 as its bidirectional

        self.fc2 = nn.Linear(self.hidden_size, output_size)

        self.sigmoid = nn.Sigmoid()

        

    def forward(self, word):

        batch_size = word[0].size(0)

        hidden = self.init_hidden(batch_size)

        

        embedded, pooled_out = self.bert(input_ids=word[0], attention_mask=word[1]) #emb -> torch.Size([10, 20, 768]) batch, seq, hidden

        output, hidden = self.lstm(embedded, hidden)

        #output = self.fc1(output[:,-1,:]) #when bidirectional=False in RNN

        

        out_fwd = output[:,-1,:(self.hidden_size)]

        out_bwd = output[:,0,(self.hidden_size):]

        output = torch.cat((out_fwd, out_bwd), 1)

        output = self.fc1(output)

        output = self.fc2(output)

        output = self.sigmoid(output)

        return output, hidden

        

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data

        hidden = (weight.new(2*self.n_layers, batch_size, self.hidden_size).zero_(),

                      weight.new(2*self.n_layers, batch_size, self.hidden_size).zero_())

        return hidden
#Model parameters

output_size = 1

layers = 2
def training(model, train_loader):#, valid_loader):

    loss_function = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.00001)



    maxEpoch = 2

    print_every = 500

    counter = 0



    model.train()

    for epochs in range(maxEpoch):

        for inputs, labels in train_loader:

            counter += 1

            model.zero_grad()



            output, hidden = model(inputs)

            loss = loss_function(output, labels.float())

            loss.backward()

            optimizer.step()



            if counter % print_every == 0:

                # Get validation loss

#                val_losses = []

#                model.eval()



#                for inputs, labels in valid_loader:



#                    output, val_h = model(inputs)

#                    val_loss = loss_function(output, labels.float())



#                    val_losses.append(val_loss.item())



#                model.train()

                print("Epoch: {}/{}...".format(epochs+1, maxEpoch),

                      "Step: {}...".format(counter),

                      "Loss: {:.6f}...".format(loss.item()))#,

#                      "Val Loss: {:.6f}".format(np.mean(val_losses)))

    return model
model = LSTMnet(output_size, n_layers=layers)

if torch.cuda.is_available():

  model = model.cuda()
model = training(model, train_loader)#, valid_loader)
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
test['text'] = test['text'].apply(preprocess)

test.head()
class TestDataset(Dataset):

    def __init__(self, df, tokenizer, max_len):

        super(TestDataset, self).__init__()

        

        self.bert_encode = tokenizer

        self.text = df.text.values

        # self.label = df.target.values

        self.max_len = max_len

    

    def __len__(self):

        return len(self.text)

    

    def __getitem__(self, idx):

        tokens, mask = self.get_token_mask(self.text[idx], self.max_len)

        #label = self.label[idx]

        if torch.cuda.is_available():

          return [torch.tensor(tokens).cuda(), torch.tensor(mask).cuda()]#, torch.tensor(label).unsqueeze(0).cuda()

        else:

          return [torch.tensor(tokens), torch.tensor(mask)]#, torch.tensor(label).unsqueeze(0)



    def get_token_mask(self, text, max_len):

        tokens = []

        mask = []

        #text = self.bert_encode.encode(text, add_special_tokens=True, max_length=20, pad_to_max_length=True, padding_side='right')

        text = self.bert_encode.encode(text, add_special_tokens=True, max_length=20)

        size = len(text)

        if size < max_len:

            pads = self.bert_encode.encode(['PAD'] * (max_len - size), add_special_tokens=False)

            tokens = text + pads

            mask = [1] * size + [0] * (max_len - size)

        else:

            tokens = text

            mask = [1] * max_len

        return tokens, mask
test_dataset = TestDataset(test, tokenizer=tokenizer, max_len=20)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
if torch.cuda.is_available():

  model = model.cpu()
with torch.no_grad():

    prediction = torch.zeros((len(test),1))

    for idx, inputs in enumerate(test_loader):

        inputs = [w.cpu() for w in inputs]

        output, hidden = model(inputs)

        pred = torch.round(output.squeeze())

        prediction[idx] = pred
try_pred = pd.DataFrame(prediction.detach().numpy(), dtype=int)
#prediction = pd.Series(prediction.detach().numpy(),dtype=int )



submission = pd.DataFrame({'id':test['id'],

                          'target':try_pred.iloc[:,0].values})
os.chdir("/kaggle/working")

submission = submission.to_csv("submission.csv", index = False)