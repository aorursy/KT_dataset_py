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
os.mkdir("/kaggle/working/models")
from __future__ import absolute_import, division, print_function

import torch

from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torch.nn import BCELoss, Sigmoid

import pandas as pd

from tqdm import tqdm

import os

from pytorch_transformers import BertForSequenceClassification, BertConfig

train_df = pd.read_csv('/kaggle/input/hcru2qes24j78at/train.csv')

test_df = pd.read_csv('/kaggle/input/hcru2qes24j78at/test.csv')



PAD = 0

CLS = 1

SEP = 2

emb_len = 40



class SC_Dataset(Dataset):

    #Sentence Comparison Dataset

    def __init__(self, df, emb_len, is_train = True):

        self.df = df

        self.is_train = is_train

        self.emb_len = emb_len



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        sent1 = list(map(lambda x: int(x) + 3, self.df.iloc[idx, 1].split()))

        sent2 = list(map(lambda x: int(x) + 3, self.df.iloc[idx, 2].split()))

        text = torch.tensor([CLS] + sent1 + [SEP] + sent2 + [SEP] + [0] * (self.emb_len - len(sent1) - len(sent2) - 3) )

        if self.is_train:

            label = self.df.iloc[idx, 3]

            return text, label

        return text



train_dset = SC_Dataset(train_df, emb_len)

train_loader = DataLoader(train_dset,  batch_size=32, shuffle=True, num_workers=2)
itr = 1

p_itr =200

lr = 1e-4

epochs = 60

total_loss = 0

PATH = '/kaggle/working/models/bert_ckpt.pt'



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

config = BertConfig(num_hidden_layers=4, num_attention_heads=8, max_position_embeddings=emb_len, num_labels=1)

model = BertForSequenceClassification(config).to(device)





optimizer = Adam(model.parameters(), lr=1e-6)

criterion = BCELoss()

sigmoid = Sigmoid()
model.train()

model.load_state_dict(torch.load(PATH))

for epoch in range(epochs):

    for text, labels  in train_loader:

        optimizer.zero_grad()

        text, labels = text.to(device), labels.type(torch.float).to(device)

        logit = model(text)[0]

        loss = criterion(sigmoid(logit.view(-1)), labels.view(-1))

        total_loss += loss.item()

        loss.backward()

        optimizer.step()

        

        if itr % p_itr == 0:

            torch.save(model.state_dict(), PATH)

            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}'.format(epoch+1, epochs, itr, total_loss/p_itr))

            total_loss = 0

            

        itr+=1
test_dset = SC_Dataset(test_df, emb_len, is_train=False)

test_loader = DataLoader(test_dset, batch_size=32, num_workers=2 )


# model.load_state_dict(torch.load(PATH))

model.eval()

test_preds = []



with torch.no_grad():

    for i, text in tqdm(enumerate(test_loader)):

        text = text.to(device)

        preds = model(text)[0].view(-1).detach()

        test_preds.append(np.sign(preds.cpu().numpy())/2 + 0.5)
result = np.concatenate(test_preds).astype(np.int64)
submission = pd.read_csv('/kaggle/input/hcru2qes24j78at/sample_submission.csv')

filename = 'submit_test.csv'



submission['label'] = result

submission.to_csv(filename, index=False)