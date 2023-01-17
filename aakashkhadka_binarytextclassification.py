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
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torchtext
from torchtext.data import Field,BucketIterator,TabularDataset
import torch.nn as nn
import sklearn.metrics as metrics
metrics.accuracy_score([1,1],[1,2])
df=pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')
df.head()
df=df.iloc[:,0:2]
df.head()
df=df.rename(columns={'v1':'labels','v2':'text'})
df.head()
train,test=train_test_split(df,test_size=0.2,random_state=42)
train=train.reset_index(drop=True)
test=test.reset_index(drop=True)
train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)
!ls
import nltk
nltk.download('punkt')
from nltk import word_tokenize
text=torchtext.data.Field(tokenize=word_tokenize)
label=torchtext.data.LabelField(dtype=torch.float)
datafields=[('labels',label),('text',text)]
trn,tst=torchtext.data.TabularDataset.splits(path="",
                                            train='train.csv',
                                            test='test.csv',
                                            format='csv',
                                            skip_header=True,
                                            fields=datafields)
len(trn),len(tst)
trn[5].__dict__.keys()
trn[5].text
trn[5].labels
text.build_vocab(trn,max_size=10500)
label.build_vocab(trn)
len(text.vocab),len(label.vocab)
print(text.vocab.freqs.most_common(50))
batch_size=64
train_iterator,test_iterator=torchtext.data.BucketIterator.splits(
    (trn,tst),
    batch_size=batch_size,
    sort_key=lambda x :len(x.text),
    sort_within_batch=False
)
class RNN(nn.Module):
    
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
  
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, text):
        
        embedded = self.embedding(text)
        
        embedded_dropout = self.dropout(embedded)
        
        output, (hidden, _) = self.rnn(embedded_dropout)
        
        hidden_1D = hidden.squeeze(0)
        
        assert torch.equal(output[-1, :, :], hidden_1D)
        
        return self.fc(hidden_1D)
input_dim=len(text.vocab)
input_dim = len(text.vocab)

embedding_dim = 100

hidden_dim = 256

output_dim = 1
model = RNN(input_dim, embedding_dim, hidden_dim, output_dim)
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr = 1e-6)
criterion = nn.BCEWithLogitsLoss()
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
                
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.labels)
        
        rounded_preds = torch.round(torch.sigmoid(predictions))
        correct = (rounded_preds == batch.labels).float() 
        
        acc = correct.sum() / len(correct)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
num_epochs = 5

for epoch in range(num_epochs):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% ')
epoch_loss = 0
epoch_acc = 0
model.eval()
with torch.no_grad():

    for batch in test_iterator:

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.labels)

        rounded_preds = torch.round(torch.sigmoid(predictions))
        
        correct = (rounded_preds == batch.labels).float() 
        acc = correct.sum() / len(correct)

        epoch_loss += loss.item()
        epoch_acc += acc.item()

test_loss = epoch_loss / len(test_iterator)
test_acc  = epoch_acc / len(test_iterator)

print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')
