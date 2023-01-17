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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

spam_df = pd.read_csv('/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')
# Any results you write to the current directory are saved as output.
spam_df.head()
mails = spam_df.values[:,1]
from sklearn.feature_extraction.text import CountVectorizer

# create an instance
count_vect = CountVectorizer()
# fit the vectorizer with data
X = count_vect.fit_transform(mails)
X = X.todense()
X = np.array(X).astype(np.float)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y = le.fit_transform(spam_df.Category)
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt

X_train,y_train = torch.FloatTensor(X_train),torch.LongTensor(y_train)
X_test,y_test = torch.FloatTensor(X_test),torch.LongTensor(y_test)

train_set = TensorDataset(X_train,y_train)
train_loader = DataLoader(dataset=train_set, batch_size=100, shuffle=True)

test_set = TensorDataset(X_test,y_test)
test_loader = DataLoader(dataset=test_set, batch_size=100, shuffle=True)
input_size = X.shape[1]
output_size = y.max().item()+1

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(input_size,256),
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
            nn.Linear(128,output_size)
        )

    def forward(self, x):
        return self.nn(x)
model = NN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()
num_epochs = 1


def test_evaluate():
    Loss = 0.
    for step, (x, y) in enumerate(test_loader):
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        Loss += loss.item()
    Loss /= (step+1)
    print("loss on test set: %.4f"%(Loss)) 
    
for epoch in range(num_epochs):
    Loss = 0.
    for step, (x, y) in enumerate(train_loader):
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        Loss += loss.item()
        optimizer.step()
    Loss /= (step+1)
    if (epoch+1)%1==0:
        test_evaluate()
        print("epoch %d, training loss %.4f"%(epoch+1,Loss))
right = 0
total = 0
for step, (x, y_true) in enumerate(test_loader):
    out = model(x)
    y_pred = torch.argmax(out,dim = 1)
    right += (y_pred==y_true).sum()
    total += len(y_true)

acc = right.item()/total
print("accuracy on test set: %.2f%%"%(acc*100))
