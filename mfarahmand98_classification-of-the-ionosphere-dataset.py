import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim
df = pd.read_csv('/kaggle/input/ionosphere/ionosphere_data.csv')

df
sns.countplot(x='column_ai', data=df)
df.drop(columns=['column_b'], inplace=True)
df.rename(columns={'column_ai': 'label'}, inplace=True)

df['label'] = df.label.astype('category')

encoding = {'g': 1, 'b': 0}

df.label.replace(encoding, inplace=True)

df
df['column_a'] = df.column_a.astype('float64')
X = df.values[:, :-1]

y = df.values[:, -1]
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

# The boolean feature does not need to be normalized.

x_train[:, 1:] = scaler.fit_transform(x_train[:, 1:])

x_test[:, 1:] = scaler.transform(x_test[:, 1:])
from sklearn.manifold import TSNE



x_embedded = TSNE(n_components=2).fit_transform(x_train)

plt.scatter(x_embedded[:, 0], x_embedded[:, 1], color=['green' if label else 'red' for label in y_train])

plt.show()
from sklearn.decomposition import PCA



x_embedded = PCA(n_components=2).fit_transform(x_train)

plt.scatter(x_embedded[:, 0], x_embedded[:, 1], color=['green' if label else 'red' for label in y_train])

plt.show()
iterations = 100

batch_size = 32
from torch.utils.data import Dataset





class TrainData(Dataset):

    

    def __init__(self, x_train, y_train):

        self.x_train = x_train

        self.y_train = y_train

    

    def __getitem__(self, index):

        return self.x_train[index], self.y_train[index]

    

    def __len__ (self):

        return len(self.x_train)



    

class TestData(Dataset):

    

    def __init__(self, x_test):

        self.x_test = x_test

        

    def __getitem__(self, index):

        return self.x_test[index]

        

    def __len__ (self):

        return len(self.x_test)
train_data = TrainData(torch.from_numpy(x_train).to(torch.float32), torch.from_numpy(y_train).to(torch.float32))

test_data = TestData(torch.from_numpy(x_test).to(torch.float32))
from torch.utils.data import DataLoader



train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_data, batch_size=1)
class Network(nn.Module):

    

    def __init__(self):

        super().__init__()

        # Number of input features is 33.

        self.linear_1 = nn.Linear(33, 64)

        self.linear_2 = nn.Linear(64, 64)

        self.linear_3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(p=0.1)

    

    def forward(self, inputs):

        out = self.linear_1(inputs)

        out = F.relu(out)

        out = self.linear_2(out)

        out = F.relu(out)

        out = self.dropout(out)

        out = self.linear_3(out)

        # Sigmoid activation is later applied by the loss function for numerical stability.

        return out
network = Network()

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(network.parameters(), lr=1e-4)
from sklearn.metrics import roc_auc_score



n_batches = len(train_loader)



network.train()



loss_li = []

score_li = []



for it in range(iterations):

    it_loss = 0

    it_score = 0

    for x_batch, y_batch in train_loader:

        optimizer.zero_grad()

        y_real = y_batch.unsqueeze(1)

        y_pred = network(x_batch)

        loss = criterion(y_pred, y_real)

        y_pred = torch.sigmoid(y_pred.detach())

        score = roc_auc_score(y_real, y_pred)

        loss.backward()

        optimizer.step()

        it_loss += loss.item()

        it_score += score

    loss_li.append(it_loss / n_batches)

    score_li.append(it_score / n_batches)

    print('[Iteration {}] Loss: {:.4f}, Area-Under-Curve: {:.4f}'.format(it, it_loss / n_batches, it_score / n_batches))

        
plt.plot(loss_li)

plt.xlabel('Iteration')

plt.ylabel('Binary Cross-Entropy Loss')

plt.show()
plt.plot(score_li)

plt.xlabel('Iteration')

plt.ylabel('Area Under Curve')

plt.show()
network.eval()



predictions = []



with torch.no_grad():

    for x_batch in test_loader:

        y_pred = network(x_batch)

        y_pred = torch.sigmoid(y_pred)

        predictions.append(y_pred.squeeze().tolist())



y_pred = np.round(predictions)
from sklearn.metrics import confusion_matrix



confusion_matrix(y_test, y_pred)
from sklearn.metrics import classification_report



classification_report(y_test, y_pred, output_dict=True)