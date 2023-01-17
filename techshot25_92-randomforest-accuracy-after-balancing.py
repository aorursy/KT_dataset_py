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
df = pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')



df.isna().sum()
df.describe()
from matplotlib import pyplot as plt



pulsar = df[df['target_class'] == 1]

nap = df[df['target_class'] == 0]



plt.title('Comparison of classes')

plt.pie((len(pulsar), len(nap)), labels=['Pulsar', 'Not a pulsar'])

plt.show()
resampled_idx = np.random.choice(nap.index, size=len(pulsar))

resampled_nap = df.iloc[resampled_idx]

resampled_df = pd.concat([pulsar, resampled_nap], axis=0)



resampled_df.sample(5)
res_pulsar = resampled_df[resampled_df['target_class'] == 1]

res_nap = resampled_df[resampled_df['target_class'] == 0]





plt.title('Comparison of classes [resampled]')

plt.pie((len(res_pulsar), len(res_nap)), labels=['Pulsar', 'Not a pulsar'])

plt.show()
X = resampled_df[resampled_df.columns[:-1]].values

y = resampled_df[resampled_df.columns[-1]].values



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.1, random_state=42)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_recall_curve



rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)
precision, recall, _ = precision_recall_curve(y_test, rfc_pred)



plt.step(recall, precision, color='r', alpha=1.0,

         where='post')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('2-class Precision-Recall curve')

plt.show()
print(f'Accuracy = {accuracy_score(y_test, rfc_pred) * 100:.5g}%')
from torch import nn, optim

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.feats = nn.Sequential(

            nn.Linear(8, 1028),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(1028, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 10),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(10, 1),

            nn.Sigmoid()

        )



    def forward(self, x):

        return self.feats(x).flatten(1)



model = Model().double().to(device)

opt = optim.Adam(model.parameters())

error = nn.BCELoss()
X_train_torch, X_test_torch, y_train_torch, y_test_torch = map(torch.tensor, [X_train, X_test, y_train, y_test])



def train(model, n_epochs=100):

    model.train()

    for epoch in range(1, n_epochs + 1):

        opt.zero_grad()

        out = model(X_train_torch.to(device))

        loss = error(out, y_train_torch.double().to(device))

        loss.backward()

        opt.step()

        

        if epoch % int(0.1 * n_epochs) == 0:

            print(f'Epoch {epoch} \t Loss: {loss.item():.5g}')



train(model, n_epochs=20000)
with torch.no_grad():

    y_pred = model(X_test_torch.to(device)).cpu().detach().numpy()
nn_pred = [1 if i >= 0.5 else 0 for i in y_pred]

nn_score = accuracy_score(nn_pred, y_test)



print(f'Neural Network accuracy = {nn_score * 100:.5g}%')
precision, recall, _ = precision_recall_curve(y_test, nn_pred)



plt.step(recall, precision, color='r', alpha=1.0,

         where='post')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('2-class Precision-Recall curve')

plt.show()