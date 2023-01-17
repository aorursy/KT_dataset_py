import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pylab import rcParams

from sklearn.preprocessing import LabelEncoder, StandardScaler



%matplotlib inline



sns.set(style='whitegrid', palette='muted', font_scale=1.5)



rcParams['figure.figsize'] = 14, 8



RANDOM_SEED = 42

LABELS = ["Normal", "Fraud"]
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.shape
pd.value_counts(df['Class'], sort = True)
frauds = df[df.Class == 1]

normal = df[df.Class == 0]
frauds.Amount.describe()
normal.Amount.describe()
data = df.drop(['Time'], axis=1)
data.shape
from sklearn.preprocessing import StandardScaler



data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data.shape
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
X_train = X_train[X_train.Class == 0]

X_train = X_train.drop(['Class'], axis=1)



y_test = X_test['Class']

X_test = X_test.drop(['Class'], axis=1)



X_train = X_train.values

X_test = X_test.values
X_train.shape
X_train.shape[1]
input_dim = X_train.shape[1]

encoding_dim = 14

batch_size = 30
import torch

import torchvision as tv

import torchvision.transforms as transforms

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable
input_neurons = input_dim

hidden_neurons = encoding_dim

# create our network

class AutoEncoder(torch.nn.Module):

    def __init__(self):

        super(AutoEncoder, self).__init__()

        self.input = nn.Linear(input_neurons,encoding_dim)

        self.tanh=nn.Tanh()

        self.relu=nn.ReLU()

        self.fc1=nn.Linear(encoding_dim,int(encoding_dim / 2))

        self.fc2=nn.Linear(int(encoding_dim / 2),int(encoding_dim / 2))

        self.fc3=nn.Linear(int(encoding_dim / 2),encoding_dim)

        self.fc4=nn.Linear(encoding_dim,input_neurons)

    def forward(self, x):

        x = self.input(x)

        x = self.tanh(self.fc1(x))

        x = self.relu(self.fc2(x))

        x = self.tanh(self.fc3(x))

        x = self.relu(self.fc4(x))

        return x



encoder = AutoEncoder()

optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

loss_fn = nn.MSELoss() 
num_epochs = 300 #you can go for more epochs, I am using a mac
encoder.train()

tensor_data = torch.from_numpy(X_train).to(torch.float32)

for epoch in range(num_epochs):

    pred_data = encoder(tensor_data)

    optimizer.zero_grad()

    loss = loss_fn(pred_data, tensor_data)

    loss.backward()

    optimizer.step()

    print("Epoch %s \t%s"%(epoch+1, loss.item()))
predictions = encoder(torch.from_numpy(X_test).to(torch.float32))
pred=predictions.detach().numpy()
import numpy as np

mse = np.mean(np.power(X_test - pred, 2), axis=1)

error_df = pd.DataFrame({'reconstruction_error': mse,

                        'true_class': y_test})
error_df.describe()
fig = plt.figure()

ax = fig.add_subplot(111)

normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < 10)]

_ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)
fig = plt.figure()

ax = fig.add_subplot(111)

fraud_error_df = error_df[error_df['true_class'] == 1]

_ = ax.hist(fraud_error_df.reconstruction_error.values, bins=10)
threshold = 3
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,

                             roc_curve, recall_score, classification_report, f1_score,

                             precision_recall_fscore_support)
groups = error_df.groupby('true_class')

fig, ax = plt.subplots()



for name, group in groups:

    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',

            label= "Fraud" if name == 1 else "Normal")

ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')

ax.legend()

plt.title("Reconstruction error for different classes")

plt.ylabel("Reconstruction error")

plt.xlabel("Data point index")

plt.show();
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]

conf_matrix = confusion_matrix(error_df.true_class, y_pred)



plt.figure(figsize=(12, 12))

sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

plt.title("Confusion matrix")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()