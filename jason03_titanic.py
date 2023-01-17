import torch

import torch.nn as nn

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
# processing train data

train = pd.read_csv("../input/titanic/train.csv")

train_y = to_categorical(train['Survived'], num_classes = 2)

train_y = torch.from_numpy(np.array(train_y)).type(torch.FloatTensor)



train_x = train.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)

train_x['Sex'] = pd.factorize(train.Sex)[0]

train_x['Embarked'] = pd.factorize(train.Embarked)[0]

train_x = train_x.fillna(train_x.mean())

train_x = torch.from_numpy(np.array(train_x)).type(torch.FloatTensor)

if np.isnan(np.array(train_x)).sum():

    print('nan exists in train_x!')

    exit()

train.describe()
# processing test data

test_x = pd.read_csv("../input/titanic/test.csv")

test_id = test_x['PassengerId']

test_x = test_x.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

test_x['Sex'] = pd.factorize(test_x.Sex)[0]

test_x['Embarked'] = pd.factorize(test_x.Embarked)[0]

test_x = test_x.fillna(test_x.mean())

test_x = torch.from_numpy(np.array(test_x)).type(torch.FloatTensor)

if np.isnan(np.array(test_x)).sum():

    print('nan exists in test_x!')

    exit()
num_epochs = 300

learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
class AnnNet(nn.Module):

    def __init__(self):

        super(AnnNet, self).__init__()

        self.fc1 = nn.Linear(7, 5)

        self.tanh = nn.Tanh()

        self.fc2 = nn.Linear(5, 5)

        self.relu = nn.ReLU()

        self.fc3 = nn.Linear(5, 2)



    def forward(self, x):

        out = self.fc1(x)

        out = self.tanh(out)

        out = self.fc2(out)

        out = self.relu(out)

        out = self.fc3(out)

        return out
best_acc = 0

best_model = None

best_lr = 0

for learning_rate in learning_rates:

    model = AnnNet()

    print(model)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []



    for epoch in range(num_epochs):

        model.train()

        optimizer.zero_grad()

        pred = model(train_x)

        loss = criterion(pred, train_y)

        losses.append(loss)

        loss.backward()

        optimizer.step()

        correct = (torch.max(pred.data, 1)[1] == torch.max(train_y.data, 1)[1]).sum()

        acc = correct.item() / len(pred)

        if epoch % 50 == 0:

            print("Epoch:[{}/{}], acc: {:.4f}, Loss:{:.4f}".format(epoch + 1, num_epochs, acc, loss.item()))

    if acc > best_acc:

        best_acc = acc

        best_model = model

        best_lr = learning_rate

print("best acc: {}, best lr: {}".format(best_acc, best_lr))
best_model.eval()

with torch.no_grad():

    test_y = best_model(test_x)

test_y = pd.DataFrame(torch.argmax(test_y, 1).numpy())

test_y = pd.concat([test_id, test_y], axis=1)

test_y.columns = ['PassengerId', 'Survived']

test_y.to_csv('submission.csv', index=False, encoding='utf8')
# plot loss

plt.figure(figsize=(6, 4), dpi=144)

plt.plot([i + 1 for i in range(num_epochs)], losses, 'r-', lw=1)

plt.yticks([x * 0.1 for x in range(15)])

plt.show()