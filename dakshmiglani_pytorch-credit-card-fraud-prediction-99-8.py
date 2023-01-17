import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
df = pd.read_csv('../input/creditcard.csv')
df.head(1) # give us a sneek preview of the dataset xD
X = df.iloc[:, :-1].values # extracting features
y = df.iloc[:, -1].values # extracting labels
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=1)
class FraudNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, 18)
        self.fc3 = nn.Linear(18, 20)
        self.fc4 = nn.Linear(20, 24)
        self.fc5 = nn.Linear(24, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.25)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x
net = FraudNet().double()
net
X_train = torch.from_numpy(X_train)
Y_train = torch.from_numpy(Y_train).double()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
training_epochs = 2
minibatch_size = 64
train = data_utils.TensorDataset(X_train, Y_train)
train_loader = data_utils.DataLoader(train, batch_size=minibatch_size, shuffle=True)
for i in range(training_epochs):
    for b, data in enumerate(train_loader, 0):
        inputs, labels = data
        y_pred = net(inputs)
        loss = loss_fn(y_pred, labels)
        
        if b % 100:
            print('Epochs: {}, batch: {} loss: {}'.format(i, b, loss))
        #reset gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
X_test = torch.from_numpy(X_test)
Y_test = torch.from_numpy(Y_test).double()
test = data_utils.TensorDataset(X_test, Y_test)
test_loader = data_utils.DataLoader(test, batch_size=minibatch_size, shuffle=True)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.double() == labels).sum().item()

print('Accuracy of the network on the {} inputs: {}'.format(
    X_test.shape[0], 100 * correct/total))