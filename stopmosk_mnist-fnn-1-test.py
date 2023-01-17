import torch

import random

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split



random.seed(0)

np.random.seed(0)

torch.manual_seed(0)

torch.cuda.manual_seed(0)

torch.backends.cudnn.deterministic = True
df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')



X = df_train.drop(['label'], axis=1)

y = df_train['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

X_train.shape, X_val.shape, y_train.shape
sns.distplot(y_train);
plt.imshow(X_train.values.reshape([-1, 28, 28])[661, :, :])

plt.show()

print(y_train.values[661])
X_train_tensor = torch.tensor(X_train.values)

y_train_tensor = torch.tensor(y_train.values)



X_val_tensor = torch.tensor(X_val.values)

y_val_tensor = torch.tensor(y_val.values)



X_train_tensor = X_train_tensor.float()

X_val_tensor = X_val_tensor.float()
# Check GPU



torch.cuda.is_available()
class MNISTNet(torch.nn.Module):

    def __init__(self, n_hidden_neurons):

        super().__init__()

        self.fc1 = torch.nn.Linear(28 * 28, n_hidden_neurons)

        self.ac1 = torch.nn.Sigmoid()

        self.fc2 = torch.nn.Linear(n_hidden_neurons, 10) 

    

    def forward(self, x):

        x = self.fc1(x)

        x = self.ac1(x)

        x = self.fc2(x)

        return x

    

mnist_net = MNISTNet(100)

mnist_net = mnist_net.cuda()
loss = torch.nn.CrossEntropyLoss()



optimizer = torch.optim.Adam(mnist_net.parameters(), lr=0.001)
batch_size = 500



train_loss_history = []

val_loss_history = []

val_accuracy_history = []



X_val_tensor = X_val_tensor.cuda()

y_val_tensor = y_val_tensor.cuda()



for epoch in range(200):

    order = np.random.permutation(len(X_train))



    for start_index in range(0, len(X_train), batch_size):

        optimizer.zero_grad()

        

        batch_indexes = order[start_index:start_index+batch_size]

        

        X_batch = X_train_tensor[batch_indexes].cuda()

        y_batch = y_train_tensor[batch_indexes].cuda()

        

        preds = mnist_net.forward(X_batch) 

        loss_value = loss(preds, y_batch)

        loss_value.backward()

        optimizer.step()





    preds = mnist_net.forward(X_val_tensor)  

    train_loss_history.append(loss_value)

    val_loss_history.append(loss(preds, y_val_tensor))

    

    accuracy = (preds.argmax(dim=1) == y_val_tensor).float().mean()

    val_accuracy_history.append(accuracy)



    if epoch % 10 == 0:

        print(accuracy)
plt.plot(val_accuracy_history);
plt.plot(train_loss_history, label='Train Loss')

plt.plot(val_loss_history, label='Val Loss')

plt.legend();
X_test_tensor = torch.tensor(df_test.values)

X_test_tensor = X_test_tensor.float()

X_test_tensor = X_test_tensor.cuda()



y_pred = mnist_net.forward(X_test_tensor)

y_pred = y_pred.argmax(dim=1).cpu().numpy()

y_pred
y_output = df_test.iloc[:, :0]

y_output.index = range(1, 28001)



y_output['Label'] = y_pred

y_output.index.name = 'ImageId'
y_output
y_output.to_csv('label_submission.csv')