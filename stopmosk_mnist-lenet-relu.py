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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

X_train.shape, X_val.shape, y_train.shape
sns.distplot(y_train);
X_train = X_train.values.reshape([-1, 28, 28])

y_train = y_train.values



X_val = X_val.values.reshape([-1, 28, 28])

y_val = y_val.values



X_train.shape, X_val.shape
plt.imshow(X_train[666, :, :])

plt.show()



print(y_train[666])
X_train_tensor = torch.tensor(X_train)

y_train_tensor = torch.tensor(y_train)



X_val_tensor = torch.tensor(X_val)

y_val_tensor = torch.tensor(y_val)



X_train_tensor = X_train_tensor.float()

X_val_tensor = X_val_tensor.float()



# Add one dimension (necessary for convolution)

X_train_tensor = X_train_tensor.unsqueeze(1).float()

X_val_tensor = X_val_tensor.unsqueeze(1).float()



X_train_tensor.shape, X_val_tensor.shape
# LeNet



class MNISTNet(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = torch.nn.Conv2d(1, 6, 5, padding=2)   # in_channels, out_channels, kernel_size

        self.a1 = torch.nn.ReLU()

        self.pool1 = torch.nn.MaxPool2d(2)

        

        self.conv2 = torch.nn.Conv2d(6, 16, 5)   # in_channels, out_channels, kernel_size

        self.a2 = torch.nn.ReLU()

        self.pool2 = torch.nn.MaxPool2d(2)



        # batch x Flatten

        

        self.fc1 = torch.nn.Linear(400, 120)

        self.ac1 = torch.nn.ReLU()



        self.fc2 = torch.nn.Linear(120, 84)

        self.ac2 = torch.nn.ReLU()



        self.fc3 = torch.nn.Linear(84, 10)

        # self.ac3 = torch.nn.Softmax()   # Мы используем Cross-Entropy LOSS, а она принимает выходы и без Softmax

    

    def forward(self, x):

        x = self.conv1(x)

        x = self.a1(x)

        x = self.pool1(x)



        x = self.conv2(x)

        x = self.a2(x)

        x = self.pool2(x)



        x = x.view(x.size(0), 400)    # 400 = x.size(1) * x.size(2) * x.size(3)

        

        x = self.fc1(x)

        x = self.ac1(x)

        

        x = self.fc2(x)

        x = self.ac2(x)

        

        x = self.fc3(x)

        #x = self.ac3(x)

                

        return x

    

mnist_net = MNISTNet()

mnist_net = mnist_net.cuda()
loss = torch.nn.CrossEntropyLoss()

#optimizer = torch.optim.Adam(mnist_net.parameters(), lr=0.001)



optimizer = torch.optim.SGD(mnist_net.parameters(), lr=0.01, momentum=0.3)
batch_size = 1000



train_loss_history = []

val_loss_history = []

val_accuracy_history = []



X_val_tensor = X_val_tensor.cuda()

y_val_tensor = y_val_tensor.cuda()



for epoch in range(75):

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

        



    train_loss_history.append(loss_value.data.cpu())   # Save train Loss



    preds = (mnist_net.forward(X_val_tensor))

    val_loss_history.append(loss(preds, y_val_tensor).data.cpu())   # Save test Loss

    

    accuracy = (preds.argmax(dim=1) == y_val_tensor).float().mean()

    val_accuracy_history.append(accuracy)



    if epoch % 10 == 0:

        print(accuracy)
plt.plot(val_accuracy_history);
plt.plot(train_loss_history, label='Train Loss')

%time plt.plot(val_loss_history, label='Test Loss')

plt.legend();
X_test_tensor = torch.tensor(df_test.values.reshape([-1, 28, 28]))

X_test_tensor = X_test_tensor.unsqueeze(1).float()

X_test_tensor = X_test_tensor.cuda()
y_pred = mnist_net.forward(X_test_tensor)  

y_pred = y_pred.argmax(dim=1).cpu().numpy()

y_pred
y_output = df_test.iloc[:, :0]

y_output.index = range(1, 28001)

y_output
y_output['Label'] = y_pred

y_output.index.name = 'ImageId'
y_output
y_output.to_csv('label_submission.csv')