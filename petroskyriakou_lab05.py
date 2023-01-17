import numpy as np

import pandas as pd

import seaborn as sns

import scipy as cp



from PIL import Image

from matplotlib import pyplot as plt

%matplotlib inline
green = Image.open('../input/chalkida-dataset/green.tif')

nir = Image.open('../input/chalkida-dataset/nir.tif')

gt = Image.open('../input/chalkida-dataset/gt.tif')
green_array = np.array(green)

green_array = green_array.reshape(green_array.shape[0] * green_array.shape[0], 1) #linear



nir_array = np.array(nir)

nir_array = nir_array.reshape(nir_array.shape[0] * nir_array.shape[0], 1)



gt_array = np.array(gt)

gt_array = gt_array.reshape(gt_array.shape[0] * gt_array.shape[0], 1)
data_array = np.concatenate((green_array, nir_array, gt_array), axis =1) #column_wise 



train_df = pd.DataFrame(data = data_array, index = range(green_array.shape[0]), columns = ['green', 'nir', 'gt'])

train_df['gt'] = train_df['gt'].replace(255,1)
sns.set(rc={'figure.figsize' : (6,6)})

sns.countplot(train_df['gt'])
sns.set(rc={'figure.figsize' : (11.7,8)})

sns.countplot(y = 'green', hue = 'gt', data = train_df,

                 order = train_df.green.value_counts().iloc[:10].index)
sns.countplot(y = 'nir', hue = 'gt', data=train_df,

                 order = train_df.green.value_counts().iloc[:10].index)
X = train_df[['green', 'nir']]

y = train_df[['gt']]







print(X.shape, y.shape)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train.values.ravel())

y_pred = knn.predict(X_test)
print(accuracy_score(y_test,y_pred))

print(classification_report(y_test, y_pred))

c1 = confusion_matrix(y_test,y_pred)



print(c1)
ax = sns.heatmap(c1, linewidth = 0.5)



plt.figure(figsize = (10,10))

sns.set(font_scale=1)

sns_plot = sns.heatmap(c1, cmap = 'Blues', annot = True, fmt='g')

ax.invert_yaxis()

ax.tick_params(labeltop=True)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

y_pred_nb = gnb.fit(X_train, y_train.values.ravel()).predict(X_test)
y_pred_nb

print(accuracy_score(y_test,y_pred_nb))

print(classification_report(y_test, y_pred_nb))

c2 = confusion_matrix(y_test,y_pred_nb)

print(c2)
plt.figure(figsize = (10,10))

sns.set(font_scale=1)

sns_plot = sns.heatmap(c2, cmap = 'Blues', annot = True, fmt='g')

ax.invert_yaxis()

ax.tick_params(labeltop=True)
from sklearn.linear_model import Perceptron

perc = Perceptron(random_state=21)

perc.fit(X_train, y_train)

y_pred_perc = perc.predict(X_test)
y_pred_perc

print(accuracy_score(y_test,y_pred_perc))

print(classification_report(y_test, y_pred_perc))

c3 = confusion_matrix(y_test,y_pred_perc)



print(c3)
plt.figure(figsize = (10,10))

sns.set(font_scale=1)

sns_plot = sns.heatmap(c3, cmap = 'Blues', annot = True, fmt='g')

ax.invert_yaxis()

ax.tick_params(labeltop=True)
def perceptron1(X, y):

    w = np.zeros(len(X[0]))

    eta = 1

    n = 10



    for n in range(n):

        for i, x in enumerate(X):

            if (np.dot(X[i], w)*Y[i]) <= 0:

                w = w + eta*X[i]*Y[i]

    return w
A = np.load('../input/indian-pines-hyperspectral-dataset/indianpinearray.npy')

B = np.load('../input/indian-pines-hyperspectral-dataset/IPgt.npy')
print(f"gt shape is {B.shape}")

print(f"tc shape is {A.shape}")

print(f"gt type is {type(B)}")
gt_array = B.reshape(B.shape[0] * B.shape[0], 1)



d_array = A.reshape(A.shape[0] * A.shape[0], 200)



data1 = np.concatenate((d_array, gt_array), axis = 1)
train_df = pd.DataFrame(data = data1, index = range(d_array.shape[0]))

data = train_df



list(data.columns.values)



label=[]

for i in range(200):

  label.append(i)



label.append("label_num")

print(label[-1:])



data.columns = label



data.drop(data[data.label_num == 0].index, inplace=True)



data['label_num'] = data['label_num'] - 1

data['label_num'].unique()
y = data[['label_num']]

print(y)

y = y.astype('int')





X = data.loc[:, data.columns != 'label_num']

print(X)



print(X.shape, y.shape)



label_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill', 

              'Soybean-clean', 'Wheat', 'Wood', 'Building-Grass-Trees-Drives', 'Stone-Steel-Towers']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=121)





from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix



svm = SVC(kernel = 'rbf')

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)





print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)



for c in range(16):

  tp = cm[c,c]

  fp = cm[:,c].sum() - tp

  fn = cm[c,:].sum() - tp

  tn = cm.sum() - fp - fn - tp



  acc = tp/(tp+fp+fn)

  rec =  tp/(tp+fn)

  prec = tp/(tp+fp)

  f1 = 2*tp/(2*tp+fn+fp)

  print(f'Class: {c}, {tp}, Acc: {acc:2.2f}, Rec:{rec:2.2f}, Prec: {prec:2.2f}')
df_cm = pd.DataFrame(cm,columns = label_names, index = label_names)





plt.figure(figsize = (10,10))

sns.set(font_scale=1)

sns_plot = sns.heatmap(df_cm, cmap = 'Blues', annot = True, fmt='g', vmin=0, vmax=200)

ax.invert_yaxis()

ax.tick_params(labeltop=True)
svm = SVC(kernel = 'linear')

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)





print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)



for c in range(16):

  tp = cm[c,c]

  fp = cm[:,c].sum() - tp

  fn = cm[c,:].sum() - tp

  tn = cm.sum() - fp - fn - tp



  acc = tp/(tp+fp+fn)

  rec =  tp/(tp+fn)

  prec = tp/(tp+fp)

  f1 = 2*tp/(2*tp+fn+fp)

  print(f'Class: {c}, {tp}, Acc: {acc:2.2f}, Rec:{rec:2.2f}, Prec: {prec:2.2f}')
df_cm = pd.DataFrame(cm,columns = label_names, index = label_names)





plt.figure(figsize = (10,10))

sns.set(font_scale=1)

sns_plot = sns.heatmap(df_cm, cmap = 'Blues', annot = True, fmt='g', vmin=0, vmax=200)

ax.invert_yaxis()

ax.tick_params(labeltop=True)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(max_depth = 15, random_state=121)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)



print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)



for c in range(16):

  tp = cm[c,c]

  fp = cm[:,c].sum() - tp

  fn = cm[c,:].sum() - tp

  tn = cm.sum() - fp - fn - tp



  acc = tp/(tp+fp+fn)

  rec =  tp/(tp+fn)

  prec = tp/(tp+fp)

  f1 = 2*tp/(2*tp+fn+fp)

  print(f'Class: {c}, {tp}, Acc: {acc:2.2f}, Rec:{rec:2.2f}, Prec: {prec:2.2f}')
df_cm = pd.DataFrame(cm,columns = label_names, index = label_names)





plt.figure(figsize = (10,10))

sns.set(font_scale=1)

sns_plot = sns.heatmap(df_cm, cmap = 'Blues', annot = True, fmt='g', vmin=0, vmax=200)

ax.invert_yaxis()

ax.tick_params(labeltop=True)
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt

import torch

import time
class Net(nn.Module):

    # Init function : defines nn architecture

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(200, 100)

        self.fc2 = nn.Linear(100, 50)

        self.fc3 = nn.Linear(50, 16)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(num_features = 100)

        self.bn2 = nn.BatchNorm1d(num_features = 50)

        self.bn3 = nn.BatchNorm1d(num_features = 16)

        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p = 0.5)



    # Forward function : contains

    def forward(self, x):

        x = self.fc1(x)

        x = self.bn1(x)

        x = self.relu(x)

        # x = self.dropout(x)

        x = self.fc2(x)

        x = self.bn2(x)

        x = self.relu(x)

        # x = self.dropout(x)

        x = self.fc3(x)

        x = self.bn3(x)

        x = self.relu(x)

        # x = self.dropout(x)

        out = self.softmax(x)

        return out



class MyNet(nn.Module):

    # Init function : defines nn architecture

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(200,50)

        self.fc2 = nn.Linear(50, 50)

        self.fc3 = nn.Linear(50, 16)

        # self.softmax = nn.Softmax(dim=1)



    # Forward function : contains

    def forward(self, X):

        X = F.relu(self.fc1(X))

        X = self.fc2(X)

        X = self.fc3(X)

        # X = self.softmax(X)

        return X



class CustomDataset(Dataset):

    def __init__(self, data, label_num):

        super().__init__()

        self.data = data

        self.label = label_num



    def __len__(self):

        return self.data.shape[0]



    def __getitem__(self, idx):

        d = {'data' : torch.from_numpy(self.data[idx].astype('int64')).type(torch.float32),

            'label': torch.Tensor([self.label[idx]]).type(torch.int64)

            }

        return d



train_dataset = CustomDataset(X_train, y_train)

test_dataset = CustomDataset(X_test, y_test)



num_epochs = 40

BATCH_SIZE = 8



trainloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)

testloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)



my_net = Net()

# print(my_net)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(my_net.parameters(), lr = 0.03)



loss_list = []



# trainloader(0)

# count = 0

# for batch in trainloader :

#     if count == 0 :

#         y = batch['label'].view(-1)

#         yy = batch['label']

#         print(y)

#         print(yy)

#         count += 1

#     else :

#         break



start_time = time.time()

for epoch in range(num_epochs):

    for batch in trainloader:

        optimizer.zero_grad()



        X = batch['data']

        y = batch['label'].view(-1)



        y_pred = my_net(X)

        

        total += float(y.size()[0])

        correct += float((torch.argmax(y_pred, axis=1) == y).sum())

        

        loss = criterion(y_pred, y)



        loss.backward()



        optimizer.step()

    loss_list.append(loss)

    # print('Epoch {} done training and the loss is {}'.format(epoch+1, loss))

print('time spent training : {}'.format(time.time() - start_time))



sns.set_style("darkgrid")

plt.plot(loss_list)

plt.show()



sns.set_style("darkgrid")

plt.plot(loss_list)

plt.show()



len(loss_list)