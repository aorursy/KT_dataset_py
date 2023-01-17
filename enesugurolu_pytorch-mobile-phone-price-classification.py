import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sys
from sklearn.datasets import load_boston
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import math
import torch.nn.functional as F
from sklearn.metrics import precision_score
df = pd.read_csv('../input/mobile-price-classification/train.csv')
df = df.sort_values(by=['price_range'])
df.head()
df.shape
df['Res'] = df['px_height']*df['px_width']
X = df.drop(['px_height','px_width'],1)
y = df['price_range']
X.head()
y.value_counts()
pd.options.display.float_format = '{:,.2f}'.format
corr_matrix = X.corr()
corr_matrix
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
plt.show()

x = df[['battery_power','ram']]

Y = y.values
plt.scatter(x.values[:,0],x.values[:,1],c=Y)
plt.xlabel('Ram')
plt.ylabel('Battery')
plt.title('Price Range Distrubtion')
X = X[['ram','battery_power','Res']]
X = X.to_numpy()
y = y.to_numpy().reshape(-1,1)
plt.scatter(X[:,0],X[:,1], c=y)
plt.xlabel('Ram')
plt.ylabel('Battery Power')
plt.title('Result Distribution')
class Model(nn.Module):

  def __init__(self,in_features=3,h1=8,h2=9,out_features=4):
    # how many layers?
    super().__init__()
    
    self.fc1 = nn.Linear(in_features,h1)
    self.fc2 = nn.BatchNorm1d(h1)
    self.fc3 = nn.Linear(h1,h2)
    self.fc4 = nn.BatchNorm1d(h2)
    self.out = nn.Linear(h2,out_features)

   
    
  def forward(self,x):
    
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = self.out(x)
    return x
model = Model()
model
X_train = torch.FloatTensor(X)
y_train = torch.LongTensor(y).reshape(-1)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
epochs = 2000
losses = []

for i in range(epochs):

  # forward and get a prediction
  y_pred = model.forward(X_train)
  # calculate loss/error
  loss = criterion(y_pred,y_train)
  losses.append(loss)
  if i%100==0:
    print('Epoch: {} and Loss: {}'.format(i,loss))
  #Backpropagation
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
y_item_2 = []
for i in range(2000):
  y_item_2.append(y_pred[i].argmax())
y_item_2 = np.array(y_item_2)
correct = 0
for i in range(2000):
  if y_item_2[i]==y_train[i]:
    correct = correct + 1
print('Accuracy: {}'.format(correct*100/2000))
plt.plot(range(epochs),losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
df_2 = pd.read_csv('../input/mobile-price-classification/test.csv')
df_2['Res'] = df_2['px_height']*df_2['px_width']
X_val = df_2[['ram','battery_power','Res']]
X_val = X_val.to_numpy()
X_test = torch.FloatTensor(X_val)
# TO EVALUATE THE ENTIRE TEST SET
with torch.no_grad():
    y_pred_2 = model.forward(X_test)
    

from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
y_item = []
for i in range(1000):
  y_item.append(y_pred_2[i].argmax())
y_item = np.array(y_item)
X_test_plot = X_test[:,[0,1]]
svm = SVC(C=0.5, kernel='linear')
svm.fit(X_test_plot, y_item)
plot_decision_regions(X_test_plot.numpy(), y_item, clf=svm, legend=2)
plt.xlabel('RAM')
plt.ylabel('Battery Power')
plt.title('Decision Boundary')
plt.show()