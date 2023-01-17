#pip install category-encoders
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tqdm import tqdm_notebook 
import seaborn as sns
import time
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder
import torch
import category_encoders as ce
ds = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
ds.head(3)
#ds.info()
ds.isnull().sum()
no_salary = 0
ds.fillna({'salary' : no_salary}, inplace=True)
ds.isnull().sum()
ds.drop(['sl_no'],axis =1,inplace = True)

ds.head(1)
x = 'salary'
data = ds[ds["salary"].notna()]
fig, ax = plt.subplots()
fig.set_size_inches(22, 10)
plt.xticks(rotation=90);
sns.countplot(x = x,palette="ch:.4", data = data)
ax.set_xlabel('Salary', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Salary Distribution', fontsize=15)
sns.despine()
plt.figure(figsize =(18,10))
plt.subplot(221)
sns.boxplot("salary", "gender", data=ds)
plt.subplot(222)
sns.boxplot("salary", "specialisation", data=ds)
plt.subplot(223)
sns.boxplot("salary", "degree_t", data=ds)
plt.subplot(224)
sns.boxplot("salary", "workex", data=ds)
plt.show()
plt.figure(figsize =(18,10))
plt.subplot(221)
sns.countplot(ds["gender"],hue=ds["status"])
plt.subplot(222)
sns.countplot(ds["specialisation"],hue=ds["status"])
plt.subplot(223)
sns.countplot(ds["workex"],hue=ds["status"])
plt.subplot(224)
sns.countplot(ds["degree_t"],hue=ds["status"])
plt.show()
ds.status = [1 if each == "Placed" else 0 for each in ds.status]
ds.head(3)
# encode the columns
categorical_columns = ['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation']
encoder = ce.OneHotEncoder(cols=categorical_columns, use_cat_names=True)
ds_encoded = encoder.fit_transform(ds)
ds_encoded.head(1)
p_status = ds_encoded.status.values
salary = ds_encoded.salary.values
p_status
data = ds_encoded.drop(['status',"salary"], axis=1).to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(data, p_status, stratify=p_status, test_size=0.2,random_state=1)
print(X_train.shape, X_test.shape, p_status.shape)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)
X_train, Y_train, X_test, Y_test = map(torch.tensor, (X_train, Y_train, X_test, Y_test))
print(X_train.shape, Y_train.shape)
X_train = X_train.float()
Y_train = Y_train.float() 
X_test = X_test.float()
Y_test = Y_test.float()
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
# Define the network and forward pass

class CNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
        nn.Linear(21, 128),
        nn.Sigmoid(),
        nn.Linear(128,1),
        nn.Sigmoid())
    def forward(self, X):
        xx = self.net( X)
        xx = torch.reshape(xx, (-1,))
        return xx
# define the fitting function
def fitt(x, y, model, opt, loss_fn, epochs = 250):
    loss_train_arr = []
    loss_test_arr = []
    acc_train_arr = []
    acc_test_arr = []
    for i in tqdm_notebook(range(epochs),total = epochs, unit = 'epoch'):
        loss_train = loss_fn(model(x), y) 
        loss_test = loss_fn(model(X_test), Y_test) 
    
        threshold = .77
        y_ht_train = (model(x)>=threshold).to(torch.float).numpy()
        y_ht_test  = (model(X_test)>=threshold).to(torch.float).numpy()
        accuracy_train = accuracy_score(y,y_ht_train)
        accuracy_test = accuracy_score(Y_test,y_ht_test)
    
        acc_train_arr.append(accuracy_train.item())
        acc_test_arr.append(accuracy_test.item())
        loss_train_arr.append(loss_train.item())
        loss_test_arr.append(loss_test.item())
    
        loss_train.backward()
        opt.step()
        opt.zero_grad()
        if i%50 == 0:
            print("Epoch : {} , Test_Loss : {:.3f}, Train_Loss : {:.3f}".format(i,loss_test,loss_train))
  
    y_ht_test  = (model(X_test)>=threshold).to(torch.float).numpy()
    print(classification_report(Y_test.detach().numpy() , y_ht_test))

    C_mat = confusion_matrix(Y_test.detach().numpy(),y_ht_test)
    sns.heatmap(C_mat,annot=True,fmt="d")
  
    plt.figure(figsize=(10, 7))
    plt.subplot(121)
    plt.plot(loss_train_arr, 'r-',label = "Training Loss")
    plt.plot(loss_test_arr, 'g-', label = "Test Loss")
    plt.legend(["Training Loss", "Test loss"], loc ="best") 
  
    plt.subplot(122)
    plt.plot(acc_train_arr, 'r-',label = "Training Loss")
    plt.plot(acc_test_arr, 'g-', label = "Test Loss")
    plt.legend(["Training accuracy", "Test accuracy"], loc ="best") 
  
    plt.show()      
    print("Final Test Accuracy : {:.3f}, Final Test Loss : {:.3f}".format(acc_test_arr[-1],loss_test_arr[-1]))
    return
# Run
fn = CNNetwork()
loss_fn = F.binary_cross_entropy
#opt = optim.SGD(fn.parameters(), lr=.8)
opt = optim.Adam(fn.parameters(), lr=0.002)
fitt(X_train, Y_train, fn, opt, loss_fn)
# Testing the model
test = X_test[8]

with torch.no_grad():
    fn.eval()
    threshold = 0.77
    output = fn.forward(test)
    if output >=threshold:
        print("Higher Probability of placement!")
        print("Chances = {:.1f} %".format(output.item()*100))

