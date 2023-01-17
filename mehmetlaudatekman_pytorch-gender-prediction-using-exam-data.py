# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



import torch

import torch.nn as nn



from sklearn.model_selection import train_test_split



import warnings as wrn



wrn.filterwarnings('ignore') # Filter unrelevant warnings

sns.set_style("darkgrid")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing data

data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
data.head()
data.info()
data["gender"].value_counts()
fig,ax = plt.subplots(figsize=(8,6))

sns.countplot(data["gender"])

plt.show()
data["race/ethnicity"].value_counts()
fig,ax = plt.subplots(figsize=(8,6))

sns.countplot(data["race/ethnicity"])

plt.show()
data["parental level of education"].value_counts()
fig,ax = plt.subplots(figsize=(8,6))

sns.countplot(data["parental level of education"])

plt.xticks(rotation=60)

plt.show()
data["lunch"].value_counts()
fig,ax = plt.subplots(figsize=(8,6))

sns.countplot(data["lunch"])

plt.show()
data["test preparation course"].value_counts()
fig,ax = plt.subplots(figsize=(8,6))

sns.countplot(data["test preparation course"])

plt.show()
fig,ax = plt.subplots(figsize=(8,6))

sns.distplot(data["math score"])

plt.show()
fig,ax = plt.subplots(figsize=(8,6))

sns.distplot(data["writing score"])

plt.show()
fig,ax = plt.subplots(figsize=(8,6))

sns.distplot(data["reading score"])

plt.show()
data.corr()
fig,ax = plt.subplots(figsize=(6,6))

sns.heatmap(data.corr(),annot=True,fmt="0.2f",linewidths=1.5)

plt.show()
gender_math = data.groupby("gender")["math score"].mean()

gender_math
sns.barplot(gender_math.index,gender_math.values)

plt.show()
gender_writing = data.groupby("gender")["writing score"].mean()

gender_writing
sns.barplot(gender_writing.index,gender_writing.values)

plt.show()

gender_reading = data.groupby("gender")["reading score"].mean()

gender_reading
sns.barplot(gender_reading.index,gender_reading.values)

plt.show()
race_math = data.groupby("race/ethnicity")["math score"].mean()

race_math
sns.barplot(race_math.index,race_math.values)

plt.show()
race_writing = data.groupby("race/ethnicity")["writing score"].mean()

race_writing
sns.barplot(race_writing.index,race_writing.values)

plt.show()
race_reading = data.groupby("race/ethnicity")["reading score"].mean()

race_reading
sns.barplot(race_reading.index,race_reading.values)

plt.show()
print(data.values[0])

data["gender"] = [1 if each == "female" else 0 for each in data["gender"]]

data.head(1)
x = data.drop("gender",axis=1)

y = data.gender

x.head()
x_encoded = pd.get_dummies(x,columns=["race/ethnicity","parental level of education","lunch","test preparation course"])

x_encoded.head()
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range=(0,1))



x_scaled = scaler.fit_transform(x_encoded)

x_scaled[0]
x_scaled.shape
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=1)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
class ANN(nn.Module):

    

    def __init__(self):

        

        super(ANN,self).__init__()

        

        # Linear function 1

        self.linear1 = nn.Linear(18,10) # 18 to 10

        self.tanh1 = nn.Tanh()

        

        # Linear function 2

        self.linear2 = nn.Linear(10,6) # 10 to 6

        self.tanh2 = nn.Tanh()

        

        # Linear function 3

        self.linear3 = nn.Linear(6,2) # 6 to output

        

    

    def forward(self,x):

        

        out = self.linear1(x)

        out = self.tanh1(out)

        

        out = self.linear2(out)

        out = self.tanh2(out)

        

        out = self.linear3(out)

        return out

    



model = ANN()

optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

error = nn.CrossEntropyLoss()
# But before fitting, we must convert numpy arrays into torch tensors



x_train = torch.Tensor(x_train)

x_test = torch.Tensor(x_test)

y_train = torch.Tensor(y_train).type(torch.LongTensor)
epochs = 200

for epoch in range(epochs):

    

    # Clearing gradients

    optimizer.zero_grad()

    

    # Forward propagation

    outs = model(x_train)

    

    # Computing loss

    loss = error(outs,y_train)

    

    # Backward propagation

    loss.backward()

    

    # Updating parameters

    optimizer.step()

    

    if epoch%50 == 0:

        print(f"Cost after iteration {epoch} is {loss}")
from sklearn.metrics import accuracy_score

# Predicting 

y_head = model(x_test)

print(y_head[0])





# Converting predictions into labels

y_pred = torch.max(y_head,-1)[1]

print(y_pred[0])



print("Accuracy of model is ",accuracy_score(y_pred,y_test))