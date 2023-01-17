# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load





"""

DATA MANIPULATİNG

"""

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



"""

NATURAL LANGUAGE PROCESSING

"""

import re 

import nltk 

from sklearn.feature_extraction.text import CountVectorizer



"""

PYTORCH

"""



import torch

import torch.nn as nn





"""

VISUALIZATION TOOLS

"""



import matplotlib.pyplot as plt

import seaborn as sns



"""

TRAIN TEST SPLIT

"""

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
true_data = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

fake_data = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')

true_data.head()
true_data.info()
fake_data.info()
# Adding labels 

true_data["label"] = np.ones(len(true_data),dtype=int)

fake_data["label"] = np.zeros(len(fake_data),dtype=int)



true_data.head()
data = pd.concat((true_data,fake_data),axis=0)

print(data.info())
data = data.sample(frac=1)

data.head(10)
sns.countplot(data["label"])

plt.show()
data["subject"].value_counts()
data = pd.get_dummies(data,columns=["subject"])

data.head()
data = data.drop("date",axis=1)

data.info()
new_text = []

pattern = "[^a-zA-Z]"

lemma = nltk.WordNetLemmatizer()



for txt in data.text:

    

    txt = re.sub(pattern," ",txt) # Cleaning

    txt = txt.lower() # Lowering

    txt = nltk.word_tokenize(txt) # Tokenizing

    txt = [lemma.lemmatize(word) for word in txt] # Lemmatizing

    txt = " ".join(txt)

    new_text.append(txt)

    

    

new_text[0]

    
new_title = []

for txt in data.title:

    

    txt = re.sub(pattern," ",txt) # Cleaning

    txt = txt.lower() # Lowering

    txt = nltk.word_tokenize(txt) # Tokenizing

    txt = [lemma.lemmatize(word) for word in txt] # Lemmatizing

    txt = " ".join(txt)

    new_title.append(txt)

new_title[0]

vectorizer_title = CountVectorizer(stop_words="english",max_features=1000)

vectorizer_text = CountVectorizer(stop_words="english",max_features=4000)



title_matrix = vectorizer_title.fit_transform(new_title).toarray() 

text_matrix = vectorizer_text.fit_transform(new_text).toarray()



print("Finished")
data.head()
data.drop(["title","text"],axis=1,inplace=True)

data.info()
print(data.shape)

print(title_matrix.shape)

print(text_matrix.shape)
# Creating Y

y = data.label

# Creating X

x = np.concatenate((np.array(data.drop("label",axis=1)),title_matrix,text_matrix),axis=1)



print(x.shape)

print(y.shape)
from sklearn.model_selection import train_test_split



# Train Test Split

X_train,X_test,Y_train,Y_test = train_test_split(x,np.array(y),test_size=0.25,random_state=1)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)

class ANN(nn.Module):

    

    def __init__(self):

        

        super(ANN,self).__init__() # Inhertiting

        

        self.linear1 = nn.Linear(5008,2000) # IN 5008 OUT 2000

        self.relu1 = nn.ReLU() # Actfunc 1

        

        self.linear2 = nn.Linear(2000,500) # IN 2000 OUT 500

        self.relu2 = nn.ReLU()

        

        self.linear3 = nn.Linear(500,100) # IN 500 OUT 100

        self.relu3 = nn.ReLU()

        

        self.linear4 = nn.Linear(100,20) # IN 100 OUT 20

        self.relu4 = nn.ReLU()

        

        self.linear5 = nn.Linear(20,2) # IN 20 OUT 2 | OUTPUT 

        

    

    def forward(self,x):

        

        out = self.linear1(x) # Input Layer 

        out = self.relu1(out)

        

        out = self.linear2(out) # Hidden Layer 1 

        out = self.relu2(out)

        

        out = self.linear3(out) # Hidden Layer 2 

        out = self.relu3(out)

        

        out = self.linear4(out) # Hidden Layer 3 

        out = self.relu4(out)



        

        out = self.linear5(out) # Output Layer

        

        return out

    



model = ANN()

optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

error = nn.CrossEntropyLoss()
# Converting numpy arrays into pytorch tensors

X_train = torch.Tensor(X_train)



# You must convert it into LongTensor. I did it once

Y_train = torch.Tensor(Y_train).type(torch.LongTensor)



X_test = torch.Tensor(X_test)

Y_test = torch.Tensor(Y_test)



EPOCHS = 20



for epoch in range(EPOCHS):

    

    # Clearing gradients

    optimizer.zero_grad()

    

    # Forward Propagation

    outs = model(X_train)

    

    # Computing Loss

    loss = error(outs,Y_train)

    

    # Backward propagation

    loss.backward()

    

    # Updating parameters

    optimizer.step()

    

    # Printing loss

    print(f"Loss after iteration {epoch} is {loss}")

    

    
# Importing metrics

from sklearn.metrics import accuracy_score,confusion_matrix





# Prediction

y_head = model(X_test)

print(y_head[0])

# Converting Prediction into labels

y_pred = torch.max(y_head,1)[1]

print(y_pred[0])



# Accuracy score

print("Model accuracy is ",accuracy_score(y_pred,Y_test))

confusion_matrix = confusion_matrix(y_pred=y_pred,y_true=Y_test)



fig,ax = plt.subplots(figsize=(6,6))

sns.heatmap(confusion_matrix,annot=True,fmt="0.1f",linewidths=1.5)

plt.show()