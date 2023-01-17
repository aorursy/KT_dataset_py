import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from keras.layers import Dense, Dropout, Activation

from keras.models import Sequential

from keras.optimizers import SGD
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

full = pd.concat([train,test],ignore_index=True)
full.head()
full.drop(["PassengerId"],axis=1,inplace=True)
full.info()
train.isna().mean()
# Replace the NaN values with "Unknown" and "Known" for the other ones

lab = lambda x : "Known" if type(x) == str else "Unknown"

full["Cabin"] = full["Cabin"].apply(lab)
full.groupby("Cabin").mean()["Survived"].plot(kind="bar",ylabel="Survival rate")
# Creating a categorical variable for the passenger's title

text = []

for i in full["Name"]:

    text.append(i.split(",")[1].split(".")[0].strip())

full["Name"] = text





full['Name'] = full['Name'].replace(['Capt', 'Col',

    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

full['Name'] = full['Name'].replace(['the Countess', 'Lady', 'Sir'], 'Royal')

full['Name'] = full['Name'].replace('Mlle', 'Miss')

full['Name'] = full['Name'].replace('Ms', 'Miss')

full['Name'] = full['Name'].replace('Mme', 'Mrs')
full.head()
# Label Encoding and One Hot Encoding for categorical features

full["Sex"] = LabelEncoder().fit_transform(full["Sex"])

full["Cabin"] = LabelEncoder().fit_transform(full["Cabin"])



full = pd.concat([full,pd.get_dummies(full["Name"])],axis=1)

full.head(10)
full.groupby("Name").mean()["Age"]
# So what i'm doing here is, replacing the median age for each name title



for i in full["Name"].unique():

    full.loc[(full["Age"].isnull()) & (full["Name"] == i),"Age"] = full.loc[(full["Age"].isnull()) & (full["Name"] == i),"Age"].replace(np.nan,full.groupby("Name").mean()["Age"][i])
full.isna().mean()
train = full[ :891]

test  = full[891: ]
# Selecting the useful columns 

features = ["Age","Fare","Pclass","Sex","Cabin","Master","Miss","Mr","Mrs","Rare","Royal"]
x = train[features]

y = train["Survived"]
# The final dataset

x.head()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
# Set the random seed for reproducible results

seed_value= 0



import os

os.environ['PYTHONHASHSEED']=str(seed_value)



import random

random.seed(seed_value)



import numpy as np

np.random.seed(seed_value)



import tensorflow as tf

tf.random.set_seed(seed_value)



model = Sequential()

model.add(Dense(64,input_dim=11))

model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(64))

model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(2))

model.add(Activation("softmax"))



y_train = pd.get_dummies(y_train)



opt = SGD(learning_rate=0.1)



model.compile(optimizer= opt ,loss = "categorical_crossentropy" , metrics = ["accuracy"])

model.fit(x_train,y_train,epochs=50,batch_size=64,validation_split=0.1)
preds = model.predict_classes(x_test)

sns.heatmap(confusion_matrix(y_test,preds),annot=True)
submission = model.predict_classes(scaler.transform(test[features]))
