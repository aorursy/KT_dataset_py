# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/train.csv")
dataset.columns
#dataset['Ticket']
#dataset[dataset['Embarked'] == 'Q']['Embarked'].count()
#print(age_not_nan)
survived_not_nan = dataset['Age'].notnull() & dataset['Survived'] == 1
age_not_survived = dataset[(dataset['Age'].notnull()) & (dataset['Survived'] == 0) & (dataset['Embarked'] == 'Q') & (dataset['Pclass'] == 3)]['Age']
count_not_survived= dataset[dataset['Age'].notnull() & (dataset['Survived'] == 0) & (dataset['Embarked'] == 'Q') & (dataset['Pclass'] == 3)]['Fare']
#print(count_not_survived.count())
#print(age_not_survived.count())
#print(dataset[dataset['Survived'] == 0]['Survived'])
#print(dataset[not_survived_not_nan].iloc[:,5])
#print(not_survived_not_nan)
plt.figure(figsize=(15, 10))
fig, ax = plt.subplots(2,2, figsize=(15, 10))
plt.subplot(221)
plt.title("Survivours Age")
plt.xlabel("Age")
plt.ylabel("No. of Survivours")
plt.hist(dataset[survived_not_nan].iloc[:,5])
#fig, ax = plt.subplot(222)
#plt.ylim(0, 500)
plt.subplot(222)
plt.xlabel("Age")
plt.ylabel("No. Non-of Survivours")
plt.title("Non-Survivours Age")
age_not_survived.hist()
plt.subplot(223)
plt.xlabel("Fare")
plt.ylabel("No. of Survivours Fare")
plt.title("Survivours Fare")
plt.hist(dataset[survived_not_nan].iloc[:, 9])
plt.subplot(224)
plt.xlabel("Fare")
plt.ylabel("No. of Non-Survivours Fare")
plt.title("Non-Survivours Fare")
count_not_survived.hist()
plt.figure(figsize=(15, 5))
plt.xlabel("Age")
plt.ylabel("Fare")
plt.scatter(x=age_not_survived, y=count_not_survived)
x = dataset.iloc[:, [2,4,5,6,7]].values
y = dataset.iloc[:, 1].values
#Filling the NA's in Age column by men strategy
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy='mean', axis = 0)
imputer = imputer.fit(x[:,2:3])
x[:, 2:3] = imputer.transform(x[:, 2:3])
#Let's do Label encoding, we have a column "Sex" which has input values "male" and "female" lets convert it to 0's and 1's
#in int format
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()  
x[:, 1] = labelencoder.fit_transform(x[:, 1])  
x
#Now the final step is to scale our data so that all no columns dominate the output
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#lets seperate the data training and testing data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#Let's use LogisticRegression to fit our training data
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
print("Accurancy: "+str((cm[0,0]+cm[1,1])/cm.sum()))
