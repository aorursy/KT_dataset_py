# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import seaborn as sns 

import matplotlib.pyplot as plt
#Importing the dataset

df = pd.read_csv("/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv",sep=",")
df.head()
df.tail()
#Dataset Information

df.info()
#Statistical Description of the dataset

df.describe().T
# Examining the banking_crisis column 

sns.countplot(df["banking_crisis"])

plt.show()
df["banking_crisis"].value_counts()
#Relation between the different Countries and their exhange rate wrt USA



plt.figure(figsize=[10,5])

sns.boxplot(df["country"],df["exch_usd"])

plt.xticks(rotation = 90)

plt.show()

#Dropping the country code column

df1 = df.drop("cc3",axis = 1)
#Label Encoding the "country" and "banking_crisis" columns 

from sklearn.preprocessing import LabelEncoder
df1["country"]= LabelEncoder().fit_transform(df1["country"].tolist())
df1["banking_crisis"]= LabelEncoder().fit_transform(df1["banking_crisis"].tolist())
df1.head()
#Importing the required Libraries

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

from statistics import mean

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report
### We shall separate the features and the target in the form of X and Y

x = df1.drop('banking_crisis',axis = 1)

y = df1['banking_crisis']
#Using Train test split 

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3 , random_state = 0)
#Checking the shapes of the training and testing data

xtrain.shape
ytrain.shape
giniDecisionTree = DecisionTreeClassifier(criterion='gini',random_state = 0,max_depth=3, min_samples_leaf=5)



giniDecisionTree.fit(xtrain,ytrain)
#Prediction 

giniPred = giniDecisionTree.predict(xtest)
#Accuracy score 

print('Accuracy Score: ',accuracy_score(ytest, giniPred))
#Classification Report 

print('Classification Report')

print(classification_report(ytest, giniPred))
entropyDecisionTree = DecisionTreeClassifier(criterion='entropy',random_state = 100,max_depth=3, min_samples_leaf=5)
entropyDecisionTree.fit(xtrain,ytrain)
#Predictions 

entropyPred = entropyDecisionTree.predict(xtest)
#Accuracy 

print('Accuracy Score: ',accuracy_score(ytest, entropyPred))