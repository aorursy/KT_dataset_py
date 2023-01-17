#importing neccessary libraries 

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import  LogisticRegression

from sklearn.metrics import accuracy_score, classification_report
#using pandas library and 'read_csv' function to read csv file

data = pd.read_csv("../input/heart-disease-uci/heart.csv")

#examining the data

data.head(10)

#finding the length of the datset

len(data)
#finding the type of every column

data.dtypes
data.target.value_counts()
#finding how many patients have heart disease

sns.countplot(x = "target", data =  data)

plt.show()
#finding the gender ratio of the patients

sns.countplot(x ='sex', data = data, palette ='bwr')

plt.xlabel("Sex (0 = female, 1 = male)")

plt.show()
x = data.drop('target', 1) #data excluding the labels

y = data['target'] #labels 

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = .2, random_state = 42) #split the data into train and test sets

X_train.shape, Y_train.shape, X_test.shape, Y_test.shape 
model = LogisticRegression() #using logistic regression

model.fit(X_train, Y_train) #fitting the model on training data
Y_test_hat = model.predict(X_test) #predicting 

accuracy_score(Y_test, Y_test_hat)
print(classification_report(Y_test, Y_test_hat, digits=6))