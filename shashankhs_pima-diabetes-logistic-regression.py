import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
pima = pd.read_csv("../input/diabetes.csv")
pima.head(5)
pima.info() #Just gives some meta data about the dataframe
pima.shape #Gives number of (rows, columns) in data
train, test = train_test_split(pima, test_size = 0.25)
train.shape
test.shape
train_feat = train.iloc[:,:8]
train_targ = train['Outcome']
lr = LogisticRegression(solver = 'liblinear')
lr.fit(train_feat,train_targ)
lr.score(train_feat, train_targ)
#Now we split the test data and find the accuracy score
test_feat = test.iloc[:,:8]
test_targ = test['Outcome']
lr.score(test_feat, test_targ)
#confusion matrix for our test on the training data
confusion_matrix(lr.predict(train_feat), train_targ)
#confusion matrix for our test on the test data
confusion_matrix(lr.predict(test_feat), test_targ)
