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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix,auc,roc_auc_score

from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

# Load CSV

df = pd.read_csv("../input/Amazon_train.csv")



df_pred = pd.read_csv("../input/Amazon_test_atrib.csv")



# Datashape, Features, stat. description.

print(df.shape)



print("First 5 lines:")

print(df.head(5))



print("describe: ")

print(df.describe())



print("info: ")

print(df.info())
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



y = df['Clase'].values #target

X = df.drop(['Clase'],axis=1).values #features



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=20, stratify=y)



print("trainset size: ", len(y_train),"\ntestset size: ", len(y_test))

# Classifier

clf = LogisticRegression()

# fit 

clf.fit(X_train,y_train)

# Predict

y_pred = clf.predict(X_test)

# Predicted probabilities

y_pred_prob = clf.predict_proba(X_test)

#trainset predictions

train_pred = clf.predict(X_train)

print('train-set confusion matrix:\n', confusion_matrix(y_train,train_pred)) 



print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred)) 

print("recall score: ", recall_score(y_test,y_pred))

print("precision score: ", precision_score(y_test,y_pred))

print("f1 score: ", f1_score(y_test,y_pred))

print("accuracy score: ", accuracy_score(y_test,y_pred))

print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob[:,1])))
#train whole data

y_train = df['Clase'].values 

X_train = df.drop(['Clase'],axis=1).values 

X_test = df_pred.values 



clf.fit(X_train,y_train)

# predict using test data

y_pred = clf.predict(X_test)

# Predicted probabilities

y_pred_prob = clf.predict_proba(X_test)



# Final Submission

Id=np.arange(1,393)

Id= Id.flatten()

my_submission = pd.DataFrame({'ID': Id, 'Clase': y_pred})



print(my_submission)

# you could use any filename. We choose submission here

my_submission.to_csv('submission_Sample_final.csv', index=False)