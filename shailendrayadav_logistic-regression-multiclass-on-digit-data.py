import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.datasets import load_digits
digits=load_digits()
dir(digits)
plt.matshow(digits.images[0])# to show images in pixel

plt.gray() # show images in black and white
for i in range(0,10):

    plt.matshow(digits.images[i]) # shoxws image from 1-9
digits.target
digits.target_names
#features

X= digits.data

#label

Y=digits.target
#test train and split

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2,random_state=1)# train/test is 80/20.

# test and train data is split, lets choose a classifier

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
#training

lr.fit(X_train,Y_train)
#prediction 

y_predict=lr.predict(X_test)
#model score

lr.score(X_test,Y_test)
#lets see the confusion matrix for the same

from sklearn.metrics import confusion_matrix

cn=confusion_matrix(Y_test,y_predict)

cn
#lets visualise the same on Heatmap

plt.figure(figsize=(6,6))

sns.heatmap(cn,annot=True)