# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/naivealgo/Naive.csv')
X = df.drop('SERIAL',axis=1)
Y = df['SERIAL']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
Accuracy = accuracy_score(y_test,y_pred)*100
accuracy