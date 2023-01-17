# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.naive_bayes import MultinomialNB # Using multinomial because our dataset is discrete

from sklearn.model_selection import train_test_split # Splitting test and train data

from sklearn.metrics import accuracy_score # To calculate accuracy



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing data set and preprocessing

dataset = pd.read_csv("../input/train.csv")

X = dataset.iloc[:, dataset.columns != 'label'].values

y = dataset.iloc[:, 0].values

y = y.reshape(-1, 1) # Convert vector to matrix of [1,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# MultinomialNB object creation and fitting our model

mnb = MultinomialNB()

mnb.fit(X_train, y_train)
#Prediction

y_pred = mnb.predict(X_test)



#Accuracy Calculation

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)



#Submission 

dataset_test = pd.read_csv("../input/test.csv")

x_question = dataset_test.values

y_submit = mnb.predict(x_question)
#New pandas dataframe for submission

df_to_submit = pd.DataFrame()

df_to_submit = pd.DataFrame(columns=[ 'Labels'])

df_to_submit['Labels'] = y_submit

print(df_to_submit)