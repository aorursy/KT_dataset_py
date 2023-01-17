# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import arff

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Trainingsdaten laden

dataset = arff.load(open('../input/classifier-train/train.arff'))

data = np.array(dataset['data'])

data = data.astype(np.float64)



df_train = pd.DataFrame({'X':data[:, 0], 'Y':data[:, 1], 'class':data[:,-1].astype(int)})

print(df_train.head())

print(df_train.shape)



X_train = df_train[['X','Y']]

y_train = df_train['class']



plt.figure(figsize=(8,6))

sns.scatterplot(x=df_train['X'], y=df_train['Y'],hue=df_train['class'])

plt.title("Trainingsdaten")

plt.show()
#Testdaten laden

test_dataset = arff.load(open('../input/classifier-test/eval.arff'))

test_data = np.array(test_dataset['data'])

test_data = test_data.astype(np.float64)



df_test = pd.DataFrame({'X':test_data[:, 0], 'Y':test_data[:, 1], 'class':test_data[:,-1].astype(int)})



X_test = df_test[['X','Y']]

y_test = df_test['class']



plt.figure(figsize=(8,6))

sns.scatterplot(x=df_test['X'], y=df_test['Y'],hue=df_test['class'])

plt.title("Testdaten")

plt.show()



print(df_test.shape)
rfc = RandomForestClassifier(n_estimators=250)

rfc.fit(X_train, y_train)



predictions = []



for i,row in X_test.iterrows():

    X=[[row['X'], row['Y']]]

    if rfc.predict(X) == y_test[i]:

        predictions.append(row)
df_pred = pd.DataFrame(predictions)



plt.figure(figsize=(8,6))

sns.scatterplot(x=df_test['X'], y=df_test['Y'],hue=df_test['class'])

sns.scatterplot(x=df_pred['X'], y=df_pred['Y'])

plt.title("Testdaten")

plt.show()



print("Accuracy:", rfc.score(X_test, y_test))
circle_data_train = pd.read_csv("../input/circledataset/train.csv")



X_train = circle_data_train[['X','Y']]

y_train = circle_data_train['class']



plt.figure(figsize=(8,6))

sns.scatterplot(x="X", y="Y", data=circle_data_train, hue="class")

plt.show()



print(circle_data_train.head())

print(circle_data_train.shape)
circle_data_test = pd.read_csv('../input/circledataset/test.csv')



plt.figure(figsize=(8,6))

sns.scatterplot(x='X', y='Y', data=circle_data_test, hue="class")

plt.show()



X_test = circle_data_test[['X','Y']]

y_test = circle_data_test['class']



circle_data_test.head()

print(circle_data_test.shape)
rfc = RandomForestClassifier(n_estimators=250)

rfc.fit(X_train, y_train)



predictions = []



for i,row in X_test.iterrows():

    X=[[row['X'], row['Y']]]

    if rfc.predict(X) == y_test[i]:

        predictions.append(row)
df_pred = pd.DataFrame(predictions)



plt.figure(figsize=(8,6))

sns.scatterplot(x=circle_data_test['X'], y=circle_data_test['Y'],hue=circle_data_test['class'])

sns.scatterplot(x=df_pred['X'], y=df_pred['Y'])

plt.title("Testdaten")

plt.show()



print("Accuracy:", rfc.score(X_test, y_test))