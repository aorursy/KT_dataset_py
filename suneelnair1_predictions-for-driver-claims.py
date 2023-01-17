# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")

train_data.describe

test_data = pd.read_csv("../input/test.csv")

test_data.describe
print(train_data.shape)

train_data.head()

train_data.isnull().any()

train_data.isnull().any(axis = 1)
train_data.dtypes
# X = train_data.iloc[:,:-2]

X = train_data.drop(train_data.columns[[0,1]],axis = 1)

train_data['target'].astype('category')

Y = train_data.target.values
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.30, random_state = 25)
from sklearn import preprocessing

std = preprocessing.StandardScaler()

std.fit(X_train)

X_train = std.transform(X_train)

X_test = std.transform(X_test)



# print(X_train)



# train_data.dtypes

# train_data['target'].astype('category')
from sklearn.linear_model import LogisticRegression

model_logit = LogisticRegression()

model_logit.fit(X_train,Y_train)
train_predictions = model_logit.predict(X_train)

test_predictions = model_logit.predict(X_test)

print(train_predictions)

print(test_predictions)

# print(Y_test)
from sklearn import metrics

from sklearn.metrics import confusion_matrix

mat1 = confusion_matrix(Y_test,test_predictions)

mat1

# print(Y_test)
from sklearn.metrics import classification_report

print(classification_report(Y_test,test_predictions))
print("Accuracy for the model: ",metrics.accuracy_score(Y_test,test_predictions))

print("Precision for the model: ",metrics.precision_score(Y_test,test_predictions))

print("Recall for the model: ", metrics.recall_score(Y_test,test_predictions))
# Plots

import seaborn as sns

# plt.plot(test_predictions,Y_test,color = 'blue',Title = 'Test predictions',linewidth = 2.0)

# plt.plot(train_predictions,Y_train,color = 'red',Title = 'Train predictions',linewidth = 2.0)



sns.violinplot(x="ps_ind_01", y = "target", data = train_data)

# sns.barplot





plt.scatter(train_data["ps_ind_01"],train_data["ps_car_01_cat"])

plt.show()
plt.plot(Y_test,test_predictions,color='r')



# plt.xlabel('Y_test')

# plt.ylabel('Predictions')

# plt.title('Predictions to Actuals')



plt.show()

print("Accuracy: ",model_logit.score(X,Y))