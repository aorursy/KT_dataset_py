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
#the usual

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt



#colored printing output

from termcolor import colored



#I/O

import io

import os

import requests



#pickle

import pickle



#math

import math



#scipy

from scipy import stats



#sk learn

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn import linear_model

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import preprocessing

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import validation_curve

from sklearn.model_selection import learning_curve

from itertools import combinations

from mlxtend.feature_selection import ColumnSelector

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



#sns style

import seaborn as sns

#sns.set_style("whitegrid")

sns.despine()

sns.set_context("talk") #larger display of plots axis labels etc..

sns.set(style='darkgrid')
bank_data = pd.read_csv("../input/Bank_Customers.csv",index_col ='RowNumber')

bank_data.head()
bank_data.shape
bank_data.hist(figsize=(15,12),bins=15,color="purple",grid=False,);
sns.pairplot(bank_data,hue="Exited",height=2)
fig = plt.figure(figsize = (15,15)); ax = fig.gca()

sns.heatmap(bank_data.corr(), annot = True, vmin= -1, vmax = 1, ax=ax)


bank_data["Gender"].value_counts().plot(kind='pie',figsize= (6,6));

def bar_chart(feature,input_df):

    Exited = input_df[input_df['Exited']==1][feature].value_counts()

    Stayed = input_df[input_df['Exited']==0][feature].value_counts()

    df = pd.DataFrame([Exited,Stayed])

    df.index = ['Exited','Stayed']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart("Geography",bank_data)
bar_chart("IsActiveMember",bank_data)
bank_data.head()
bank_data.describe()
Bank_data = bank_data.drop(['CustomerId','Surname'],axis=1)

Bank_data.head()
Bank_data.shape
# https://www.kaggle.com/shrutimechlearn/types-of-regression-and-stats-in-depth

Geo_bank = pd.get_dummies(prefix='Geo',data=Bank_data,columns=['Geography'])

Geo_bank.head()
Gen_bank = Geo_bank.replace(to_replace={'Gender': {'Female': 1,'Male':0}})

Gen_bank.head()
churn_bank = Gen_bank
X = churn_bank.drop(['Exited'],axis=1)

y = churn_bank.Exited
print(X.shape)

print(y.shape)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 0)
# Feature Scaling because yes we don't want one independent variable dominating the other and it makes computations easy

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# sequential model to initialise our ann and dense module to build the layers

from keras.models import Sequential

from keras.layers import Dense
classifier = Sequential()

# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))



# Adding the second hidden layer

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN | means applying SGD on the whole ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100,verbose = 0)



score, acc = classifier.evaluate(X_train, y_train,

                            batch_size=10)

print('Train score:', score)

print('Train accuracy:', acc)

# Part 3 - Making predictions and evaluating the model



# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)



print('*'*20)

score, acc = classifier.evaluate(X_test, y_test,

                            batch_size=10)

print('Test score:', score)

print('Test accuracy:', acc)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
#import classification_report

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
from sklearn.metrics import roc_curve

y_pred_proba = classifier.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='ANN')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC curve')

plt.show()
#Area under ROC curve

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test,y_pred_proba)
# print('Best Parameters after tuning: {}'.format(best_parameters))

# print('Best Accuracy after tuning: {}'.format(best_accuracy))