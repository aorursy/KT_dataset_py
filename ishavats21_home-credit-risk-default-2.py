# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#loading the train data

train=pd.read_csv("../input/application_train.csv")

print('Size of application_train data', train.shape)
#loading the test data

test=pd.read_csv("../input/application_test.csv")

print('Size of application_test data', test.shape)
#Distribution of target variables

train['TARGET'].value_counts()

train['TARGET'].astype(int).plot.hist();
#Finding missing values

total = train.isnull().sum().sort_values(ascending = False)

percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)

missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_train_data.head(20)
total = test.isnull().sum().sort_values(ascending = False)

percent = (test.isnull().sum()/test.isnull().count()*100).sort_values(ascending = False)

missing_test_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_test_data.head(20)
train.dtypes.value_counts()

train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
# Create a label encoder object

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le_count = 0



# Iterate through the columns

for col in train:

    if train[col].dtype == 'object':

        # If 2 or fewer unique categories

        if len(list(train[col].unique())) <= 2:

            # Train on the training data

            le.fit(train[col])

            # Transform both training and testing data

            train[col] = le.transform(train[col])

            test[col] = le.transform(test[col])

            

            # Keep track of how many columns were label encoded

            le_count += 1

            

print('%d columns were label encoded.' % le_count)
# one-hot encoding of categorical variables

train = pd.get_dummies(train)

test = pd.get_dummies(test)



print('Training Features shape: ', train.shape)

print('Testing Features shape: ', test.shape)
train_labels = train['TARGET']



# Align the training and testing data, keep only columns present in both dataframes

train, test = train.align(test, join = 'inner', axis = 1)



# Add the target back in

train['TARGET'] = train_labels



print('Training Features shape: ', train.shape)

print('Testing Features shape: ', test.shape)
# Find correlations with the target and sort

correlations = train.corr()['TARGET'].sort_values()



# Display correlations

print('Most Positive Correlations:\n', correlations.tail(15))

print('\nMost Negative Correlations:\n', correlations.head(15))
from sklearn.preprocessing import MinMaxScaler, Imputer



# Drop the target from the training data

if 'TARGET' in train:

    train_data = train.drop(columns = ['TARGET'])

else:

    train_data = train.copy()

    

# Feature names

features = list(train_data.columns)



# Copy of the testing data

test_data = test.copy()



# Median imputation of missing values

imputer = Imputer(strategy = 'median')



# Scale each feature to 0-1

scaler = MinMaxScaler(feature_range = (0, 1))



# Fit on the training data

imputer.fit(train_data)



# Transform both training and testing data

train_data = imputer.transform(train_data)

test_data = imputer.transform(test)

# Repeat with the scaler

scaler.fit(train_data)

train_data = scaler.transform(train_data)

test_data = scaler.transform(test_data)



print('Training data shape: ', train_data.shape)

print('Testing data shape: ', test_data.shape)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(train_data,train_labels,test_size=0.25,random_state=0)
#Building the logistice regression model

from sklearn.linear_model import LogisticRegression



# instantiate the model (using the default parameters)

logreg = LogisticRegression()



# fit the model with data

logreg.fit(X_train,y_train)



#

y_pred=logreg.predict(X_test)
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix
#Analysis of confusion matrix

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
#ROC curve

y_pred_proba = logreg.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
submit = test[['SK_ID_CURR']]

submit['TARGET'] = y_pred



submit.head()