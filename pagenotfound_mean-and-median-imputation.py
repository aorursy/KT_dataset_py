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
import pandas as pd

import numpy as np



import matplotlib.pyplot as plt





# for regression problems

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor



# for classification

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC



# to split and standarize the datasets

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# to evaluate regression models

from sklearn.metrics import mean_squared_error



# to evaluate classification models

from sklearn.metrics import roc_auc_score



import warnings

warnings.filterwarnings('ignore')
# load the Titanic Dataset with a few variables for demonstration



data = pd.read_csv('/kaggle/input/titanic/train.csv', usecols = ['Age', 'Fare','Survived'])

data.head()
# let's look at the percentage of NA



data.isnull().mean()
# let's separate into training and testing set



X_train, X_test, y_train, y_test = train_test_split(data, data.Survived, test_size=0.3,

                                                    random_state=0)

X_train.shape, X_test.shape
# let's make a function to create 2 variables from Age:

# one filling NA with median, and another one filling NA with zeroes



def impute_na(df, variable, median):

    df[variable+'_median'] = df[variable].fillna(median)

    df[variable+'_zero'] = df[variable].fillna(0)
median = X_train.Age.median()

median
impute_na(X_train, 'Age', median)

X_train.head(15)
impute_na(X_test, 'Age', median)
# we can see a change in the variance after imputation



print('Original Variance: ', X_train['Age'].std())

print('Variance after median imputation: ', X_train['Age_median'].std())
# we can see that the distribution has changed slightly with now more values accumulating towards the median

fig = plt.figure()

ax = fig.add_subplot(111)

X_train['Age'].plot(kind='kde', ax=ax)

X_train.Age_median.plot(kind='kde', ax=ax, color='red')

lines, labels = ax.get_legend_handles_labels()

ax.legend(lines, labels, loc='best')
# filling NA with zeroes creates a peak of population around 0, as expected

fig = plt.figure()

ax = fig.add_subplot(111)

X_train['Age'].plot(kind='kde', ax=ax)

X_train.Age_zero.plot(kind='kde', ax=ax, color='red')

lines, labels = ax.get_legend_handles_labels()

ax.legend(lines, labels, loc='best')
# Let's compare the performance of Logistic Regression using Age filled with zeros or alternatively the median



# model on NA imputed with zeroes

logit = LogisticRegression(random_state=44, C=1000) # c big to avoid regularization

logit.fit(X_train[['Age_zero', 'Fare']], y_train)

print('Train set zero imputation')

pred = logit.predict_proba(X_train[['Age_zero', 'Fare']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set zero imputation')

pred = logit.predict_proba(X_test[['Age_zero', 'Fare']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

print()



# model on NA imputed with median

logit = LogisticRegression(random_state=44, C=1000) # c big to avoid regularization

logit.fit(X_train[['Age_median', 'Fare']], y_train)

print('Train set median imputation')

pred = logit.predict_proba(X_train[['Age_median', 'Fare']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set median imputation')

pred = logit.predict_proba(X_test[['Age_median', 'Fare']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
print('Average total survival:', X_train.Survived.mean())
print('Average real survival of children: ', X_train[X_train.Age<15].Survived.mean())

print('Average survival of children when using Age imputed with zeroes: ', X_train[X_train.Age_zero<15].Survived.mean())

print('Average survival of children when using Age imputed with median: ', X_train[X_train.Age_median<15].Survived.mean())
# Let's compare the performance of SVM using Age filled with zeros or alternatively the median



SVM_model = SVC(random_state=44, probability=True, max_iter=-1, kernel='linear',)

SVM_model.fit(X_train[['Age_zero', 'Fare']], y_train)

print('Train set zero imputation')

pred = SVM_model.predict_proba(X_train[['Age_zero', 'Fare']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set zero imputation')

pred = SVM_model.predict_proba(X_test[['Age_zero', 'Fare']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

print()

SVM_model = SVC(random_state=44, probability=True,  max_iter=-1, kernel='linear')

SVM_model.fit(X_train[['Age_median', 'Fare']], y_train)

print('Train set median imputation')

pred = SVM_model.predict_proba(X_train[['Age_median', 'Fare']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set median imputation')

pred = SVM_model.predict_proba(X_test[['Age_median', 'Fare']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

print()
# Let's compare the performance of Random Forests using Age filled with zeros or alternatively the median



rf = RandomForestClassifier(n_estimators=100, random_state=39, max_depth=3)

rf.fit(X_train[['Age_zero', 'Fare']], y_train)

print('Train set zero imputation')

pred = rf.predict_proba(X_train[['Age_zero', 'Fare']])

print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set zero imputation')

pred = rf.predict_proba(X_test[['Age_zero', 'Fare']])

print('Random Forests zero imputation roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

print()

rf = RandomForestClassifier(n_estimators=100, random_state=39, max_depth=3)

rf.fit(X_train[['Age_median', 'Fare']], y_train)

print('Train set median imputation')

pred = rf.predict_proba(X_train[['Age_median', 'Fare']])

print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set median imputation')

pred = rf.predict_proba(X_test[['Age_median', 'Fare']])

print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

print()