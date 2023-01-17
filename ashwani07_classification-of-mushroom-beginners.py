# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Load the dataset

data = pd.read_csv("../input/mushrooms.csv")
#Check the data

data.head()
#Shape of the data

data.shape
data.info()
data.describe(include='all').transpose()
#dropping variable veil_type

data = data.drop(["veil-type"], axis = 1)
#seperating the dependant and independant variables.

features = data.columns

target = 'class'

features = list(features.drop(target))

features
# There are 21 variables, so a plot is divided into 11 rows and 2 columns.

fig, axs = plt.subplots(nrows=11, ncols=2, figsize=(11, 66))



for f, ax in zip(features, axs.ravel()):

    sns.countplot(x=f, hue='class', data=data, ax = ax)
#Converting categories to numbers (Label encoding)

from sklearn import preprocessing

labelEncoder = preprocessing.LabelEncoder()

for col in data.columns:

    data[col] = labelEncoder.fit_transform(data[col])
data.describe().transpose()
plt.figure(figsize=(14,12))

sns.heatmap(data.corr(),linewidths=.1,cmap="GnBu", annot=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.loc[:,features],data.loc[:,target],test_size=0.3,random_state=0)

print ('Train data set', X_train.shape)

print ('Test data set', X_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score



LR = LogisticRegression(random_state = 0)
# Using all features

LR.fit(X_train,y_train)
y_pred = LR.predict(X_test)
#Let's see how our model performed

print(classification_report(y_test, y_pred))
#Confusion matrix for the LR classification

print(confusion_matrix(y_test, y_pred))
#AUC for the LR classification

auc_roc=roc_auc_score(y_test,y_pred)

print ('AUC using LR %0.3f' % (auc_roc))
#Now lets try to do some evaluation for random forest model using cross validation.

LR_eval = cross_val_score(estimator = LR, X = X_train, y = y_train, cv = 10)

LR_eval.mean()