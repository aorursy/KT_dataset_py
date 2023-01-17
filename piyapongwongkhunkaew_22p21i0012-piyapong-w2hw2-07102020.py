# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Set your own project id here

PROJECT_ID = 'your-google-cloud-project'

from google.cloud import automl_v1beta1 as automl

automl_client = automl.AutoMlClient()

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)

from google.cloud import bigquery

bigquery_client = bigquery.Client(project=PROJECT_ID)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score



pd.set_option('display.max_rows', 1000)

import math

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc

from sklearn.model_selection import GridSearchCV,StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from collections import Counter

%pip install ppscore # installing ppscore, library used to check non-linear relationships between our variables

import ppscore as pps # importing ppscore

import string



seed =45



plt.style.use('fivethirtyeight')
data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

data_test = pd.read_csv("/kaggle/input/titanic/test.csv")

data_sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
data_train.shape
data_train.head()
data_train.describe()
data_train.info()
data_train.isnull().sum()
data_train.columns
# separate between numeric and categorical

num_df = data_train[['Age','SibSp','Parch','Fare']]

cate_df = data_train[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]
## Correlation between categorical factor

print(data_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print()

print(data_train[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print()

print(data_train[["Ticket", "Survived"]].groupby(['Ticket'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print()

print(data_train[["Cabin", "Survived"]].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print()

print(data_train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print()
W = data_train.loc[data_train.Sex == 'female']["Survived"]

Wsur = sum(W)/len(W)



print("women survived:", Wsur, "%")



M = data_train.loc[data_train.Sex == 'male']["Survived"]

Msur = sum(M)/len(M)



print("men survived:", Msur, "%")
#comparing survival and each of these categorical variables

print(pd.pivot_table(data_train ,index = 'Survived',columns = 'Pclass',values =  'Ticket',aggfunc = 'count'))

print()

print(pd.pivot_table(data_train , index = 'Survived', columns = 'Sex', values = 'Ticket' ,aggfunc ='count'))

print()

print(pd.pivot_table(data_train , index = 'Survived', columns = 'Embarked', values = 'Ticket' ,aggfunc ='count'))
for i in cate_df.columns:

    sns.barplot(cate_df[i].value_counts().index,cate_df[i].value_counts()).set_title(i)

    plt.show()
pd.pivot_table(data_train , index = 'Survived',values = ['Age','SibSp','Parch','Fare'])
# Correlation between numeric factor

fig = plt.figure(figsize=(10,10))

#sns.heatmap(num_df.corr(),cmap='RdYlGn', annot=True, linewidths=1)

sns.heatmap(data_train.corr(),cmap='RdYlGn', annot=True, linewidths=1)

print(num_df.corr())
# Age distribution

plt.figure(figsize=(14,8))

sns.distplot(data_train[(data_train['Age']>0)]['Age'], bins=50)



plt.title('Distribution of age passengers', fontsize=14)

plt.xlabel('Age', fontsize=14)

plt.ylabel('Frequency', fontsize=14)

plt.tight_layout()
# df after tuning

def basic_details(df):

    b = pd.DataFrame()

    b['Missing value'] = data_train.isnull().sum()

    b['N unique value'] = data_train.nunique()

    b['dtype'] = data_train.dtypes

    return b

basic_details(data_train)
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df_train.info(), df_test.info()
df_train = df_train[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
df_test = df_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
df_test.info(), df_train.info()
# One hot encoder

dfo_train = pd.get_dummies(df_train)

dfo_test = pd.get_dummies(df_test)
# df after tuning

def basic_details(df):

    b = pd.DataFrame()

    b['Missing value'] = dfo_train.isnull().sum()

    b['N unique value'] = dfo_train.nunique()

    b['dtype'] = dfo_train.dtypes

    return b

basic_details(dfo_train)
# df after tuning

def basic_details(df):

    b = pd.DataFrame()

    b['Missing value'] = dfo_test.isnull().sum()

    b['N unique value'] = dfo_test.nunique()

    b['dtype'] = dfo_test.dtypes

    return b

basic_details(dfo_test)
# Na Problem

import numpy as np

# 1.Drop

#dfo_train.dropna(axis=0)

#dfo_test.dropna(axis=0)



# 2.Fill

meanAge = np.mean(dfo_train.Age) # Get mean value

dfo_train.Age = dfo_train.Age.fillna(meanAge) # Fill missing values with mean

meanAge = np.mean(dfo_test.Age) # Get mean value

dfo_test.Age = dfo_test.Age.fillna(meanAge) # Fill missing values with mean



meanFare = np.mean(dfo_test.Fare) # Get mean value

dfo_test.Fare = dfo_test.Fare.fillna(meanFare) # Fill missing values with mean
# df after tuning

def basic_details(df):

    b = pd.DataFrame()

    b['Missing value'] = dfo_train.isnull().sum()

    b['N unique value'] = dfo_train.nunique()

    b['dtype'] = dfo_train.dtypes

    return b

basic_details(dfo_train)
# df after tuning

def basic_details(df):

    b = pd.DataFrame()

    b['Missing value'] = dfo_test.isnull().sum()

    b['N unique value'] = dfo_test.nunique()

    b['dtype'] = dfo_test.dtypes

    return b

basic_details(dfo_test)
split = len(data_train)

train = dfo_train[:split]

test = dfo_train[split:]
# X_train = dfo_train[['Age','SibSp','Parch','Fare','Pclass','Sex','Embarked']]

X_train = dfo_train.drop('Survived', axis=1)

y_train = dfo_train["Survived"]

#X_test = dfo_test[['Age','SibSp','Parch','Fare','Pclass','Sex','Embarked']]

X_test = dfo_test

X_train.shape, y_train.shape, X_test.shape
X_train.info(), X_test.info()
from sklearn import tree

from sklearn import metrics



# Decision_Tree



dec_t = tree.DecisionTreeClassifier()

y_pred_dec_t = dec_t.fit(X_train, y_train).predict(X_train)



dec_t_acc = metrics.accuracy_score(y_train, y_pred_dec_t)

dec_t_recall = metrics.recall_score(y_train, y_pred_dec_t)

dec_t_precision = metrics.precision_score(y_train, y_pred_dec_t)

dec_t_f1 = metrics.f1_score(y_train, y_pred_dec_t)



print('Accuracy    : {0:0.5f}'.format(dec_t_acc))

print('Recall      : {0:0.5f}'.format(dec_t_recall))

print('Precision   : {0:0.5f}'.format(dec_t_precision))

print('F-Measure   : {0:0.5f}'.format(dec_t_f1))
#Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred_gnb = gnb.predict(X_train)



gnb_acc = metrics.accuracy_score(y_train, y_pred_gnb)

gnb_recall = metrics.recall_score(y_train, y_pred_gnb)

gnb_precision = metrics.precision_score(y_train, y_pred_gnb)

gnb_f1 = metrics.f1_score(y_train, y_pred_gnb)



print('Accuracy    : {0:0.5f}'.format(gnb_acc))

print('Recall      : {0:0.5f}'.format(gnb_recall))

print('Precision   : {0:0.5f}'.format(gnb_precision))

print('F-Measure   : {0:0.5f}'.format(gnb_f1))
# multi-layer perceptron (MLP)

from sklearn.neural_network import MLPClassifier



mlpc = MLPClassifier(solver='lbfgs', alpha=1e-5,

                    hidden_layer_sizes=(20, ), random_state=1)

mlpc.fit(X_train, y_train)

y_pred_mlpc = mlpc.predict(X_train)



mlpc_acc = metrics.accuracy_score(y_train, y_pred_mlpc)

mlpc_recall = metrics.recall_score(y_train, y_pred_mlpc)

mlpc_precision = metrics.precision_score(y_train, y_pred_mlpc)

mlpc_f1 = metrics.f1_score(y_train, y_pred_mlpc)



print('Accuracy    : {0:0.5f}'.format(mlpc_acc))

print('Recall      : {0:0.5f}'.format(mlpc_recall))

print('Precision   : {0:0.5f}'.format(mlpc_precision))

print('F-Measure   : {0:0.5f}'.format(mlpc_f1))
import numpy as np

from sklearn.model_selection import KFold



kf = KFold(n_splits=5)

for train, test in kf.split(dfo_train):

    print("%s %s" % (train, test))
import numpy as np

from sklearn.model_selection import KFold

kf = KFold(n_splits=5)

kf.get_n_splits(X_train)



print(kf)
for train_index, test_index in kf.split(X_train):

    print("TRAIN:", train_index, "TEST:", test_index)

    #X_train, X_test = X_train[train_index], X_train[test_index]

    #y_train, y_test = y_train[train_index], y_train[test_index]

    X_train, X_test = X_train.values[train_index], X_train.values[test_index]

    y_train, y_test = y_train.values[train_index], y_train.values[test_index]
# Decision_Tree



dec_t = tree.DecisionTreeClassifier()

y_pred_dec_t = dec_t.fit(X_train, y_train).predict(X_test)



dec_t_acc = metrics.accuracy_score(y_test, y_pred_dec_t)

dec_t_recall = metrics.recall_score(y_test, y_pred_dec_t)

dec_t_precision = metrics.precision_score(y_test, y_pred_dec_t)

dec_t_f1 = metrics.f1_score(y_test, y_pred_dec_t)



print('Accuracy    : {0:0.5f}'.format(dec_t_acc))

print('Recall      : {0:0.5f}'.format(dec_t_recall))

print('Precision   : {0:0.5f}'.format(dec_t_precision))

print('F-Measure   : {0:0.5f}'.format(dec_t_f1))
#Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred_gnb = gnb.predict(X_test)



gnb_acc = metrics.accuracy_score(y_test, y_pred_gnb)

gnb_recall = metrics.recall_score(y_test, y_pred_gnb)

gnb_precision = metrics.precision_score(y_test, y_pred_gnb)

gnb_f1 = metrics.f1_score(y_test, y_pred_gnb)



print('Accuracy    : {0:0.5f}'.format(gnb_acc))

print('Recall      : {0:0.5f}'.format(gnb_recall))

print('Precision   : {0:0.5f}'.format(gnb_precision))

print('F-Measure   : {0:0.5f}'.format(gnb_f1))
# multi-layer perceptron (MLP)

from sklearn.neural_network import MLPClassifier



mlpc = MLPClassifier(solver='lbfgs', alpha=1e-5,

                    hidden_layer_sizes=(20, ), random_state=1)

mlpc.fit(X_train, y_train)

y_pred_mlpc = mlpc.predict(X_test)



mlpc_acc = metrics.accuracy_score(y_test, y_pred_mlpc)

mlpc_recall = metrics.recall_score(y_test, y_pred_mlpc)

mlpc_precision = metrics.precision_score(y_test, y_pred_mlpc)

mlpc_f1 = metrics.f1_score(y_test, y_pred_mlpc)



print('Accuracy    : {0:0.5f}'.format(mlpc_acc))

print('Recall      : {0:0.5f}'.format(mlpc_recall))

print('Precision   : {0:0.5f}'.format(mlpc_precision))

print('F-Measure   : {0:0.5f}'.format(mlpc_f1))