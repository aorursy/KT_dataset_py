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
df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
import pandas as pd

import numpy as np

import random

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from scipy import stats

from scipy.stats import norm,skew
df.head()
chi_list = ['gender','SeniorCitizen','Partner','Dependents','PhoneService',

            'MultipleLines','InternetService','OnlineSecurity','OnlineBackup',

            'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']
sns.set(style="ticks", color_codes=True)



fig, axes = plt.subplots(nrows = 3,ncols = 5,figsize = (25,15))

sns.countplot(x = "gender", data = df, ax=axes[0][0])

sns.countplot(x = "Partner", data = df, ax=axes[0][1])

sns.countplot(x = "Dependents", data = df, ax=axes[0][2])

sns.countplot(x = "PhoneService", data = df, ax=axes[0][3])

sns.countplot(x = "MultipleLines", data = df, ax=axes[0][4])

sns.countplot(x = "InternetService", data = df, ax=axes[1][0])

sns.countplot(x = "OnlineSecurity", data = df, ax=axes[1][1])

sns.countplot(x = "OnlineBackup", data = df, ax=axes[1][2])

sns.countplot(x = "DeviceProtection", data = df, ax=axes[1][3])

sns.countplot(x = "TechSupport", data = df, ax=axes[1][4])

sns.countplot(x = "StreamingTV", data = df, ax=axes[2][0])

sns.countplot(x = "StreamingMovies", data = df, ax=axes[2][1])

sns.countplot(x = "Contract", data = df, ax=axes[2][2])

sns.countplot(x = "PaperlessBilling", data = df, ax=axes[2][3])

ax = sns.countplot(x = "PaymentMethod", data = df, ax=axes[2][4])

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.show(fig)
df.drop('customerID',axis=1,inplace=True)
for i in chi_list:

    print('-------------*******------------')

    dataset_table=pd.crosstab(df[i],df['Churn'])

    print(dataset_table)

    stat, p, dof, expected = stats.chi2_contingency(dataset_table)

    prob = 0.95

    critical = stats.chi2.ppf(prob, dof)

    if abs(stat) >= critical:

	    print('Dependent (reject H0)')

    else:

	    print('Independent (fail to reject H0)')

     

    alpha = 1.0 - prob

    if p <= alpha:

	    print('Dependent (reject H0)')

    else:

	    print('Independent (fail to reject H0)') 
chi_list.remove('PhoneService')

chi_list.append('gender')
df.drop(['gender','PhoneService'],axis=1,inplace=True)
df.dtypes
df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
from sklearn.model_selection import train_test_split
features = df.drop('Churn',axis=1)

labels = df['Churn']
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.2,random_state=42)
from sklearn.preprocessing import LabelEncoder
labelencoder = ['Partner','Dependents','OnlineSecurity','DeviceProtection','OnlineBackup','TechSupport','StreamingTV',

                'StreamingTV','StreamingMovies','PaperlessBilling','MultipleLines','Contract']
le = LabelEncoder()

for cols in labelencoder:

    X_train[cols] = le.fit_transform(X_train[cols])

    X_test[cols] = le.transform(X_test[cols])
X_train.head()
X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)
X_train['TotalCharges'] = X_train.TotalCharges.fillna(X_train.TotalCharges.mean())

X_test['TotalCharges'] = X_test.TotalCharges.fillna(X_test.TotalCharges.mean())
X_train['TotalCharges'] = X_train.TotalCharges.fillna(X_train.TotalCharges.mean())

X_test['TotalCharges'] = X_test.TotalCharges.fillna(X_test.TotalCharges.mean())
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)

X_test = ss.transform(X_test)
X_train
X_test
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

from catboost import CatBoostClassifier
!pip install catboost
classifiers = [['DecisionTree :',DecisionTreeClassifier()],

               ['RandomForest :',RandomForestClassifier()], 

               ['Naive Bayes :', GaussianNB()],

               ['KNeighbours :', KNeighborsClassifier()],

               ['SVM :', SVC()],

               ['Neural Network :', MLPClassifier()],

               ['LogisticRegression :', LogisticRegression()],

               ['ExtraTreesClassifier :', ExtraTreesClassifier()],

               ['AdaBoostClassifier :', AdaBoostClassifier()],

               ['GradientBoostingClassifier: ', GradientBoostingClassifier()],

               ['XGB :', XGBClassifier()],

               ['CatBoost :', CatBoostClassifier(logging_level='Silent')]]
for name,classifier in classifiers:

    classifier = classifier

    classifier.fit(X_train,y_train)

    predication = classifier.predict(X_test)

    accuracy = accuracy_score(y_test,predication)

    roc_score = roc_auc_score(y_test,predication)

    print('classifier name:',name,end='')

    print('accuracy:',accuracy,end='')

    print('roc auc score:',roc_score)

    print('----------------------')
from sklearn.ensemble import VotingClassifier

clf1 = GradientBoostingClassifier()

clf2 = LogisticRegression()

clf3 = XGBClassifier()

clf4 = RandomForestClassifier()



eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

eclf1.fit(X_train, y_train)

predictions = eclf1.predict(X_test)

print(accuracy_score(y_test, predictions))