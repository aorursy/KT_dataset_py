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
data = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data
data.isnull()
data.isnull().sum()
import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt
data.head()
print('Rows = ', data.shape[0])

print('Columns = ', data.shape[1])

print('\nMissing values =\n', data.isnull().sum())

print('\nName of columns =\n', data.columns)

print('\nUnique values =\n', data.nunique())
data['gender'] = data['gender'].map({'Male':1, 'Female':0})

data.head()
data['Partner'] = data['Partner'].map({'Yes':1, 'No':0})

data['Dependents'] = data['Dependents'].map({'Yes':1, 'No':0})

data['PhoneService'] = data['PhoneService'].map({'Yes':1, 'No':0})

data['PaperlessBilling'] = data['PaperlessBilling'].map({'Yes':1, 'No':0})

data['MultipleLines'] = data['MultipleLines'].map({'Yes':1, 'No':0, 'No phone service':0})

data['Churn'] = data['Churn'].map({'Yes':1, 'No':0})

pdinternetservice = pd.get_dummies(data['InternetService'])

data = pd.concat([data, pdinternetservice], axis = 1)

pdcontract = pd.get_dummies(data['Contract'])

data = pd.concat([data, pdcontract], axis = 1)

pdpaymentmethod  = pd.get_dummies(data['PaymentMethod'])

data = pd.concat([data, pdpaymentmethod], axis = 1)

data.head()
data.rename(columns ={'No':'No internet service'}, inplace = True)
data.Contract.unique()
data.drop('customerID', inplace = True, axis =1)

data.drop('OnlineSecurity', inplace = True, axis =1)

data.drop('OnlineBackup', inplace = True, axis =1)

data.drop('DeviceProtection', inplace = True, axis =1)

data.drop('TechSupport', inplace = True, axis =1)

data.drop('StreamingTV', inplace = True, axis =1)

data.drop('StreamingMovies', inplace = True, axis =1)

data.drop('Contract', inplace = True, axis =1)

data.drop('InternetService', inplace = True, axis =1)

data.drop('PaymentMethod', inplace = True, axis =1)

data.head()
target = data['Churn']

target.head()
data.drop('Churn', inplace = True, axis =1)
data.dtypes.sample(20)
data
data['TotalCharges'] = data["TotalCharges"].replace(" ",np.nan)

data['TotalCharges'] = data.TotalCharges.median()

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])
data.TotalCharges.isnull().sum()
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

def compute_score(clf, X, y, scoring='accuracy'):

    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)

    return np.mean(xval)

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')

clf = clf.fit(data, target)

features = pd.DataFrame()

features['feature'] = data.columns

features['importance'] = clf.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)
features.importance
features.plot(kind='barh', figsize=(25, 25))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data,target,random_state=50)
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

rfc = RandomForestClassifier(n_estimators=100)

dtc = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=3) 

svm = SVC(kernel = 'linear')

lr = LogisticRegression()

models = [rfc, dtc, svm, lr]

for model in models:

    model.fit(X_train, y_train)
for model in models:

    model.predict(X_test)
for model in models:

    print("Accuracy of", model)

    print("=", model.score(X_test, y_test))
drop_gender = data['gender']

data.drop("gender", inplace = True, axis = 1)
rfc_gender = RandomForestClassifier(n_estimators=100)

dtc_gender = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=3)  

svm_gender = SVC(kernel = 'linear')

lr_gender = LogisticRegression()

models_gender = [rfc_gender, dtc_gender, svm_gender, lr_gender]

for model in models_gender:

    model.fit(X_train, y_train)
for model in models_gender:

    model.predict(X_test)
for model in models_gender:

    print("Accuracy of", model)

    print("=", model.score(X_test, y_test))
data = pd.concat([data, drop_gender], axis = 1)

drop_dependents = data['Dependents']

data.drop("Dependents", inplace = True, axis = 1)
rfc_dependents = RandomForestClassifier(n_estimators=100)

dtc_dependents = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=3)   

svm_dependents = SVC(kernel = 'linear')

lr_dependents = LogisticRegression()

models_dependents = [rfc_dependents, dtc_dependents, svm_dependents, lr_dependents]

for model in models_dependents:

    model.fit(X_train, y_train)
for model in models_dependents:

    model.predict(X_test)
for model in models_dependents:

    print("Accuracy of", model)

    print("=", model.score(X_test, y_test))
data = pd.concat([data, drop_dependents], axis = 1)

drop_paperlessbilling = data['PaperlessBilling']

data.drop("PaperlessBilling", inplace = True, axis = 1)
rfc_paperlessbilling = RandomForestClassifier(n_estimators=100)

dtc_paperlessbilling = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=3)   

svm_paperlessbilling = SVC(kernel = 'linear')

lr_paperlessbilling = LogisticRegression()

models_paperlessbilling = [rfc_paperlessbilling, dtc_paperlessbilling, svm_paperlessbilling, lr_paperlessbilling]

for model in models_paperlessbilling:

    model.fit(X_train, y_train)
for model in models_paperlessbilling:

    model.predict(X_test)
for model in models_paperlessbilling:

    print("Accuracy of", model)

    print("=", model.score(X_test, y_test))