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
import pandas as pd

higgs = pd.read_csv('/kaggle/input/higgs-boson/training.zip',index_col=0)

higgs_test = pd.read_csv('/kaggle/input/higgs-boson/test.zip',index_col=0)
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()



higgs['Label'] = enc.fit_transform(higgs['Label'])

higgs.head()
# X and y

X = higgs.drop(['Label','Weight'],axis=1)

y = higgs['Label']

X
# checking missing values

from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=-999, strategy='median')

higgs_imputed = imp.fit_transform(X)

higgs_imputed
# standardize training data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_standard = scaler.fit_transform(X)

X_standard
# split training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_standard, y, test_size=0.33, random_state=42)
# decisionTreeModel

from sklearn.tree import DecisionTreeClassifier

Decision_Tree_Model = DecisionTreeClassifier()

Decision_Tree_Model.fit(X_train,y_train)

from sklearn.metrics import f1_score

print('Decision_Tree_Model Train Score is   : ' ,Decision_Tree_Model.score(X_train, y_train))

print('Decision_Tree_Model Train Score is   : ' ,Decision_Tree_Model.score(X_test, y_test))

y_pred = Decision_Tree_Model.predict(X_test)

print('Decision_Tree_Model Train Score is   : ' ,f1_score(y_test, y_pred))
# linear discriminant analysis

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

Linear_Discriminant_Analysis = LinearDiscriminantAnalysis()

Linear_Discriminant_Analysis.fit(X_train, y_train)

from sklearn.metrics import f1_score

print('Linear_Discriminant_Analysis Train Score is   : ' ,Linear_Discriminant_Analysis.score(X_train, y_train))

print('Linear_Discriminant_Analysis Train Score is   : ' ,Linear_Discriminant_Analysis.score(X_test, y_test))

y_pred = Linear_Discriminant_Analysis.predict(X_test)

print('Linear_Discriminant_Analysis Train Score is   : ' ,f1_score(y_test, y_pred))
# quadratic discriminant analysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

Quadratic_Discriminant_Analysis = QuadraticDiscriminantAnalysis()

Quadratic_Discriminant_Analysis.fit(X_train, y_train)

print('Quadratic_Discriminant_Analysis Train Score is   : ' ,Quadratic_Discriminant_Analysis.score(X_train, y_train))

print('Quadratic_Discriminant_Analysis Train Score is   : ' ,Quadratic_Discriminant_Analysis.score(X_test, y_test))

y_pred = Quadratic_Discriminant_Analysis.predict(X_test)

print('Quadratic_Discriminant_Analysis Train Score is   : ' ,f1_score(y_test, y_pred))
# Gaussian Naive Bayes Model

from sklearn.naive_bayes import GaussianNB

GaussianNBModel = GaussianNB()

GaussianNBModel.fit(X_train,y_train)

print('GaussianNBModel Train Score is   : ' ,GaussianNBModel.score(X_train, y_train))

print('GaussianNBModel Train Score is   : ' ,GaussianNBModel.score(X_test, y_test))

y_pred = GaussianNBModel.predict(X_test)

print('GaussianNBModel Train Score is   : ' ,f1_score(y_test, y_pred))
# BernoulliNBModel

from sklearn.naive_bayes import BernoulliNB

BernoulliNBModel = BernoulliNB()

BernoulliNBModel.fit(X_train, y_train)

print('BernoulliNBModel Train Score is   : ' ,BernoulliNBModel.score(X_train, y_train))

print('BernoulliNBModel Train Score is   : ' ,BernoulliNBModel.score(X_test, y_test))

y_pred = BernoulliNBModel.predict(X_test)

print('BernoulliNBModel Train Score is   : ' ,f1_score(y_test, y_pred))
# logistic regression

from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()

logistic_regression.fit(X_train, y_train)

print('logistic_regression Train Score is   : ' ,logistic_regression.score(X_train, y_train))

print('logistic_regression Train Score is   : ' ,logistic_regression.score(X_test, y_test))

y_pred = logistic_regression.predict(X_test)

print('logistic_regression Train Score is   : ' ,f1_score(y_test, y_pred))