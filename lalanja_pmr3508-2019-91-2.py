# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

import seaborn as sns

from sklearn.preprocessing import StandardScaler

import sklearn.linear_model as linear_model

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
adult = pd.read_csv('../input/adult-income-dataset/adult.data',names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

nadult = adult.dropna()
adult.shape

adult.head()
test_adult = pd.read_csv('../input/adult-income-dataset/adult.test',names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

ntest_adult = test_adult.dropna()

ntest_adult.shape
del nadult["Education"]

del ntest_adult["Education"]
def number_encode_features(df):

    result = df.copy()

    encoders = {}

    for column in result.columns:

        if result.dtypes[column] == np.object:

            encoders[column] = preprocessing.LabelEncoder()

            result[column] = encoders[column].fit_transform(result[column])

    return result, encoders

encoded_data, encoders = number_encode_features(nadult)

train_encoded_data, train_encoders = number_encode_features(ntest_adult)

sns.heatmap(encoded_data.corr(), square=True)

plt.show()

encoded_data.corr()
X_train, y_train = encoded_data[["Age", "Workclass", "fnlwgt","Education-Num", "Martial Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country"]], encoded_data["Target"]

X_test, y_test = train_encoded_data[["Age", "Workclass", "fnlwgt","Education-Num", "Martial Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country"]], train_encoded_data["Target"]

scaler = preprocessing.StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

X_test = scaler.transform(X_test)
cls = linear_model.LogisticRegression()



cls.fit(X_train, y_train)

y_pred = cls.predict(X_test)

accuracy_score(y_test, y_pred)
cross_val_model = linear_model.LogisticRegression()

scores = cross_val_score(cross_val_model, X_train, y_train, cv=5)

print(np.mean(scores))
from sklearn.tree import DecisionTreeClassifier

ArX_train, Ary_train = encoded_data[["Age", "Workclass", "fnlwgt","Education-Num", "Martial Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country"]], encoded_data["Target"]

ArX_test, Ary_test = train_encoded_data[["Age", "Workclass", "fnlwgt","Education-Num", "Martial Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country"]], train_encoded_data["Target"]
clf = DecisionTreeClassifier(criterion="gini")

clf.fit(ArX_train,Ary_train)

y_pred = clf.predict(ArX_test)

accuracy_score(Ary_test, y_pred)
clf = DecisionTreeClassifier(criterion="gini",max_depth=9)

clf.fit(ArX_train,Ary_train)

y_pred = clf.predict(ArX_test)

accuracy_score(Ary_test, y_pred)
from sklearn.naive_bayes import GaussianNB

NBX_train, NBy_train = encoded_data[["Age", "Workclass", "fnlwgt","Education-Num", "Martial Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country"]], encoded_data["Target"]

NBX_test, NBy_test = train_encoded_data[["Age", "Workclass", "fnlwgt","Education-Num", "Martial Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country"]], train_encoded_data["Target"]

scaler = preprocessing.StandardScaler()

NBX_train = pd.DataFrame(scaler.fit_transform(NBX_train), columns=NBX_train.columns)

NBX_test = scaler.transform(NBX_test)
gnb = GaussianNB()

gnb.fit(NBX_train, NBy_train)

y_pred = gnb.predict(NBX_test)

accuracy_score(NBy_test, y_pred)


scores = cross_val_score(gnb, NBX_train, NBy_train, cv=5)

print(np.mean(scores))
import pandas as pd

test_data = pd.read_csv("../input/atividade-3-pmr3508/test.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

train_data = pd.read_csv("../input/atividade-3-pmr3508/train.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

test_data.head()



train_data.shape
train_data, test_data = train_data.dropna(), test_data.dropna()

X1_train, y1_train = train_data[['longitude','latitude','median_age','total_rooms','total_bedrooms', 'median_income','population']], train_data['median_house_value']

X1_test = test_data[['longitude','latitude','median_age','total_rooms','total_bedrooms', 'median_income','population']]

X1_train = pd.DataFrame(scaler.fit_transform(X1_train), columns=X1_train.columns)

X1_test = scaler.transform(X1_test)
from sklearn.linear_model import LinearRegression

clr = LinearRegression()

clr.fit(X1_train,y1_train)

scores = cross_val_score(clr, X1_train, y1_train, cv=5)

print(np.mean(scores))
knn = KNeighborsClassifier(n_neighbors=50)

knn.fit(X1_train,y1_train)

scores = cross_val_score(knn, X1_train, y1_train, cv=5)

print(np.mean(scores))
clf = DecisionTreeClassifier(criterion="gini",max_depth=9)

clf.fit(X1_train,y1_train)

scores = cross_val_score(clf, X1_train, y1_train, cv=5)

print(np.mean(scores))
cls.fit(X1_train,y1_train)

scores = cross_val_score(cls, X1_train, y1_train, cv=5)

print(np.mean(scores))