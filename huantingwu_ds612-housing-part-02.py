import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn_pandas import DataFrameMapper

from helpers import *

import warnings

warnings.filterwarnings("ignore")

%load_ext autoreload

%autoreload 2
train_raw = pd.read_csv("/kaggle/input/sfsu-ds612-housing-dataset-project/housing_data_train.csv")

train_raw = train_raw.set_index("Id")

train_raw = train_raw.copy()
train_raw.isnull().sum().sort_values(ascending=False)[:20]
train_raw['SalePrice'].isnull().sum()
train_raw = train_raw.fillna(0.0)
numerical, categorical = columns()

train_raw[categorical] = train_raw[categorical].astype('str')

train_raw[numerical] = train_raw[numerical].astype('float')
train_raw['OverallCond'].value_counts()
train_raw['OverallCond'].value_counts() / train_raw['OverallCond'].size * 100
train_x_raw, train_y = train_raw.drop("OverallCond", axis=1), train_raw['OverallCond']
print('train_x_raw shape:', train_x_raw.shape)

print('train_y_raw shape:', train_y.shape)
train_x_raw.head()
train_y.head()
distinct_y = train_y.unique()
y_value_count = train_y.value_counts()

print("y_value_count:\n", y_value_count)
feature_class = pd.read_csv('/kaggle/input/sfsu-ds612-housing-dataset-project/feature_class.csv', names = ['feature', 'class'])
feature_class['feature'] = feature_class['feature'].str.strip()

feature_class['class'] = feature_class['class'].str.strip()
numerical = []

nominal = []

ordinal = []
for i in train_x_raw.columns:

    if feature_class['class'][feature_class['feature']==i].values == 'Numerical': 

        numerical.append(i)

    elif feature_class['class'][feature_class['feature']==i].values == "Nominal":

        nominal.append(i)

    elif feature_class['class'][feature_class['feature']==i].values == "Ordinal":

        ordinal.append(i)
print('numerical:', numerical)

print('nominal:', nominal)

print('ordinal:', ordinal)
train_x_raw[numerical].skew().sort_values(ascending=False)
train_x_raw[numerical].skew().mean()
mapper = DataFrameMapper([

    (numerical, preprocessing.StandardScaler()),

    (nominal, preprocessing.OneHotEncoder(sparse=False)),

    (ordinal, preprocessing.OrdinalEncoder())], df_out=True)
train_x_normalized = mapper.fit_transform(train_x_raw)
train_x, dev_x, train_y, dev_y = train_test_split(train_x_normalized, train_y, random_state=42)
print('train_x shape:', train_x.shape)

print('dev_x shape:', dev_x.shape)

print('train_y shape:', train_y.shape)

print('dev_y shape:', dev_y.shape)
from sklearn.linear_model import LogisticRegression



logistic_regression = LogisticRegression(random_state=42).fit(train_x, train_y)



measure(logistic_regression, dev_x, dev_y)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



lda = LinearDiscriminantAnalysis().fit(train_x, train_y)



measure(lda, dev_x, dev_y)
from sklearn.tree import DecisionTreeClassifier



decision_tree = DecisionTreeClassifier(random_state=42).fit(train_x, train_y)



measure(decision_tree, dev_x, dev_y)
from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier(random_state=42).fit(train_x, train_y)



measure(random_forest, dev_x, dev_y)
from sklearn.ensemble import GradientBoostingClassifier



gradient_boost = GradientBoostingClassifier(random_state=42).fit(train_x, train_y)



measure(gradient_boost, dev_x, dev_y)