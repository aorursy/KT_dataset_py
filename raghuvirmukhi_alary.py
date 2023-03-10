import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('../input/salary/sal.csv',
                      names = ['age',
                               'workclass',
                               'fnlwgt',
                               'education',
                               'education-num',
                               'marital-status',
                               'occupation',
                               'relationship',
                               'race',
                               'gender',
                               'capital-gain',
                               'capital-loss',
                               'hours-per-week',
                               'native-country',
                               'salary'],
                               na_values = ' ?')
dataset.head(10)
dataset.shape
dataset.isnull().sum()

from sklearn.impute import SimpleImputer
imp = SimpleImputer()
X[:, [0, 2, 4, 10, 11, 12]] = imp.fit_transform(X[:, [0, 2, 4, 10, 11, 12]])
test = pd.DataFrame(X[:, [1, 3, 5, 6, 7, 8, 9, 13]])
test[0].value_counts()
test[1].value_counts()
test[2].value_counts()
test[3].value_counts()
test[4].value_counts()
test[5].value_counts()
test[6].value_counts()
test[7].value_counts()
test[0] = test[0].fillna(' Private')
test[0].value_counts()
test[1] = test[1].fillna(' HS-grad')
test[1].value_counts()

test[2] = test[2].fillna(' Married-civ-spouse')
test[2].value_counts()
test[3] = test[3].fillna(' Prof-specialty')
test[3].value_counts()
test[4] = test[4].fillna(' Husband')
test[4].value_counts()

test[5] = test[5].fillna(' White')
test[5].value_counts()
test[6] = test[6].fillna(' Male')
test[6].value_counts()
test[7] = test[7].fillna(' United-States')
test[7].value_counts()

X[:, [1, 3, 5, 6, 7, 8, 9, 13]] = test
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:, 1] = lab.fit_transform(X[:, 1].astype(str))
X[:, 3] = lab.fit_transform(X[:, 3].astype(str))
X[:, 5] = lab.fit_transform(X[:, 5].astype(str))
X[:, 6] = lab.fit_transform(X[:, 6].astype(str))
X[:, 7] = lab.fit_transform(X[:, 7].astype(str))
X[:, 8] = lab.fit_transform(X[:, 8].astype(str))
X[:, 9] = lab.fit_transform(X[:, 9].astype(str))
X[:, 13] = lab.fit_transform(X[:, 13].astype(str))
y = lab.fit_transform(y.astype(str))
lab.classes_
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        
         OneHotEncoder(), 
         [1]              
         )
    ],
    remainder='passthrough' 
)
X = transformer.fit_transform(X.tolist())
X = X.astype('float64')
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

log_reg.score(X, y)
from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor()
reg.fit(X,y)
reg.score(X,y)

