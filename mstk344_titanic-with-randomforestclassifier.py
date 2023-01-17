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
from sklearn.linear_model import RidgeCV

from sklearn.decomposition import PCA

from sklearn.preprocessing import (

    StandardScaler,

    MinMaxScaler,

    PolynomialFeatures,

    LabelEncoder,

    OneHotEncoder,

    Imputer)

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

EXOG = ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked')

ENDOG = 'Survived'
class MultiLabelEncoder(object):

    def __init__(self, columns):

        self.columns = np.array(columns)

        self.dtype = self.columns.dtype

        self.le = {col: LabelEncoder() for col in np.array(columns)}

    

    def fit(self, X, y=None):

        if self.dtype in (np.dtype(int), np.dtype(float)):

            for k in self.le.keys():

                self.le[k].fit(X.iloc[:, k].values)

        else:

            for k in self.le.keys():

                self.le[k].fit(X.loc[:, k].values)

        return self

    

    def transform(self, X):

        output = X.copy()

        if self.dtype in (np.dtype(int), np.dtype(float)):

            for k in self.le.keys():

                output.iloc[:, k] = self.le[k].transform(

                    output.iloc[:, k].values)

        else:

            for k in self.le.keys():

                output.loc[:, k] = self.le[k].transform(

                    output.loc[:, k].values)

        return output

    

    def fit_transform(self, X, y=None):

        return self.fit(X, y).transform(X)
def predict_age(X_train, y_train, X_test):

    exog = list(EXOG)

    exog.remove('Age')

    endog = 'Age'

    pipe = Pipeline(

        [('ohe', OneHotEncoder(categorical_features=[5, 6],

                               sparse=False)),

         ('scl', MinMaxScaler()),

         ('poly', PolynomialFeatures(degree=2,

                                     interaction_only=True)),

         ('pca', PCA(n_components=0.99)),

         ('ridge', RidgeCV(alphas=(0.1, 0.3, 1.0, 3.0, 10.0)))])

    pipe.fit(X_train, y_train)

    return pipe.predict(X_test)
def get_X(df, mle=None):

    df = df.copy()

    df['Cabin'] = [str(i)[0] for i in df['Cabin'].fillna('Z')]

    df['Embarked'].fillna('Z', inplace=True)

    df['Fare'].fillna(df['Fare'].mean(), inplace=True)

    X = df.loc[:, EXOG]

    if mle:

        X = mle.transform(X)

    else:

        mle = MultiLabelEncoder(columns=['Sex', 'Cabin', 'Embarked'])

        X = mle.fit_transform(X)

    is_null_age = X['Age'].isnull()

    X.loc[is_null_age, 'Age'] = predict_age(

        X.loc[~is_null_age, EXOG].drop('Age', axis=1),

        X.loc[~is_null_age, 'Age'],

        X.loc[is_null_age, EXOG].drop('Age', axis=1))

    return X, mle
X_train, mle = get_X(df_train)

y_train = df_train[ENDOG]

X_test, _ = get_X(df_test, mle=mle)
forest = RandomForestClassifier(n_estimators=1000)

forest.fit(X_train, y_train)
print(forest.predict(X_test))