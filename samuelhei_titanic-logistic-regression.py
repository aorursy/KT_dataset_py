# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import re
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df
df.describe()
x_columns = list(df.columns)
x_columns.remove('Survived')

X_train, X_test, y_train, y_test = train_test_split(
    df[x_columns], df['Survived'], test_size=0.15, random_state=42
)
def transform(df):
    new_df = df.copy()
    new_df['Age'].replace(np.NaN, new_df['Age'].mean(), inplace=True)
    
    new_df['Age_c1'] = new_df['Age'].apply(lambda _: np.round(max(0,_) / new_df['Age'].std())).astype('category')
    new_df['Age_n1'] = new_df['Age'].apply(lambda _: max(0,_) / new_df['Age'].std())
    new_df['Age_log'] = new_df['Age'].apply(lambda _: max(0,np.log(_)))
    
    
    new_df['Fare_c1'] = new_df['Fare'].apply(lambda _: np.round(_ / new_df['Fare'].std())).astype('category')
    new_df['Fare_n1'] = new_df['Fare'].apply(lambda _: _ / new_df['Fare'].std())
    new_df['Fare_log'] = new_df['Fare'].apply(lambda _: max(0,np.log(_)))
    
    new_df['Pclass_c'] = new_df['Pclass'].astype('category')
    new_df['PSibSp_c'] = new_df['SibSp'].astype('category')
    new_df['Parch_c'] = new_df['Parch'].astype('category')
    
    new_df['Cabin'].fillna(0, inplace=True)
    new_df['Cabin'] = new_df['Cabin'].apply(lambda _: 1 if _ != 0 else 0)
    new_df['Ticket'] = new_df['Ticket'].apply(lambda _:re.sub('[^A-Z]','',_))
    new_df['Ticket'] = new_df['Ticket'].apply(lambda _: 'other' if len(_) == 0 else _)
    new_df['Name'] = new_df['Name'].apply(lambda _: list(filter(lambda c: '.' in c, list(_.split(' '))))[0])
    new_df.drop(['PassengerId','Name'],axis=1,inplace=True)
    
    return pd.get_dummies(new_df)

transform(df)
transf_test = transform(X_test)
transf_train = transform(X_train)
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

columns = intersection(transf_train.columns, transf_test.columns)
transf_test = transf_test[columns]
transf_train = transf_train[columns]
transf_train
grid = GridSearchCV(LogisticRegression(), {
    "tol" : [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7],
    "C" : [0.01, 0.1, 0.5, 1, 10, 100],
    "fit_intercept" : [True, False],
    "solver" : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
})

grid.fit(transf_train, y_train)

y_pred = grid.predict(transf_test)

print(classification_report(y_test, y_pred))

!cat ../input/gender_submission.csv
result = pd.read_csv('../input/test.csv')
#result.fillna(0,inplace=True)
result_transf = transform(result)
result_transf = result_transf[columns].fillna(0)
result_transf.describe()
result['Survived'] = grid.predict(result_transf[columns])
result[['PassengerId','Survived']]
result[['PassengerId','Survived']].to_csv('logistic_regressiong.csv', index=False)