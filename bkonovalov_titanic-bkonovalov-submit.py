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
import pandas as pd

from pandas import DataFrame 

df = pd.read_csv("../input/titanic/train.csv")

test_df = pd.read_csv("../input/titanic/test.csv")
test_df = test_df.fillna(df.mean())

df = df.fillna(df.mean())
df
columns = list(df.columns)

columnstest = list(test_df.columns)

print(columns)

print(columnstest)





x_cols = ['Sex']



#dfx1 = df.copy().drop(drop_cols, 1)

#X = dfx1.copy()



y_cols = 'Survived'

print(df[x_cols])
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()





tmp1 = []

tmp2 = []

for i in df[x_cols]:

    tmp1.append(i)

for i in test_df[x_cols]:

    tmp2.append(i)

test_df[x_cols] = le.fit_transform(tmp2)

df[x_cols] = le.fit_transform(tmp1)





print(df[x_cols])

# test_df = test_df[x_cols]

    

    

#   test_df['Sex'] = le.transform(col_vals)



print(df.head())

print(test_df.head())

test_df = test_df.fillna(df.mean())

df = df.fillna(df.mean())
print(df)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

#['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

X = df[features].values

test_X = test_df[features].values



y = df['Survived'].values

print(X)



df

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

#model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=10, objective= 'binary:logistic', colsample_bytree=0.8, subsample=0.8, booster='gbtree')



model = KNeighborsClassifier(n_neighbors = 1)

model.fit(train_X,train_y)



acc_knn = round(model.score(train_X, train_y) * 100, 2)

print(acc_knn)

# mae = mean_absolute_error(yp, val_y)



# gbtree: tree-based models

# gblinear: linear models

#0.3358242068151077

#0.3358242068151077



scores = -1 * cross_val_score(model, train_X, train_y,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print(scores)

print(scores.mean())

model.fit(train_X,train_y)

yp = model.predict(test_X)
# from sklearn.metrics import accuracy_score

# yp = np.rint(yp)

# acc = accuracy_score(yp, y)

# print(yp[:10])

# print(y[:10])

# print(acc)

yp = np.rint(yp)

print(yp[:10])

yp = yp.astype(int)

print(yp[:10])


sub = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": yp

    })

print(sub.head())
sub.to_csv('submission.csv', index=False)

print('done')