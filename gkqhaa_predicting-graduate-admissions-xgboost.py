# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

df.head()
print(df.shape)
print(df.info())
X = df.drop(['Chance of Admit '], axis=1)

y = df['Chance of Admit ']

X.info()

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(X_train,y_train)

coef = pd.DataFrame(lr.coef_, X_test.columns, columns = ['Co-efficient'])

print(coef)
import xgboost as xgb



xgtrain = xgb.DMatrix(X_train, label=y_train)

xgtest = xgb.DMatrix(X_test)



xg_reg = xgb.XGBRegressor()

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
from sklearn.metrics import r2_score



preds_train = xg_reg.predict(X_train)

print("r_square score (test): ", r2_score(y_test, preds))

print("r_square score (train): ", r2_score(y_train, preds_train))
print("Some predictions vs real data:")

for i in range(0,5):

    print(preds[i], "\t",y_test.iloc[i])