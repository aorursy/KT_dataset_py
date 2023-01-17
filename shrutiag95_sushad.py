# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
#df_train.fillna(df_train.mean())

for col in df_train.columns:
    df_train[col].fillna(df_train[col].mode()[0],inplace=True)

for col in df_test.columns:
    df_test[col].fillna(df_test[col].mode()[0],inplace=True)
    
    
matrix = df_train.corr()
interesting_variables = matrix['SalePrice'].sort_values(ascending=False)
# Filter out the target variables (SalePrice) and variables with a low correlation score (v such that -0.6 <= v <= 0.6)
interesting_variables = interesting_variables[abs(interesting_variables) >= 0.6]
interesting_variables = interesting_variables[interesting_variables.index != 'SalePrice']
interesting_variables





from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

pred_vars = [v for v in interesting_variables.index.values if v != 'SalePrice']
target_var = 'SalePrice'

X = df_train[pred_vars]
y = df_train[target_var]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

pred_ = [v for v in interesting_variables.index.values if v != 'SalePrice']
pred= df_test[pred_]
Id=df_test["Id"]
y_pred = model.predict(pred)
SalePrice=y_pred
dic={"Id":Id, "SalePrice":SalePrice}
df=pd.DataFrame(dic)
df.to_csv("predictions.csv")
##from sklearn.metrics import mean_squared_log_error, mean_absolute_error

#print('MAE:\t$%.2f' % mean_absolute_error(y_test, y_pred))
#print('MSLE:\t%.5f' % mean_squared_log_error(y_test, y_pred))