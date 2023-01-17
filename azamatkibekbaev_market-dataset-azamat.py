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
df=pd.read_excel('/kaggle/input/market/data1.xlsx')
df.head()
df=df.drop(['дата приобретения товара','период','дата заказа','дата отправки','Unnamed: 0'],axis=1)
# use it if you want to convert nominals to numericals
from sklearn.preprocessing import LabelEncoder
x=['товар', 'адрес', 'количество', 'цена', 'общая стоимость',
       'кол-во возвращенного товара', 'кол-во возвращенного товара по браку',
       'кол-во возвращенного товара по сроку годности', 'себестоимость товара',
       'расходы по возврату', 'прибыль от продажи',
       'прибыль на единицу продукции']
for i in x:
    a=LabelEncoder()
    df[i]=a.fit_transform(df[i])
df
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['товар', 'адрес','цена', 'общая стоимость',
       'кол-во возвращенного товара', 'кол-во возвращенного товара по браку',
       'кол-во возвращенного товара по сроку годности', 'себестоимость товара',
       'расходы по возврату', 'прибыль от продажи',
       'прибыль на единицу продукции']], df['количество'], test_size=0.3, random_state=3)
from sklearn.linear_model import LinearRegression,SGDRegressor, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
drugTree = LinearRegression()
drugTree.fit(X_train,y_train)
predTree = drugTree.predict(X_test)

knn = RidgeCV()
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

nbc = RandomForestRegressor()
nbc.fit(X_train,y_train)
y_pred = nbc.predict(X_test)

logmodel = xgb.XGBRegressor()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
print("LinearRegression's Accuracy: ", r2_score(y_test, predTree))


print("RidgeCV's Accuracy: ", r2_score(y_test, pred))


print("RandomForestRegressor's Accuracy: ", r2_score(y_test, y_pred))


print("XGBoostRegressor's Accuracy: ", r2_score(y_test, predictions ))
