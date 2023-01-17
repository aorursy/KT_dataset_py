# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.model_selection import train_test_split # random split to train and test subset
data = pd.read_csv('../input/train.csv')
# eldobjuk azokat a recordokat (sorokat), ahol nincs ár
data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = data.SalePrice
# eldobjuk a SalePrice oszlopot, ebbol lesz a tanulo es teszthalmaz
# azokat az oszlopokat is eldobjuk, amelyek nem szám tipusuak
x = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
# 20% legyen a teszt adathalmaz
train_x, test_x, train_y, test_y = train_test_split(x.as_matrix(), y.as_matrix(), test_size = 0.1)
# adattisztitas
from sklearn.preprocessing import Imputer
# feltolti a hianyzo elemeket az adott feature kozepertekevel
imp = Imputer()
train_x = imp.fit_transform(train_x)
test_x = imp.transform(test_x)
print(train_x.size)
print(test_x.size)
from xgboost import XGBRegressor # XGBoost package for extreme gradient boosting

xgboost_model = XGBRegressor()
start_time = time.time()
xgboost_model.fit(train_x, train_y, verbose=False)
t_xgb = time.time() - start_time
print("XGBoost tanitas: %s sec" % t_xgb)
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor()
start_time = time.time()
gb_model.fit(train_x, train_y)
t_gb = time.time() - start_time
print("Gradiens Boost tanitas: %s sec" % t_gb)
print(t_gb/t_xgb*100)
xgb_predict = xgboost_model.predict(test_x)
gb_predict = gb_model.predict(test_x)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error az XGBoost modellben: " + str(mean_absolute_error(xgb_predict, test_y)))
print("Mean Absolute Error a sima Gradiens boost modellben: " + str(mean_absolute_error(gb_predict, test_y)))
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
st = time.time()
my_model.fit(train_x, train_y, early_stopping_rounds=5, 
             eval_set=[(test_x, test_y)], verbose=False)
end = time.time()
pred = my_model.predict(test_x)
print("Mean Absolute Error az XGBoost modellben: " + str(mean_absolute_error(pred, test_y)))
print("Futasi ido: %s sec" % (end-st))
