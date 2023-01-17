# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn import metrics

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

import operator



import pandas as pd

import datetime

import numpy as np

import matplotlib.pyplot as plt
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

egitim = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

test  = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
items.info()
egitim.info()
test.info()
egitim['item_cnt_day'] = egitim['item_cnt_day'].astype(np.int16)
egitim.sort_values(by="item_cnt_day")
egitim.loc[egitim.item_cnt_day < 0, "item_cnt_day"] = 0
egitim.sort_values(by=["item_cnt_day"]).head(2)
egitim.drop_duplicates(inplace=True)
egitim.reset_index(inplace=True)

egitim.drop(["index"], axis=1, inplace=True)
egitim.info()
egitim.drop(["item_price", "date"], axis=1, inplace=True) # Kullanilmayacagini dusundugum sutunlari drop ediyorum.
toplamEsya = egitim.groupby(["date_block_num","shop_id","item_id"]).sum().reset_index()
tumVeriler = []



for i in range(34):

  for shop in toplamEsya.shop_id.unique():

    for item in toplamEsya.item_id.unique():

      tumVeriler.append([i, shop, item])
tumVerilerDF = pd.DataFrame(tumVeriler, columns=["date_block_num", "shop_id", "item_id"])
tumVerilerDF.info()
tumVerilerDF.head()
aylikToplam = pd.merge(tumVerilerDF, toplamEsya, on=['date_block_num','shop_id','item_id'], how='left')

aylikToplam.fillna(0, inplace=True)
aylikToplam.rename(columns={"item_cnt_day": "item_cnt_month"},inplace=True)
aylikToplam.loc[aylikToplam.shop_id == 42].loc[aylikToplam.item_id == 20949]
testVeriSeti = aylikToplam.loc[aylikToplam.shop_id == 42].loc[aylikToplam.item_id == 20949].reset_index()

testVeriSeti.drop(["index"], axis=1, inplace=True)
x = testVeriSeti.date_block_num

y = testVeriSeti.item_cnt_month



x = x[:, np.newaxis]

y = y[:, np.newaxis]



polynomial_features= PolynomialFeatures(degree=4)

x_poly = polynomial_features.fit_transform(x)



model = LinearRegression()

model.fit(x_poly, y)

y_poly_pred = model.predict(x_poly)



rmse = np.sqrt(mean_squared_error(y,y_poly_pred))

r2 = r2_score(y,y_poly_pred)

print("rmse:",rmse)

print("r2",r2)



plt.scatter(x, y, s=10)

sort_axis = operator.itemgetter(0)

sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)

x, y_poly_pred = zip(*sorted_zip)

plt.plot(x, y_poly_pred, color='m')

plt.show()