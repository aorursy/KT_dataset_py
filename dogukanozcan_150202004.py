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
item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv") 

sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv") 

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
df = item_categories.merge(items,on="item_category_id")
df = sales_train.merge(df, on="item_id")
df = df.merge(shops, on="shop_id")
df.head(10)
df.info()
df['date'] = pd.to_datetime(df['date'])
df.head(10)
df = df.drop(['shop_name','item_category_name','item_name','date'], axis=1)
from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test = train_test_split(df.drop('item_cnt_day',axis=1), 

                                                    df['item_cnt_day'], test_size=0.35, random_state=101)

                                      
df.head(10)
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

from sklearn import metrics 

logmodel = RandomForestRegressor(n_estimators=25,random_state=0)

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
print(r2_score(y_test,predictions))

metrics.mean_absolute_error(y_test, predictions)
predictions = pd.DataFrame(predictions,columns=["item_cnt_month"])

predictions = predictions.clip(0,20)
submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")
submission.drop(columns=['item_cnt_month'],inplace=True)

submission=pd.concat([submission,predictions],axis=1)
submission = submission.dropna()
submission["ID"] = pd.to_numeric(submission['ID'])
submission.to_csv('submission.csv', index=False)