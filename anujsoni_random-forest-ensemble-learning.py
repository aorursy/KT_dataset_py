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
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

cat = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

test  = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
train.head()
grp = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day':['sum']})
grp
x= np.array(list(map(list, grp.index.values)))

y_train = grp.values
test['date_block_num'] = train['date_block_num'].max()+1

x_test = test[['date_block_num', 'shop_id', 'item_id']].values
from sklearn.preprocessing import OneHotEncoder

oh1 = OneHotEncoder(categories='auto').fit(x[:,1].reshape(-1, 1))

x1 = oh1.transform(x[:,1].reshape(-1, 1))

x1_t = oh1.transform(x_test[:,1].reshape(-1, 1))
x_train= np.concatenate((x[:,:1],x1.toarray(),x[:,2:]),axis=1)

x_test = np.concatenate((x_test[:,:1],x1_t.toarray(),x_test[:,2:]),axis=1)
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

rfr.fit(x_train,y_train.ravel())

y_test = rfr.predict(x_test)

submission['item_cnt_month'] = y_test

submission.to_csv('submission.csv',index=False)