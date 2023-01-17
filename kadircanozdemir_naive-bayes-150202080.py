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
sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
sales_train.head()
df = sales_train[['date_block_num','shop_id','item_id','item_cnt_day']].groupby(['date_block_num','shop_id','item_id']).sum().reset_index()

df
y = df.item_cnt_day.values

x = df.drop(["item_cnt_day"], axis=1)

x = (x - np.min(x)) / (np.max(x) - np.min(x))

x
y = y.reshape(-1,1)

y.shape
x.shape
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x, y)
print(nb.score(x_test, y_test))
y_head = nb.predict(x_test)
tbl = x_test
tbl['results'] = pd.Series(y_head)
tbl.describe()
test = test [['ID', 'shop_id', 'item_id']]
test = (test - np.min(test)) / (np.max(test) - np.min(test))
test.ID = 1.0
test
test_head = nb.predict(test)
test_tbl = test
test_tbl['results'] = pd.Series(test_head)
test_tbl.describe()