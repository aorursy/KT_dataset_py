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
path = '/kaggle/input/competitive-data-science-final-project/'
sales_train = pd.read_csv(path+'sales_train.csv.gz')

sales_train.head(5)
shops = pd.read_csv(path+'shops.csv')

shops.head(5)
item_categories = pd.read_csv(path+'item_categories.csv')

item_categories.head(5)
item_cats = pd.read_csv(path+'item_cats.csv')

item_cats.head(5)
items = pd.read_csv(path+'items.csv')

items.head(5)
test = pd.read_csv(path+'test.csv.gz')

test.head()
sample_submission = pd.read_csv(path+'sample_submission.csv.gz')

sample_submission.head(5)
# sales_train[['day','month','year']] = sales_train['date'].str.split(".",expand=True)
# sales_train.head()
sales_train_shop_item = sales_train.groupby(by=['date_block_num','shop_id','item_id'])['item_cnt_day'].sum()
sales_train_shop_item
sales_train_shop_item = sales_train_shop_item.reset_index()
sales_train_shop_item
sales_train_shop_item.rename(columns={"item_cnt_day":"item_cnt_month"}, inplace=True)
sales_train_shop_item.head()
from sklearn.linear_model import LinearRegression
import numpy as np

from sklearn.linear_model import LinearRegression

X = sales_train_shop_item[["date_block_num", "shop_id", "item_id"]]

y = sales_train_shop_item["item_cnt_month"]

reg = LinearRegression().fit(X, y)

# >>> reg.score(X, y)

# 1.0

# >>> reg.coef_

# array([1., 2.])

# >>> reg.intercept_

# 3.0000...

# >>> reg.predict(np.array([[3, 5]]))
y_test = reg.predict(test)
y_test
submit = pd.DataFrame({'ID':np.arange(len(y_test)),'item_cnt_month':np.clip(y_test, a_min = 0, a_max = 20)},columns=['ID','item_cnt_month'])
submit.to_csv('submission.csv',index = False)