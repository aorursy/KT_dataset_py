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
!ls
import pandas as pd

import pandas_profiling as pdp
#DICTIONARY READ TRANSLATE RUSSIAN TO ENGLISH

dic_shop_EN = pd.read_csv('../input/future-sales-en-dic/shop_en.csv')

dic_shop_EN.head(20)
#DICTIONARY TO EN for item categories.

dic_itemcat_EN = pd.read_csv('../input/future-sales-en-dic/item_categories_en.csv',encoding='cp932',usecols=[0,1,2,3])

dic_itemcat_EN.head(20)
sales_train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

print(len(sales_train))

#ADD SALES AMT

sales_train["Sales AMT"]= sales_train["item_price"]* sales_train["item_cnt_day"]

#英語名の店名を追記 Add En shopname

sales_train = pd.merge(sales_train, dic_shop_EN, on ="shop_id", how="left")

sales_train.head(10)
sales_train.tail(10)
sales_train.isnull().sum()

sales_train.describe()
#pdp.ProfileReport(sales_train)

#実行すると、とても重いです。（GPU浪費）SUPER HEAVY WHEN YOU RUN PROFILE REPORT(>GPU Waste)
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

print(len(test))

test.head(5)
test.tail(5)

test.isnull().sum()
item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

item_categories.to_csv("item_categories.csv")

print(len(item_categories))

item_categories.head(5)
item_categories.tail(5)
item = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

#商品カテゴリに英文を追加。ADD ITEM CAT INTO item df

item = pd.merge(item,dic_itemcat_EN, on="item_category_id", how="left")

print(len(item))

item.head(5)

item.tail(5)
item.isnull().sum()
#店舗データ抽出用　shop = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

#shop.to_csv("shop.csv")

#print(len(shop))

#shop.head(5)
#shop.tail(5)

#shop.isnull().sum()
sales_train = pd.merge(sales_train, item, on ="item_id", how="left")

delete_columns = ['Column1_y','Column1_x']#なぜかshop_idとitem_category_idが重複するので列削除。(leftmergeの使い方が悪い）Delete unnecessary columns due to left-merges.

sales_train.drop(delete_columns, axis=1, inplace=True)

#DateをYYYYMM単位に簡潔化

sales_train["date"]=pd.to_datetime(sales_train["date"])

sales_train["date"]= sales_train["date"].dt.strftime("%Y%m")



print(len(sales_train))#行数は293,549のままのはず...Length is originally 293,549.Check.

sales_train.head(10)
sales_train.tail(10)
###CSV EXPORT CSVファイル出力するときには＃を外してください。Delete"#" when you export CSV. "###



#sales_train.to_csv("sales_train_EN01.csv")#DLすると６００MB 超えなので注意。VERY HEAVY DATA(>600MB)
#年月別カテゴリ別売上 Monthly Sales AMT by item cat.

#sales_train.groupby(["date","item_category_name_EN"]).sum()["Sales AMT"]
print("SUCCESS")