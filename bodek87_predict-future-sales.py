import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sales = pd.read_csv("../input/sales_train.csv")
items = pd.read_csv("../input/items.csv")
shops = pd.read_csv("../input/shops.csv")
item_cat = pd.read_csv("../input/item_categories.csv")
sales.head(5)
sales["date"] = pd.to_datetime(sales["date"],format="%d.%m.%Y")
sales.date[2]
df = pd.merge(sales, shops, on='shop_id')
df =pd.merge(df,items, on="item_id")
df = pd.merge(df,item_cat, on="item_category_id")
df.head(20)
df.item_category_name.value_counts()
df.describe()
df.shape
df.head(10)
test = pd.read_csv("../input/test.csv")
test.head(5)
submission = pd.read_csv("../input/sample_submission.csv")
submission.head()
df.item_name[df.item_category_id == 37].value_counts()
temp = df[df.item_name == "ОБЛИВИОН (BD)"]
temp.head()
sns.countplot(y= temp.date_block_num)

sns.countplot(y= temp.item_cnt_day)
df.data_range('2013-01-02','2013-01-31')
