import pandas as pd
import numpy as np
from sklearn import tree
import os

category = pd.read_csv("../input/item_categories.csv")
items = pd.read_csv("../input/items.csv")
train = pd.read_csv("../input/sales_train.csv")
sample = pd.read_csv("../input/sample_submission.csv")
shops = pd.read_csv("../input/shops.csv")
test = pd.read_csv("../input/test.csv")

#group by'date_block_num','shop_id','item_id' to calc monthly item_cnt
train = train.groupby(['date_block_num','shop_id','item_id'], as_index=False)['item_cnt_day'].sum().rename(columns={'item_cnt_day': 'item_cnt_month'})
train.head()
#print(train)
target = train["item_cnt_month"].values
features = train[["date_block_num","shop_id","item_id"]].values
train.head()
#fit target data
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features, target)

test_features = test[["shop_id","item_id"]].values

test_features = pd.DataFrame(np.insert(test_features, 0, 3, axis=1))

my_prediction = my_tree_one.predict(test_features)
ShopId = np.array(test["ID"]).astype(int)
my_solution = pd.DataFrame(my_prediction, ShopId, columns = ["item_cnt_month"])

my_solution.to_csv("my_sub.csv", index_label = ["ID"])
