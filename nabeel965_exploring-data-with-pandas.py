import pandas as pd
train=pd.read_csv("../input/train.tsv", sep = "\t")

test=pd.read_csv("../input/test.tsv", sep = "\t")
train.head()
train['item_condition_id'][:1000].value_counts()
List=train['brand_name'][:1000].value_counts()

print(List[:10])
List=train['category_name'][:1000].value_counts()

print(List[:10])
train['shipping'][:1000].value_counts()
pd.get_dummies(train[:10])