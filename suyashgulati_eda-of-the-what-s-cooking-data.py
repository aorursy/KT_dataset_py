import pandas as pd
import numpy as np
train = pd.read_json("../input/train.json")
train.cuisine.unique()
import matplotlib.pyplot as plt
plt.style.use('ggplot')
ax = train['cuisine'].value_counts().plot(kind='bar',
                                    figsize=(14,8),
                                    title="Cuisine with most data")
ax.set_xlabel("Cuisine")
ax.set_ylabel("Frequency")
from sklearn.preprocessing import MultiLabelBinarizer
def count_ingredients():
    mlb = MultiLabelBinarizer()
    df = pd.DataFrame(mlb.fit_transform(train['ingredients']),columns=mlb.classes_, index=train.index)
    df = pd.concat([train,df],axis=1)
    df = df.drop("ingredients",axis =1)
    df = df.drop("id",axis =1)
    df = df.groupby('cuisine').sum()
    return df
df = count_ingredients()
dd = pd.DataFrame(np.where(df>0, 1, 0),columns = df.columns)
for_only_one = dd.drop([col for col, val in dd.sum().iteritems() if val >=2], axis=1).columns
for_only_one_final = df[for_only_one].drop([col for col, val in df[for_only_one].sum().iteritems() if val <1], axis=1).columns
only_one = df[for_only_one_final].transpose()
def cuisine_specific(cuisine):
  x = only_one[only_one[cuisine]>0].loc[:,cuisine].sort_values(ascending=False, axis=0).iloc[:15]
  ax = x.plot(kind='barh',figsize=(15,8),title="Items found only in cuisine "+cuisine,width = 0.9,alpha =0.9)
  ax.set_ylabel("Items")
  ax.set_xlabel("Count")
cuisine_specific("brazilian")
cuisine_specific("italian")
cuisine_specific("indian")
df['my_sum'] = df.iloc[:,:].sum(1)
x = df['my_sum']/train['cuisine'].value_counts()
x = x.sort_values(ascending=False)
ax = x.plot(kind='bar',figsize=(12,8),title="Average number of ingredients per dish",width = 0.9,alpha =0.9)
ax.set_xlabel("Cuisine")
ax.set_ylabel("Average")
df = df.drop("my_sum",axis =1)
common_ingredients = df.sum(axis =0).sort_values(ascending=False, axis=0).iloc[:10]
ax = common_ingredients.plot(kind='bar',figsize=(10,8),title="Most common ingredients",width = 0.5)
ax.set_xlabel("Ingredients")
ax.set_ylabel("Count")
df.drop(common_ingredients.index, axis=1, inplace=True)
def top_ingredients(cuisine,number_of):
    bx = df[df.index== cuisine].sort_values(by=cuisine, ascending=False, axis=1).iloc[:,:number_of].transpose().plot(kind='barh',figsize=(14,6),title="Top ingredients from cuisine",width = 0.5)
    bx.set_ylabel("Ingredients")
    bx.set_xlabel("Count")
    return
top_ingredients("italian",10)
top_ingredients("indian",10)
top_ingredients("mexican",10)
