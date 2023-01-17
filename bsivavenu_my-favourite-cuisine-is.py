import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_json('../input/train.json')
train.head()
test = pd.read_json('../input/test.json')
test.head()
sub = pd.read_csv('../input/sample_submission.csv')
sub.head()
train.set_index('id',inplace=True)
train.head()
temp = train.cuisine.value_counts()
temp
plt.figure(figsize=(20,5))
sns.countplot(train.cuisine)
plt.show()
(train.iloc[3,1]+train.iloc[4,1])
from collections import Counter
top_n = Counter([item for sublist in train.ingredients for item in sublist]).most_common(20)
top_n
ingredients = pd.DataFrame(top_n,columns=['ingredient_name','cnt'])
ingredients
plt.figure(figsize=(20,20))
sns.barplot(x = ingredients.cnt,y = ingredients.ingredient_name)
plt.show
temp1 = train[train['cuisine'] == 'chinese']
n=6714 # total ingredients in train data
top = Counter([item for sublist in temp1.ingredients for item in sublist]).most_common(n)
temp= pd.DataFrame(top)
temp.columns = ['ingredient','total_count']
temp = temp.head(20)
plt.figure(figsize=(20,20))
sns.barplot(y = temp.ingredient,x=temp.total_count)
plt.show()
temp1 = train[train['cuisine'] == 'indian']
n=6714 # total ingredients in train data
top = Counter([item for sublist in temp1.ingredients for item in sublist]).most_common(n)
temp= pd.DataFrame(top)
temp.columns = ['ingredient','total_count']
temp = temp.head(20)
plt.figure(figsize=(20,20))
sns.barplot(y = temp.ingredient,x=temp.total_count)
plt.show()
temp1 = train[train['cuisine'] == 'italian']
n=6714 # total ingredients in train data
top = Counter([item for sublist in temp1.ingredients for item in sublist]).most_common(n)
temp= pd.DataFrame(top)
temp.columns = ['ingredient','total_count']
temp = temp.head(20)
plt.figure(figsize=(20,20))
sns.barplot(y = temp.ingredient,x=temp.total_count)
plt.show()
temp1 = train[train['cuisine'] == 'mexican']
n=6714 # total ingredients in train data
top = Counter([item for sublist in temp1.ingredients for item in sublist]).most_common(n)
temp= pd.DataFrame(top)
temp.columns = ['ingredient','total_count']
temp = temp.head(20)
plt.figure(figsize=(20,20))
sns.barplot(y = temp.ingredient,x=temp.total_count)
plt.show()