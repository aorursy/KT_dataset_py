# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/BlackFriday.csv')
df.info()
df.isnull().sum().sort_values(ascending=False)
df.sample(10)
df.nunique()
df.Purchase.describe()
box1 = go.Box(y=df.Purchase,
              name='Purchase',
              marker = dict(color = 'rgba(15, 100, 150)'),
              boxmean='sd')

databox1 = [box1]
iplot(databox1)
df['Wealthiness'] = np.where(df.Purchase < df.Purchase.mean(), "Poor", "Rich")
df.Wealthiness.value_counts()
bar1 = go.Bar(x=df.Wealthiness.value_counts().index,
              y=df.Wealthiness.value_counts().values,
              marker = dict(color = 'rgba(150, 180, 32)',
                            line=dict(color='rgb(104,32,0)',width=1.5))
              )
databar = [bar1]
iplot(databar)
df.groupby('Gender').agg(dict(Purchase=['min', 'mean', 'max']))
plt.figure(figsize=(7,5))
sns.barplot(x=df.Gender.value_counts().index,y=df.Gender.value_counts().values )
plt.show()
df.Age.value_counts()
plt.figure(figsize=(7,5))
sns.barplot(x=df.Age.value_counts().index, y=df.Age.value_counts().values)
plt.show()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_new = le.fit_transform(df.Age) 

plt.figure(figsize=(7,5))
sns.distplot(df_new,hist=False)
plt.show()
plt.figure(figsize=(7,5))
sns.distplot(df.Purchase)
print('Skewness:', df.Purchase.skew())
print('Kurtosis:', df.Purchase.kurt())
plt.figure(figsize=(6,6))
sns.boxplot(x=df.Purchase)
plt.show()
df2 = df.drop(['User_ID'],axis=1)

plt.figure(figsize=(8,8))
sns.heatmap(df2.corr(), annot=True, linewidths=0.5, linecolor='black', cmap='Blues')
plt.xticks(rotation=90)
plt.show()
moneybox = []
consumer_list = list(df.User_ID.unique())

for i in consumer_list:
    money = df[df.User_ID == i].Purchase.sum()
    moneybox.append(money)
users_purchases = pd.DataFrame(dict(Users=consumer_list, Purchases=moneybox))
indices = (users_purchases.Purchases.sort_values(ascending=False)).index.values[:10]
users_purchases = users_purchases.reindex(indices).iloc[:10]
users_purchases
plt.figure(figsize=(8,6))
users_purchases.groupby(["Users"])['Purchases'].sum().sort_values(ascending=False).plot('bar', position=0.2)
plt.xticks(rotation=45)
plt.ylabel('Purchases')
plt.show()
result = users_purchases.groupby(["Users"])['Purchases'].aggregate(np.median).reset_index().sort_values('Purchases')

plt.figure(figsize=(8,6))
sns.barplot(x=users_purchases.Users, y=users_purchases.Purchases, order=result['Users'])
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(12,7))
sns.countplot(x=df.Product_Category_1, hue=df.Gender)
plt.show()
df['Marit_of_Genders'] = df.apply(lambda x:'%s.%s' % (x['Gender'],x['Marital_Status']),axis=1)
plt.figure(figsize =(12,7))
sns.countplot(x=df.Age, hue=df.Marit_of_Genders)
plt.show()
plt.figure(figsize =(12,7))
sns.countplot(x=df.Product_Category_1, hue=df.Marit_of_Genders)
plt.show()
plt.figure(figsize=(7,5))
df.groupby('Gender')['Age'].value_counts().sort_values().plot('bar')
plt.xticks(rotation=60)
plt.show()
plt.figure(figsize=(7,5))
df.groupby('Product_ID')['Purchase'].count().nlargest(10).sort_values().plot('barh')
plt.show()