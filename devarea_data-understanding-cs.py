import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb
!ls ../
df = pd.read_csv('../input/pupils/pupils.csv')
df.head()

df.info()

df.describe()
#df.corr()



#df.query("Age > 10 and rooms < 5")

#df.Country.value_counts()

#df.Country.unique()

#df.Country.nunique()

#df.sort_values(['Age','Weight'],ascending=False)

#df.groupby('Country').mean()

#df.groupby(['Age','type']).count()



#df.agg('count')

#df.agg(['count','max'])

#df.agg({'Name':'count', 'income':'sum' , 'Age':'max'})



#sb.pairplot(df,hue='gen')

#sb.heatmap(df.corr(),cmap="YlGnBu")

#sb.countplot(x='Age',data=df)

#sb.countplot(x='rooms',data=df,hue='type')

#sb.catplot(x='rooms',hue='gen',data=df,col='type',kind='count')

#sb.distplot(df['Age'])

#df['income'].hist()

#sb.boxplot(x='rooms',y='Age',data=df)

#plt.scatter(df.Height,df.Weight)