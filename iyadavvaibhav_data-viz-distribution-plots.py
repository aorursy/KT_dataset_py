import seaborn as sns
import pandas as pd
import os as os
%matplotlib inline
tips = pd.read_csv('../input/tips.csv')
tips.head()
sns.distplot(tips['total_bill'])
sns.distplot(tips['total_bill'],kde=False,bins=30)
sns.jointplot(x='total_bill',y='tip',data=tips)
sns.jointplot(x='total_bill',y='tip',data=tips,kind='hex')
sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg')
sns.pairplot(tips)
sns.pairplot(tips, hue='sex')
sns.rugplot(tips['total_bill'])
sns.kdeplot(tips['total_bill'])
sns.rugplot(tips['total_bill'])
