import numpy as np
import pandas as pd

import seaborn as sns
%matplotlib inline
tips = pd.read_csv('../input/tips.csv')
tips.head()
sns.lmplot(x='total_bill',y='tip',data=tips)
sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex')
sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',palette='coolwarm',markers=['o','v'],scatter_kws={'s':100})
sns.lmplot(x='total_bill',y='tip',data=tips,col='sex')
sns.lmplot(x='total_bill',y='tip',data=tips,col='smoker',hue='sex')
sns.lmplot(x="total_bill", y="tip", row="sex", col="time",data=tips)

sns.lmplot(x="total_bill", y="tip", row="sex", col="time",data=tips, hue='smoker')

sns.lmplot(x='total_bill',y='tip',data=tips,col='day',hue='sex',palette='coolwarm',
          aspect=0.6,size=8)
