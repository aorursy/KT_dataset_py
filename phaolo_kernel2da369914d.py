# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import seaborn as sns
import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/Most-Recent-Cohorts-Scorecard-Elements.csv')
df.shape
df2=df[['UNITID','INSTNM','STABBR','CITY','CONTROL','SATVRMID','SATMTMID','SATWRMID','ACTCMMID','ACTENMID','ACTMTMID','ACTWRMID','NPT4_PUB','NPT4_PRIV','MD_EARN_WNE_P10','GRAD_DEBT_MDN_SUPP']]

df2.isna().sum()
city_count =df2['CITY'].value_counts()
city_count = city_count[:10,]
plt.figure(figsize=(10,5))
g=sns.barplot(city_count.index, city_count.values, alpha=0.8)
plt.title('Top 10 number of colleges in the USA cities')
plt.ylabel('Number of Colleges', fontsize=12)
plt.xlabel('city', fontsize=12)
plt.xticks(rotation=30)
plt.show()
mystring=df2['CONTROL'].replace([1,2,3], ['Public','Private nonprofit','Private for-profit'])
city_count =mystring.value_counts()

city_count = city_count[:10,]
plt.figure(figsize=(10,5))
g=sns.barplot(city_count.index, city_count.values, alpha=0.8)
plt.title('3 types of colleges')
plt.ylabel('Number of Colleges', fontsize=12)
plt.xlabel('Ownership', fontsize=12)
plt.show()
city_count
describedf=df2[['SATVRMID','NPT4_PRIV']]
describedf.describe()
df2['SATVRMID'].max()
df2['CONTROL']=df2['CONTROL'].replace([1,2,3], ['Public','Private nonprofit','Private for-profit'])

df2[['INSTNM','SATVRMID','CONTROL','NPT4_PRIV']][(df2.SATVRMID > 725)].sort_values(by=['SATVRMID'], ascending=False)
datasat2=df2[['INSTNM','SATVRMID','CONTROL','NPT4_PRIV']][(df2.SATVRMID > 725)]
ax=sns.jointplot(x="NPT4_PRIV", y="SATVRMID", data=datasat2, kind="reg")
plt.subplots_adjust(top=0.9)
ax.fig.suptitle('SAT score compared to price regression plot')
df2[['INSTNM','SATVRMID','CONTROL','NPT4_PRIV']][(df2.SATVRMID > 740)&(df2.NPT4_PRIV < 20000 )].sort_values(by=['SATVRMID'], ascending=False)
df3=df2.dropna(subset = ['NPT4_PRIV', 'SATVRMID'])
from sklearn.linear_model import LinearRegression
X = df3[['NPT4_PRIV']]
y = df3.SATVRMID

lfit=LinearRegression()
lfit.fit(X, y)
lfitpredict=lfit.predict(X)


plt.scatter(X, y, alpha=0.4)
plt.plot(X,lfitpredict)
plt.title("Linear regression on SAT score and cost")
plt.xlabel("NPT4_PRIV")
plt.ylabel("SATVRMID")
plt.show()

lfit.score(X, y)

lfit.coef_
lfit.intercept_

df2['NPT4_PRIV'].max()
df2[['INSTNM','SATVRMID','CONTROL','NPT4_PRIV']][df2.NPT4_PRIV > 50000.0]
