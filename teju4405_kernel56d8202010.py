# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
county=pd.read_csv('/kaggle/input/us-counties-covid-19-dataset/us-counties.csv')

county.head()
county.info()
county['cases'].sum()
county['deaths'].sum()
us_county=county.set_index(['county','state'])

us_county.head(50)
us_county.describe()
cases=us_county[(us_county['cases']>10)&(us_county['deaths']>10)]

cases.head()
us_county['cases'].nlargest(3)
us_county['deaths'].nlargest(3)
group=county.groupby('state')['county','cases','deaths'].mean()

group.head()
group.plot.bar(subplots=True,figsize=(10,10))
washington=county[(county['state']=='Washington')|(county['state']=='New York')|(county['state']=='California')]

washington.head()
sno_group=washington.groupby('state')['county','cases','deaths'].mean()

sno_group.plot.bar(subplots=True,figsize=(7,7),title='Washington,California,New York')
import seaborn as sns

from matplotlib import pyplot as plt

sns.regplot(x='cases',y='deaths',data=county)

plt.ylim(0,)

from sklearn.model_selection import train_test_split

x=us_county['cases'].values.reshape(-1,1)

y=us_county['deaths'].values.reshape(-1,1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)


from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

gnb=GaussianNB()



pred=gnb.fit(x_train,y_train).predict(x_test)

print('accuracy score is',accuracy_score(y_test,pred,normalize=True))
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

std=StandardScaler()

lm=LinearRegression()

pf=PolynomialFeatures(degree=2,include_bias=False)
s=[('Scale',StandardScaler()),('polynomial',PolynomialFeatures(degree=2)),('mode',LinearRegression())]

pipe=Pipeline(s)
z=pipe.fit(x,y)
s=pipe.predict(y)

print(s)
p=pipe.predict(x)

print(p)

sns.regplot(s,p)