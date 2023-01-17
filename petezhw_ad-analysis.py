# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import and merge the Months data Show list

xlsxFile1 = '/kaggle/input/ad-data-from-ld/Project Data_ Month 1.xlsx'

df1 = pd.read_excel(xlsxFile1)

xlsxFile2 = '/kaggle/input/ad-data-from-ld/Project Data_ Month 2.xlsx'

df2 = pd.read_excel(xlsxFile2)

xlsxFile3 = '/kaggle/input/ad-data-from-ld/Project Data_ Month 3.xlsx'

df3 = pd.read_excel(xlsxFile3)

frames = [df1, df2, df3]

df=pd.concat(frames)

df.head()
df.dtypes
import pandas_profiling

df.profile_report(title='Ad dataset')
# convert creative_id int to object

df.creative_id=df.creative_id.astype(str)

df.ad_unit_id=df.ad_unit_id.astype(str)
df.dtypes
# Get some general information about dataframe:

df.describe()
df.groupby('device').size()
df.groupby('device').sum()[['conversions']]
df['creative_id'].describe()
df['ad_unit_id'].describe()
# Determine if ANY Value in df is Missing

df.isnull().values.any()


df['creative_id']=df['creative_id'].astype('category').cat.codes

df['ad_unit_id']=df['ad_unit_id'].astype('category').cat.codes
df['Year'] = pd.DatetimeIndex(df['date']).year

df['Month'] = pd.DatetimeIndex(df['date']).month

df['Day'] = pd.DatetimeIndex(df['date']).day

df['Weekday'] = pd.DatetimeIndex(df['date']).weekday

df.head()
# Weekdays start with 0 (Monday) thru 6 (Sunday)

weekdayList = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df.groupby('device').size().sort_values(ascending=False).plot(kind='bar')
df.groupby('device').publisher_split.sum().sort_values(ascending=False).plot(kind='bar')
df.groupby('device').conversions.sum().sort_values(ascending=False).plot(kind='bar')
ds=df.groupby('device')['impressions','referrals','conversions','publisher_split'].sum()

ds['r/i']=(ds['referrals']/ds['impressions']).replace(np.nan,0)

ds['c/r']=(ds['conversions']/ds['referrals']).replace(np.nan,0)

ds['p/c']=(ds['publisher_split']/ds['conversions']).replace(np.nan,0)

ds['r/i']=ds['r/i'].replace(np.inf,0)

ds['c/r']=ds['c/r'].replace(np.inf,0)

ds['p/c']=ds['p/c'].replace(np.inf,0)

ds.head()
ax=ds.plot(kind='bar',y='conversions')

ds.plot(kind='line',y='p/c',color='red',secondary_y=True, ax=ax)
ax=ds.plot(kind='bar',y='r/i')

ds.plot(kind='line',y='c/r',color='red',secondary_y=True, ax=ax)
df['r/i']=(df['referrals']/df['impressions']).replace(np.nan,0)

df['c/r']=(df['conversions']/df['referrals']).replace(np.nan,0)

df['p/c']=(df['publisher_split']/df['conversions']).replace(np.nan,0)

df['r/i']=df['r/i'].replace(np.inf,0)

df['c/r']=df['c/r'].replace(np.inf,0)

df['p/c']=df['p/c'].replace(np.inf,0)

df.head()
df.groupby('Weekday')['conversions'].sum().plot(kind='line')
df.groupby(['Weekday','device']).conversions.sum().unstack()
df.groupby(['Weekday','device']).conversions.sum().unstack('device').plot()
dwc=df.groupby(['Weekday','creative_id']).conversions.sum().unstack('creative_id')

idcx=dwc.sum(axis=0).sort_values(ascending=False).head(5).index

dwc[idcx].head()
dwc[idcx].plot()
dwa=df.groupby(['Weekday','ad_unit_id']).conversions.sum().unstack('ad_unit_id')

idax=dwa.sum(axis=0).sort_values(ascending=False).head(5).index

dwa[idax].plot()
dc=df.groupby('creative_id')['impressions','referrals','conversions'].sum()

dc['r/i']=(dc['referrals']/dc['impressions']).replace(np.nan,0)

dc['c/r']=(dc['conversions']/dc['referrals']).replace(np.nan,0)

dc=dc.reset_index()

dc=dc.sort_values(by='conversions',ascending=False)

dc=dc.loc[dc['conversions']>0]

dc.head()
dc.plot(kind='bar',x='creative_id',y='conversions')
dc=dc.sort_values(by='c/r',ascending=False)

dc.plot(kind='bar',x='creative_id',y='c/r')
da=df.groupby('ad_unit_id')['impressions','referrals','conversions'].sum()

da['r/i']=(da['referrals']/da['impressions']).replace(np.nan,0)

da['c/r']=(da['conversions']/da['referrals']).replace(np.nan,0)

da=da.reset_index()

da=da.sort_values(by='conversions',ascending=False)

da=da.loc[da['conversions']>50]

da.head()
da.plot(kind='bar',x='ad_unit_id',y='conversions')
da=da.sort_values(by='c/r',ascending=False)

da.plot(kind='bar',x='ad_unit_id',y='c/r')
dset=['Desktop','Mobile','Other','Tablet']

for i in dset:

    df[i]=np.where(df['device']==i, 1, 0)

df.head()
dt=df.groupby('date')['impressions','referrals','conversions'].sum()

dt.head()
ax=dt.plot(kind='line',y='conversions')

dt.plot(kind='line',y='impressions',color='red',secondary_y=True, ax=ax)
ax=dt.plot(kind='line',y='conversions')

dt.plot(kind='line',y='referrals',color='red',secondary_y=True, ax=ax)
df.groupby(['date','device']).conversions.sum().unstack('device').plot()
dc=df.groupby(['date','creative_id']).conversions.sum().unstack('creative_id')

idx=dc.sum(axis=0).sort_values(ascending=False).head(5).index

dc[idx].head()
dc[idx].plot()
da=df.groupby(['date','ad_unit_id']).conversions.sum().unstack('ad_unit_id')

idx=da.sum(axis=0).sort_values(ascending=False).head(5).index

da[idx].head()
da[idx].plot()
# Check correlation of variables 

corr=df.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
df.groupby('conversions').size()
dfn=df.copy()

dfn['conversions'][dfn['conversions'] > 1] = 1

dfn.groupby('conversions').size()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(x='conversions',data=dfn)
dfn['logimpressions']=np.log(dfn['impressions'])

dfn['logreferrals']=np.log(dfn['referrals'])

dfn['logpublisher_split']=np.log(dfn['publisher_split'])

dfn.head()
def plot_distribution( df , var , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )

    facet.map( sns.kdeplot , var , shade= True )

    facet.set( xlim=( 0 , df[ var ].quantile(1) ) )

    facet.add_legend()
plot_distribution( dfn , var = 'logreferrals' , target = 'conversions'  )
plot_distribution( dfn , var = 'logimpressions' , target = 'conversions'  )


plot_distribution( dfn , var = 'logpublisher_split' , target = 'conversions'  )
dfx=df[df['conversions']>0].copy()

%matplotlib inline

dfx.boxplot(column=['impressions','referrals'])
dfx['logimpressions']=np.log(dfx['impressions'])

dfx['logreferrals']=np.log(dfx['referrals'])

dfx.head()
dfx.boxplot(column=['logimpressions','logreferrals'])
dfx['impressions'].quantile([0,0.25, 0.5,0.75, 1])
dfx['referrals'].quantile([0,0.25,0.5,0.75,1])
# filter the outliers

dfx1=dfx.loc[(dfx['impressions']>690) & (dfx['impressions']<9116.5)].copy()

dfx1.boxplot(column=['impressions'])
dfx1=dfx1.loc[(dfx1['referrals']>12) & (dfx1['referrals']< 149)].copy()

dfx1.boxplot(column=['referrals'])
from matplotlib import pyplot as plt

plt.scatter(dfx.logimpressions,dfx.logreferrals)
plt.scatter(dfx1.impressions,dfx1.referrals)
plot_distribution( dfx1 , var = 'referrals' , target = 'conversions'  )
plot_distribution( dfx , var = 'logreferrals' , target = 'conversions'  )
corr=dfx.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
# Linear Regression

import statsmodels.api as sm

x=dfx1[['impressions','referrals','publisher_split']]

y=dfx1['conversions']

model = sm.OLS(y, x).fit()

model.summary()
# Logistice Regression

dfn=df.copy()

dfn['conversions'][dfn['conversions'] > 1] = 1

dfn.groupby('conversions').size()
dfn.groupby('conversions').size().plot.bar()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics 

from sklearn.metrics import classification_report
dfn.conversions=dfn.conversions.astype(str)
dft = dfn.iloc[:,4:8].copy()

dft.head()
datatest0=dft[dft['conversions']=='0'].copy()

datatest1=dft[dft['conversions']=='1'].copy()
# Ramdomly select the same number rows from 'conversions' = 0 column

datatest0 = datatest0.sample(n=4676)
frames = [datatest0, datatest1]

datatest = pd.concat(frames)
Xl = datatest.iloc[:,0:3].values

yl = datatest.iloc[:,3].values
Xl_train, Xl_test, yl_train, yl_test = train_test_split(Xl, yl, test_size = .3, random_state=25)
LogReg = LogisticRegression()

LogReg.fit(Xl_train, yl_train)
yl_pred = LogReg.predict(Xl_test)
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(yl_test, yl_pred)

print(classification_report(yl_test, yl_pred))
# Decision Tree Model

from sklearn.tree import DecisionTreeClassifier  

classifier = DecisionTreeClassifier()  

classifier.fit(Xl_train, yl_train)  
yl_pred = classifier.predict(Xl_test) 
from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(yl_test, yl_pred))  

print(classification_report(yl_test, yl_pred))
#naive_bayes

from sklearn.naive_bayes import MultinomialNB  

from sklearn.metrics import precision_recall_curve  

from sklearn.metrics import classification_report
clf= MultinomialNB().fit(Xl_train, yl_train)

pre= clf.predict(Xl_test)

print(classification_report(yl_test, pre))