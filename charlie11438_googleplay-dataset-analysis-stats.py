import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import re

import math

sns.set()
original=pd.read_csv('../input/googleplaystore.csv')
print(original.head())

print(original.shape)

print(original.columns)

print(original.isnull().sum())
original['Rating']=original['Rating'].fillna(round(np.mean(original['Rating']),2))
original.dropna(inplace=True)
print(original.isnull().sum())

print(original.shape)
for i in range(original.index.shape[0]):

    original.iloc[i,4]=re.sub(r'[M]','',str(original.iloc[i,4]))
original.head()
for i in range(original.index.shape[0]):

    original.iloc[i,5]=re.sub(r'\D','',str(original.iloc[i,5]))
for i in range(original.index.shape[0]):

    original.iloc[i,4]=re.sub(r'[k]','000',str(original.iloc[i,4]))
for i in range(original.index.shape[0]):

    if original.iloc[i,4]=='Varies with device':

        original.iloc[i,4]=0
original['Size']=original['Size'].astype('float')

print(original['Size'].mean()/100)
for i in range(original.index.shape[0]):

    if original.iloc[i,4]==0:

        original.iloc[i,4]=round(np.mean(original['Size'])/100,1)
original['Reviews']=original['Reviews'].astype('int')

original['Installs']=original['Installs'].astype('int')
original['Price'].unique()[0:5]
for i in range(original.index.shape[0]):

    original.iloc[i,7]=re.sub('[$]','',str(original.iloc[i,7]))
original['Price']=original['Price'].astype('float')
original.tail()
original.dtypes
de1=list(stats.describe(original.loc[:,'Rating']))

de2=list(stats.describe(original.loc[:,'Reviews']))

de3=list(stats.describe(original.loc[:,'Size']))

de4=list(stats.describe(original.loc[:,'Installs']))

de5=list(stats.describe(original.loc[:,'Price']))

des=pd.DataFrame()

des['Rating']=de1

des['Reviews']=de2

des['Size']=de3

des['Installs']=de4

des['Price']=de5

x=['n','(min,max)','mean','variance','skewness','kurtosis']

des.index=x

des
def cv(data):

    standard=data.std()

    xbar=data.mean()

    return round((standard/xbar)*100,2)
print('coefficient of variation')

print('=============================')

print('Rating:%1.2f%%'%cv(original['Rating']))

print('Reviews:%1.2f%%'%cv(original['Reviews']))

print('Size:%1.2f%%'%cv(original['Size']))

print('Installs:%1.2f%%'%cv(original['Installs']))

print('Price:%1.2f%%'%cv(original['Price']))

print('=============================')
x=['Rating','Reviews','Size','Installs','Price']

z0=plt.figure(figsize=(15,35))

for i in range(len(x)):

    z0.add_subplot(6,1,i+1)

    sns.distplot(original[str(x[i])])

    plt.title('%s distribution plot'%str(x[i]))

else:

    pass

plt.show()
cateRating=pd.DataFrame(original.groupby(by='Category').mean()['Rating'])
RatingSort=cateRating.sort_values(by='Rating',ascending=False)
z1=RatingSort.plot.barh(figsize=(15,14),legend=False)

z1.get_figure()

plt.title('Category average Rating bar graph',fontsize='large')

plt.xticks(np.linspace(0,5,15))

r=np.array(RatingSort['Rating'])

x=np.arange(RatingSort.index.shape[0])

for i,j in zip(x,r):

    plt.text(j+0.07,i-0.05,'%1.2f'%j,ha='center',color='blue')

else:

    pass

plt.show()
print('sample mean:',original.loc[:,'Rating'].mean())

print()

print('Rating 95% confidence interval:')

stats.norm.interval(0.05,loc=original.loc[:,'Rating'].mean(),scale=stats.sem(original.loc[:,'Rating']))
def zstar(data):

    para=4.2

    up=data.mean()-para

    down=data.std()/math.sqrt(len(data))

    return up/down
alpha=0.05

teststats=zstar(original.loc[:,'Rating'])

zalpha=stats.norm.pdf(alpha/2)

if math.fabs(teststats)<=zalpha:

    print('|',teststats,'|','<=',zalpha)

    print('not reject H0')

else:

    print('|',teststats,'|','>',zalpha)

    print('reject H0')
x=original.loc[:,['Rating','Reviews']].corr()

r=x.iloc[1,0]
def b1(data1,data2,r):

    return r*(data2.std()/data1.std())

def b2(co1,data1,data2):

    return data2.mean()-(co1*data1.mean())
coe1=b1(original['Rating'],original['Reviews'],r)

coe2=b2(coe1,original['Rating'],original['Reviews'])

print(coe2,'+',coe1,'x')
x=np.array(original['Rating'])

y=np.array(original['Reviews'])

x1=np.linspace(np.min(x),np.max(x),1000)

y1=x1*coe1+coe2

z2=plt.figure(figsize=(15,14))

plt.scatter(x,y)

plt.plot(x1,y1)

plt.title('scatter(rating,reviews)')

plt.xlabel('Rating')

plt.ylabel('Reviews')

plt.show()
sampledata=original.loc[:,['Rating','Reviews']].sample(1000)

sampledata['predict']=sampledata['Rating']*coe1+coe2

sampledata['Residual']=sampledata['Reviews']-sampledata['predict']

sampledata.head()
stats.describe(sampledata['Residual'])