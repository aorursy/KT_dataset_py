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
#dataiteam1@gmail.com
#private to public make public,  commit

df=pd.read_csv('/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv')

df.head()

df.tail()
df.info()
df.corr()
import matplotlib.pyplot as plt
import seaborn as sns


f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(df.corr(), annot=True, linewidth=.5, fmt='.3f',ax=ax)

df.describe()
#Line Plot

df.High.plot(kind="line",color='g', label='High', linewidth='5',alpha=0.5, grid=True,Linestyle=':')
df.Close.plot(kind="line",color='r', label='Close', linewidth='1',alpha=0.5, grid=True,Linestyle='-')
plt.legend(loc='upper right')
plt.show()
df.plot(kind='scatter', x='Open', y='Close',linewidth=.1,alpha=.5, color='red')
plt.xlabel('Open Level')
plt.ylabel('Close Level')
plt.title('Scatter')

plt.show()

#Be able to use in graph, i change the names of parenthesis included  column names  


df=df.rename(columns={'Volume_(BTC)':'VolumeBTC','Volume_(Currency)':'VolumeCurrency'})



df.corr()
#Then try to reach meaningful graphs
df.VolumeBTC.plot(kind="line",color='g', label='VolBtc', linewidth='5',alpha=0.5, grid=True,Linestyle=':')
df.VolumeCurrency.plot(kind="line",color='r', label='VolCur', linewidth='1',alpha=0.5, grid=True,Linestyle='-')
plt.legend(loc='upper right')
plt.show()

df.plot(kind='scatter', x='VolumeBTC', y='VolumeCurrency',linewidth=.1,alpha=.5, color='red')
plt.xlabel('VolumeBTC')
plt.ylabel('VolumeCurrency')
plt.title('Scatter')

plt.show()

#1 more try to reach sth different
df.plot(kind='scatter', x='VolumeBTC', y='Weighted_Price',linewidth=.1,alpha=.5, color='g')
plt.xlabel('VolumeBTC')
plt.ylabel('Weighted_Price')
plt.title('Scatter')

plt.show()

#Histogram of Open Values with different distributions quantities

df.Open.plot(kind = 'hist',bins = 25)
plt.show()

df.Open.plot(kind = 'hist',bins = 50)
plt.show()


df.Open.plot(kind = 'hist',bins = 100)
plt.show()


df.Open.plot(kind = 'hist',bins = 200)
plt.show()

#Serie vs DF

serie=df['Weighted_Price']
dataf=df[['Weighted_Price']]

f1=df['Weighted_Price']>19.659499e+03
df[f1]
df[(df['Weighted_Price']>19.659499e+03) | (df['Open']<0.029659e+03)]
#Hocam burada açılış ve kapanış değerlerini kıyaslayarak artış ve azalışlarını çıkarmak istedim

flagp=0
flagn=0
flag0=0
for Open,Close in df.items():
    if (Open-Close)>0:
        flagp+=1
    elif(Open-Close)<0:
        flagn+=1
    else:
        flag0+=1
print(flagp," ", flagn," ", flag0)
        
    
