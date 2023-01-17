# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/yoga-bar2/review2.csv',parse_dates=['Date'])

df.head()
df.drop(columns='Unnamed: 0',axis=0,inplace=True)

df.drop_duplicates(keep='first',inplace=True)

df['Star'].replace('out of 5 stars', '', regex=True, inplace=True)
df['Date'].replace('Reviewed in India on', '', regex=True, inplace=True)
df['Date'].replace('January', '/01/', regex=True, inplace=True)

df['Date'].replace('February', '/02/', regex=True, inplace=True)

df['Date'].replace('March', '/03/', regex=True, inplace=True)

df['Date'].replace('April', '/04/', regex=True, inplace=True)

df['Date'].replace('May', '/05/', regex=True, inplace=True)

df['Date'].replace('June', '/06/', regex=True, inplace=True)

df['Date'].replace('July', '/07/', regex=True, inplace=True)

df['Date'].replace('August', '/08/', regex=True, inplace=True)

df['Date'].replace('September', '/09/', regex=True, inplace=True)

df['Date'].replace('October', '/10/', regex=True, inplace=True)

df['Date'].replace('November', '/11/', regex=True, inplace=True)

df['Date'].replace('December', '/12/', regex=True, inplace=True)

df.head(20)
df['Date']=pd.to_datetime(df['Date'])



type(df['Date'][0])
df=df.set_index("Date")

df=df.sort_index()

df.tail()
df['Star']=df['Star'].astype(float)
plt.style.use(['seaborn-whitegrid'])

sns.countplot(df['Star'])

plt.show()

plt.savefig('Value Counts.png')
df['Star'].describe()
df['Star'].plot()
df.describe()
df.index
filt='2019-02'

df.loc[filt]['Star'].mean()
avg_month_rating=[]

for i in range(1,13):

    filt='2019-'+str(i)

    avg_month_rating.append(df.loc[filt]['Star'].mean())

for i in range(1,10):

    filt='2020-'+str(i)

    avg_month_rating.append(df.loc[filt]['Star'].mean())    

    

avg_month_rating    
df_w=df['Star'].resample('W').mean()
plt.figure(figsize=(30,5))

df_w.plot(marker='.',color='c')

plt.title('Mean Weekly Ratings')

plt.ylabel('Stars')

plt.savefig('Weekly status.jpg',optimmize=True)
df_d=df['Star'].resample('D').mean()

df_d.plot()

plt.savefig('Daily.jpg',optimize=True)
df_M=df['Star'].resample('M').mean()

df_M.plot(marker='o',color='r')

plt.title('Mean Monthly Ratings')

plt.ylabel('Stars')

plt.savefig('Monthly avg.jpg')
plt.style.use(['seaborn-whitegrid'])

df_q=df['Star'].resample('Q').mean()

df_q.plot(marker='D',color='m',)

plt.title('Mean Quarterly Ratings')

plt.ylabel('Stars')

plt.savefig('Quarterly Report.jpg',optimmize=True)
df_sm=df['Star'].resample('SM').mean()

df_sm.plot()

plt.title('Mean Semi-Monthly Ratings')

plt.ylabel('Stars')

plt.savefig('Semi Monthly rating.jpg',optimmize=True)
df_q.plot(kind='kde')

plt.title('Distribution of Quarterly Rating(KDE) ')

plt.xlabel('Stars')

plt.savefig('KDE quaterly.jpg',optimmize=True)
df.plot(kind='kde')

plt.title('Distribution of Rating(KDE) ')

plt.xlabel('Stars')

plt.savefig('KDE anual.jpg',optimmize=True)
print(plt.style.available)
df_r=df.sample(n=175,random_state=123)

df_rm=df_r.resample('M').mean()

df_rm.plot()

plt.savefig('Random sample monthly report.jpg',optimize=True)
df.shape
df.to_excel('Result.xls')