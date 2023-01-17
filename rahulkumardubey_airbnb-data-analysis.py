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
Train=pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip')
Train
Train['signup_method'].value_counts()
Train['gender'].value_counts()
Train['date_first_booking'].dropna(inplace=True)
Train.dropna(inplace=True)
Train
Train.reset_index(drop=True,inplace=True)
Train
import matplotlib.pyplot as plt
import matplotlib as mpl

print ('Matplotlib version: ', mpl.__version__) 
mpl.style.use(['ggplot'])
Train['country_destination'].value_counts()
Train.gender.value_counts().plot(kind='bar')
Train['age'].describe()
Train[Train.age>135]['age'].describe()
Train[Train.age<18]['age'].describe()
Train.loc[Train.age<15,'age']=np.nan
Train.loc[Train.age>122,'age']=np.nan
Train.dropna(inplace=True)
Train.shape
men=sum(Train.gender=='MALE')

women=sum(Train.gender=='FEMALE')

Fav_F_Dest=Train.loc[Train.gender=='FEMALE','country_destination'].value_counts()/women *100

Fav_M_Dest=Train.loc[Train.gender=='MALE','country_destination'].value_counts()/men*100

width=0.4

Fav_F_Dest.plot(kind='bar',label='Female',color='b',position=0,width=width)

Fav_M_Dest.plot(kind='bar',label='Male',color='g',position=1,width=width)

plt.legend()
Train['country_destination'].value_counts()
import seaborn as sns

sns.distplot(Train['age'])
young=sum(Train.age<40)

old=sum(Train.age>40)

Young_dest=Train.loc[Train.age<40,'country_destination'].value_counts()/young *100

Old_dest=Train.loc[Train.age>40,'country_destination'].value_counts()/old * 100

Young_dest.plot(kind='bar',color='b',position=0,width=width,label='Younger')

Old_dest.plot(kind='bar',color='g',position=1,width=width,label='Older')

plt.legend()
Train['date_first_booking'] = pd.to_datetime(Train['date_first_booking'])

Train['date_first_booking'].dtype


dest_2014_=Train.loc[Train.date_first_booking > pd.to_datetime(20140101,format='%Y%m%d')]
dest_2014_['date_first_booking'].value_counts().plot(kind='line')
dest_2014_=dest_2014_.loc[dest_2014_.date_first_booking<pd.to_datetime(20150101,format='%Y%m%d')]
dest_2014_['date_first_booking'].value_counts().plot(kind='line')
Train['date_account_created'] = pd.to_datetime(Train['date_account_created'])

Train['date_account_created'].dtype
acc_2014=Train.loc[Train.date_account_created>pd.to_datetime(20140101,format='%Y%m%d')]
acc_2014=acc_2014.loc[acc_2014.date_account_created<pd.to_datetime(20150101,format='%Y%m%d')]
acc_2014['date_account_created'].value_counts().plot(kind='line')
acc_2014
acc_2014['date_account_created'].value_counts().plot(kind='line',label='account created')

dest_2014_['date_first_booking'].value_counts().plot(kind='line',label='bookings')

plt.legend()
dest_2013_=Train.loc[Train.date_first_booking>pd.to_datetime(20130101,format='%Y%m%d')]
dest_2013_=dest_2013_.loc[dest_2013_.date_first_booking<pd.to_datetime(20140101,format='%Y%m%d')]
acc_2013=Train.loc[Train.date_account_created>pd.to_datetime(20130101,format='%Y%m%d')]
acc_2013=acc_2013.loc[acc_2013.date_account_created<pd.to_datetime(20140101,format='%Y%m%d')]
dest_2013_['date_first_booking'].value_counts().plot(kind='line',color='b',label='bookings',figsize=(20,10))

acc_2013['date_account_created'].value_counts().plot(kind='line',color='r',label='accounts created',figsize=(20,10))

plt.legend()
Train['date_first_booking'].value_counts().plot(kind='line',color='b',label='bookings',figsize=(20,10))

Train['date_account_created'].value_counts().plot(kind='line',color='r',label='accounts created',figsize=(20,10))

plt.legend()
dest_2013_US=dest_2013_.loc[dest_2013_.country_destination=='US']

dest_2013_US
dest_2013_US['date_first_booking'].value_counts().plot(kind='line',figsize=(20,10))
dest_2013_FR=dest_2013_.loc[dest_2013_.country_destination=='FR']
dest_2013_FR['date_first_booking'].value_counts().plot(kind='line',figsize=(20,10))
dest_2013_IT=dest_2013_.loc[dest_2013_.country_destination=='IT']
dest_2013_IT['date_first_booking'].value_counts().plot(kind='line',figsize=(20,10))
dest_2013_GB=dest_2013_.loc[dest_2013_.country_destination=='GB']

dest_2013_GB['date_first_booking'].value_counts().plot(kind='line',figsize=(20,10))
dest_2013_ES=dest_2013_.loc[dest_2013_.country_destination=='ES']

dest_2013_ES['date_first_booking'].value_counts().plot(kind='line',figsize=(20,10))
dest_2013_CA=dest_2013_.loc[dest_2013_.country_destination=='CA']

dest_2013_CA['date_first_booking'].value_counts().plot(kind='line',figsize=(20,10))
dest_2013_DE=dest_2013_.loc[dest_2013_.country_destination=='DE']

dest_2013_DE['date_first_booking'].value_counts().plot(kind='line',figsize=(20,10))
dest_2013_NL=dest_2013_.loc[dest_2013_.country_destination=='NL']

dest_2013_NL['date_first_booking'].value_counts().plot(kind='line',figsize=(20,10))
dest_2013_AU=dest_2013_.loc[dest_2013_.country_destination=='AU']

dest_2013_AU['date_first_booking'].value_counts().plot(kind='line',figsize=(20,10))
dest_2013_PT=dest_2013_.loc[dest_2013_.country_destination=='PT']

dest_2013_PT['date_first_booking'].value_counts().plot(kind='line',figsize=(20,10))
x=(30,25)

dest_2013_US['date_first_booking'].value_counts().plot(kind='line',figsize=x,label='US')

dest_2013_PT['date_first_booking'].value_counts().plot(kind='line',figsize=x,label='Portugal')

dest_2013_AU['date_first_booking'].value_counts().plot(kind='line',figsize=x,label='Australia')

dest_2013_NL['date_first_booking'].value_counts().plot(kind='line',figsize=x,label='Netherlands')

dest_2013_DE['date_first_booking'].value_counts().plot(kind='line',figsize=x,label='Germany')

dest_2013_CA['date_first_booking'].value_counts().plot(kind='line',figsize=x,label='Canada')

dest_2013_ES['date_first_booking'].value_counts().plot(kind='line',figsize=x,label='Spain')

dest_2013_GB['date_first_booking'].value_counts().plot(kind='line',figsize=x,label='Great Britain')

dest_2013_IT['date_first_booking'].value_counts().plot(kind='line',figsize=x,label='Italy')

dest_2013_FR['date_first_booking'].value_counts().plot(kind='line',figsize=x,label='France')

plt.legend()