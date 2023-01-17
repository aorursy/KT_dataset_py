import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df_rossman  = pd.read_csv("../input/train.csv",dtype='unicode')

df_store     = pd.read_csv("../input/store.csv",dtype='unicode')

df_test      = pd.read_csv("../input/test.csv",dtype='unicode')
df_rossman.describe()
df_rossman.fillna(0,inplace=True)

from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()

df_rossman['StateHoliday']=le.fit_transform(df_rossman['StateHoliday'])

df_rossman.head()
df_store.head()
df_test[df_test.isnull().T.any()]

#replace Open by 1 because there is no holiday for these stores on the given dayes and dayOfWeek!=7

df_test['Open'].fillna(1,inplace=True)

df_test['StateHoliday']=le.fit_transform(df_test['StateHoliday'])

df_test[['Id','Store','DayOfWeek','Open','Promo','StateHoliday','SchoolHoliday']]=df_test[['Id','Store','DayOfWeek','Open','Promo','StateHoliday','SchoolHoliday']].astype('int')
df_test.head()
df_store.fillna(0,inplace=True)

#from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()

df_store['StoreType']=le.fit_transform(df_store['StoreType'])

df_store['Assortment']=le.fit_transform(df_store['Assortment'])

df_store.head()
df_store[['Store','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2','Promo2SinceWeek','Promo2SinceYear']]=df_store[['Store','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2','Promo2SinceWeek','Promo2SinceYear']].astype(int)



df_rossman[['Store','DayOfWeek','Sales','Customers','Open','Promo','SchoolHoliday','StateHoliday']]=df_rossman[['Store','DayOfWeek','Sales','Customers','Open','Promo','SchoolHoliday','StateHoliday']].astype(int)
df_rossman= df_rossman.merge(df_store,on='Store',how='left')





df_rossman['CompetitionOpenSinceYear'][0]
df_rossman=df_rossman.sort_values(by=['Store','Date'])

df_rossman.head()
x=sorted(list(df_test['Store'].unique()))

y=sorted(list(df_rossman['Store'].unique()))

new_stores=[i for i in x if i not in y ]

new_stores

#there is no new store in test set
df_rossman['CompetitionDistance']=df_rossman['CompetitionDistance']/1000

df_rossman['CompetitionDistance']=df_rossman['CompetitionDistance'].round()

df_rossman['CompetitionDistance']=df_rossman['CompetitionDistance'].round()

df_rossman['CompetitionDistance']=df_rossman['CompetitionDistance']*1000

df_rossman['CompetitionDistance']=df_rossman['CompetitionDistance'].astype('int')

df_rossman['CompetitionDistance'].unique()
df_rossman['store_year'] = pd.DatetimeIndex(df_rossman['Date']).year

df_rossman['store_month'] = pd.DatetimeIndex(df_rossman['Date']).month

df_rossman['store_day'] = pd.DatetimeIndex(df_rossman['Date']).day

df_rossman=df_rossman.reset_index()
df_rossman['CompetitionOpenSinceDay']=1  #Approximation

df_rossman['diff_year']=df_rossman['store_year']-df_rossman['CompetitionOpenSinceYear']

df_rossman['diff_month']=df_rossman['store_month']-df_rossman['CompetitionOpenSinceMonth']

df_rossman['diff_day']=df_rossman['store_day']-df_rossman['CompetitionOpenSinceDay']

df_rossman['diff_day'][(df_rossman['diff_year']<=0)&(df_rossman['diff_month']<0)]=0

#df_rossman['diff_month'][df_rossman['diff_year']==0]=0

#df_rossman['diff_year'][df_rossman['diff_year']<0]=0



df_rossman['competition_start_since_days']=df_rossman['diff_year']*365+df_rossman['diff_month']*30+df_rossman['CompetitionOpenSinceDay']
df_rossman['competition_start_since_days'].loc[df_rossman['competition_start_since_days']<0]=0

df_rossman['competition_start_since_days'].unique()
df_rossman['Date']= pd.to_datetime(df_rossman['Date'], errors = 'coerce')



df_rossman[['Date','Sales']][df_rossman['Store']==6].set_index('Date').plot()

#2013-12-01



print('avg_sales_before_competition:',np.mean(df_rossman['Sales'][(df_rossman['competition_start_since_days']==0)&(df_rossman['Store']==6)]))

print('avg_sales_after_competition:',np.mean(df_rossman['Sales'][(df_rossman['competition_start_since_days']>0)&(df_rossman['Store']==6)]))



#we can see what difference competition brings to sales
df_rossman[['competition_start_since_days','Sales']][(df_rossman['Store']==6)&(df_rossman['competition_start_since_days']>1)].set_index('competition_start_since_days').plot()



#There is not much pattern in this graph but we can say that sales continuously but gradualy decrease 

#as competition days increase 
#df_rossman['Date'][(df_rossman['Store']==1051)&(df_rossman['competition_start_since_days']==0)]

#date= 2015-01-01



print('avg_sales_before_competition:',np.mean(df_rossman['Sales'][(df_rossman['competition_start_since_days']==0)&(df_rossman['Store']==6)]))

print('avg_sales_after_competition(0:50 days):',np.mean(df_rossman['Sales'][(df_rossman['competition_start_since_days']>0)&(df_rossman['competition_start_since_days']<50)&(df_rossman['Store']==6)]))

print('avg_sales_after_competition(50:100 days):',np.mean(df_rossman['Sales'][(df_rossman['competition_start_since_days']>50)&(df_rossman['competition_start_since_days']<100)&(df_rossman['Store']==6)]))

print('avg_sales_after_competition(100:150 days):',np.mean(df_rossman['Sales'][(df_rossman['competition_start_since_days']>100)&(df_rossman['competition_start_since_days']<150)&(df_rossman['Store']==6)]))

print('avg_sales_after_competition(150:200 days):',np.mean(df_rossman['Sales'][(df_rossman['competition_start_since_days']>150)&(df_rossman['competition_start_since_days']<200)&(df_rossman['Store']==6)]))

print('avg_sales_after_competition(200:250 days):',np.mean(df_rossman['Sales'][(df_rossman['competition_start_since_days']>200)&(df_rossman['competition_start_since_days']<250)&(df_rossman['Store']==6)]))

print('avg_sales_after_competition(250:300 days):',np.mean(df_rossman['Sales'][(df_rossman['competition_start_since_days']>250)&(df_rossman['competition_start_since_days']<300)&(df_rossman['Store']==6)]))

print('avg_sales_after_competition(300:350 days):',np.mean(df_rossman['Sales'][(df_rossman['competition_start_since_days']>300)&(df_rossman['competition_start_since_days']<350)&(df_rossman['Store']==6)]))



#What we can see is after 50 days the sales attain equilibrium

#so we can remove continuous variable competition_start_since_days

#and replace it with a discrete variable
store_ids=df_rossman['Store'][(df_rossman['competition_start_since_days']==0)&(df_rossman['Date']=='2013-01-01')]

store_ids=store_ids.reset_index()

store_ids['flag']=1

store_ids.drop('index',inplace=True,axis=1)

df_rossman=df_rossman.merge(store_ids,on='Store',how='left')



df_rossman.fillna(0,inplace=True)

df_rossman.head()
filtered_stores= df_rossman[df_rossman['flag']==1.0]
print('avg_sales_before_competition:',np.mean(filtered_stores['Sales'][(filtered_stores['competition_start_since_days']>0)]))

print('avg_sales_after_competition:(0:50 days)',np.mean(filtered_stores['Sales'][(filtered_stores['competition_start_since_days']>0)&(filtered_stores['competition_start_since_days']<50)]))

print('avg_sales_after_competition:(50:100 days)',np.mean(filtered_stores['Sales'][(filtered_stores['competition_start_since_days']>50)&(filtered_stores['competition_start_since_days']<100)]))

print('avg_sales_after_competition:(100:150 days)',np.mean(filtered_stores['Sales'][(filtered_stores['competition_start_since_days']>100)&(filtered_stores['competition_start_since_days']<150)]))

print('avg_sales_after_competition:(150:200 days)',np.mean(filtered_stores['Sales'][(filtered_stores['competition_start_since_days']>150)&(filtered_stores['competition_start_since_days']<200)]))



# A certain equilibrium is achieved after 50 days since start of the competition 
df_rossman['competition_start_since_days'][(df_rossman['competition_start_since_days']<=50)]=0

df_rossman['competition_start_since_days'][(df_rossman['competition_start_since_days']>50)&(df_rossman['competition_start_since_days']==0)]=1
df_rossman.columns



#columns that we need in prediction are: 
import seaborn as sns



sns.barplot(x=df_rossman['DayOfWeek'],y=df_rossman['Sales'])



#day=7 than all store are closed
df_rossman[df_rossman['StateHoliday']==1]
df_rossman=df_rossman[df_rossman['Open']==1]

df_rossman.head()
import matplotlib.pyplot as plt

#fig, (axis1,axis2,axis3,axis4) = plt.subplots(2,2,figsize=(10,12))

fig= plt.figure()

fig.set_figheight(8)

fig.set_figwidth(8)

plt.subplot(2,2,1)

sns.barplot(x=df_rossman['StoreType'],y=df_rossman['Sales'])

plt.subplot(2,2,2)

sns.barplot(x=df_rossman['Assortment'],y=df_rossman['Sales'])

plt.subplot(2,2,3)

sns.barplot(x=df_rossman['SchoolHoliday'],y=df_rossman['Sales'])

plt.subplot(2,2,4)

sns.barplot(x=df_rossman['Promo'],y=df_rossman['Sales'])
import matplotlib.pyplot as plt

fig, ((axis1,axis2),(axis3,axis4)) = plt.subplots(2,2,figsize=(10,12))

sns.barplot(x=df_rossman['StoreType'],y=df_rossman['Sales'],ax=axis1)

sns.barplot(x=df_rossman['Assortment'],y=df_rossman['Sales'],ax=axis2)

sns.barplot(x=df_rossman['SchoolHoliday'],y=df_rossman['Sales'],ax=axis3)

sns.barplot(x=df_rossman['Promo'],y=df_rossman['Sales'],ax=axis4)


df_rossman['diff']=df_rossman['diff_year']*365+df_rossman['diff_month']*30



df_rossman['diff'].head()
df_rossman.head()