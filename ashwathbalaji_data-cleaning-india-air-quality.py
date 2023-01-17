import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df = pd.read_csv('/kaggle/input/india-air-quality-data/data.csv' , encoding='mac_roman')
df.head()
df.info()
missing_ratio = np.round(df.isna().sum()/len(df)*100,2)

missing_ratio = pd.DataFrame(missing_ratio , columns=['Missing_Ratio'])

missing_ratio.sort_values('Missing_Ratio',ascending=False)
list(df.columns)
df.drop(['stn_code','sampling_date','pm2_5','spm'] , axis=1 , inplace=True)
cat_cols = list(df.select_dtypes(include=['object']).columns)

num_cols = list(df.select_dtypes(exclude=['object']).columns)
print('\nNumerical Columns : ' , num_cols)

print('\nCategorical Columns : ' , cat_cols)
df['date'] = pd.to_datetime(df['date'])
df['date'].isna().sum()
df = df[df['date'].isna()==False]
print('The number of missing values are : ',df['type'].isna().sum())
df['type'] = df['type'].fillna('NA')
df['type'].value_counts()
res_str='Residential|RIRUO'

ind_str = 'Industrial'

sen_str = 'Sensitive'



rro_mask = df['type'].str.contains(res_str , regex=True)

ind_mask = df['type'].str.contains(ind_str)

sen_mask = df['type'].str.contains(sen_str)
df['type'][rro_mask] = 'RRO'

df['type'][ind_mask] = 'Industrial'

df['type'][sen_mask] = 'Sensitive'
df['type'].value_counts()
print('The number of missing values are : ',df['agency'].isna().sum())
df['agency'].fillna('NA',inplace=True)
print('The number of missing values are : ',df['location_monitoring_station'].isna().sum())
df['location_monitoring_station'].fillna('NA',inplace=True)
print('The number of missing values are : ',df['location'].isna().sum())
print('The number of missing values are : ',df['state'].isna().sum())
num_cols
df['so2'].describe()
print('Distribution of SO2')

fig,ax=plt.subplots(1,2,figsize=(13,4))

sns.distplot(df['so2'].dropna() , ax=ax[0])

sns.boxplot(df['so2'].dropna() , ax=ax[1])



so2_skew = df['so2'].skew()

plt.show()

print('Skewness = ',so2_skew)
Q1=df['so2'].quantile(0.25)

Q3=df['so2'].quantile(0.75)

IQR=Q3-Q1

df=df[~((df['so2']<(Q1-1.5*IQR))|(df['so2']>(Q3+1.5*IQR)))]
print('Distribution of SO2')

sns.distplot(df['so2'].dropna())



so2_skew = df['so2'].skew()

plt.show()

print('Skewness = ',so2_skew)
print('The number of missing values in SO2 are : ' , df['so2'].isna().sum())
print('Distribution of SO2')

sns.kdeplot(df['so2'].dropna())

plt.axvline(df['so2'].mean(), color='r')

plt.axvline(df['so2'].median(), color='g')



plt.legend(['So2','Mean','Median'])

plt.show()
df1= df.copy()

df2=df.copy()
df1['so2'] = df1['so2'].fillna(df1['so2'].mean())
df2['so2'] = df2['so2'].fillna(method='ffill')
print('Distribution of SO2')



fig,ax=plt.subplots(1,2,figsize=(13,4))



sns.kdeplot(df1['so2'] , ax=ax[0])

ax[0].axvline(df1['so2'].mean(), color='r' )

ax[0].axvline(df1['so2'].median(), color='g')

ax[0].set_title('Mean Imputation')  

ax[0].legend(['So2','Mean','Median'])



sns.kdeplot(df2['so2'] , ax=ax[1])

ax[1].axvline(df2['so2'].mean(), color='r')

ax[1].axvline(df2['so2'].median(), color='g')

ax[1].set_title('Forward Fill')

ax[1].legend(['So2','Mean','Median'])

                    

                    

plt.show()
df['so2'] = df['so2'].fillna(method='ffill')
df['no2'].describe()
print('Distribution of NO2')

fig,ax=plt.subplots(1,2,figsize=(13,4))

sns.distplot(df['no2'].dropna() , ax=ax[0])

sns.boxplot(df['no2'].dropna() , ax=ax[1])

no2_skew = df['no2'].skew()

plt.show()

print('Skewness = ',no2_skew)
Q1=df['no2'].quantile(0.25)

Q3=df['no2'].quantile(0.75)

IQR=Q3-Q1

df=df[~((df['no2']<(Q1-1.5*IQR))|(df['no2']>(Q3+1.5*IQR)))]
print('Distribution of NO2')

fig,ax=plt.subplots(1,2,figsize=(13,4))

sns.distplot(df['no2'].dropna() , ax=ax[0])

sns.boxplot(df['no2'].dropna() , ax=ax[1])

no2_skew = df['no2'].skew()

plt.show()

print('Skewness = ',no2_skew)
print('The number of missing values in NO2 are : ' , df['no2'].isna().sum())
print('Distribution of NO2')

sns.kdeplot(df['no2'])

plt.axvline(df['no2'].mean(), color='r')

plt.axvline(df['no2'].median(), color='g')

plt.legend(['No2','Mean','Median'])

plt.show()
df1 = df.copy()

df2 = df.copy()
#Mean Imputation

df1['no2'] = df1['no2'].fillna(df1['no2'].mean())

#Forward Fill

df2['no2'] = df2['no2'].fillna(method='ffill')
print('Distribution of NO2')



fig,ax=plt.subplots(1,2,figsize=(13,4))



sns.kdeplot(df1['no2'] , ax=ax[0])

ax[0].axvline(df1['no2'].mean(), color='r' )

ax[0].axvline(df1['no2'].median(), color='g')

ax[0].set_title('Mean Imputation')    

ax[0].legend(['No2','Mean','Median'])



sns.kdeplot(df2['no2'] , ax=ax[1])

ax[1].axvline(df2['no2'].mean(), color='r')

ax[1].axvline(df2['no2'].median(), color='g')

ax[1].set_title('Forward Fill')

ax[1].legend(['no2','Mean','Median'])

                    

                    

plt.show()
df['no2'] = df['no2'].fillna(method='ffill')
df['rspm'].describe()
print('Distribution of RSPM')

fig,ax=plt.subplots(1,2,figsize=(13,4))

sns.distplot(df['rspm'].dropna() , ax=ax[0])

sns.boxplot(df['rspm'].dropna() , ax=ax[1])

plt.show()

print('Skewness = ',df['rspm'].skew())
Q1=df['rspm'].quantile(0.25)

Q3=df['rspm'].quantile(0.75)

IQR=Q3-Q1

df=df[~((df['rspm']<(Q1-1.5*IQR))|(df['rspm']>(Q3+1.5*IQR)))]
print('Distribution of RSPM')

fig,ax=plt.subplots(1,2,figsize=(13,4))

sns.distplot(df['rspm'].dropna() , ax=ax[0])

sns.boxplot(df['rspm'].dropna() , ax=ax[1])

plt.show()

print('Skewness = ',df['rspm'].skew())
print('The number of missing values in RSPM are : ' , df['rspm'].isna().sum())
df1 = df.copy()

df2 = df.copy()
#Mean Imputation

df1['rspm'] = df1['rspm'].fillna(df1['rspm'].mean())

#Forward Fill

df2['rspm'] = df2['rspm'].fillna(method='ffill')
print('Distribution of RSPM')



fig,ax=plt.subplots(1,2,figsize=(13,4))



sns.kdeplot(df1['rspm'] , ax=ax[0])

ax[0].axvline(df1['rspm'].mean(), color='r' )

ax[0].axvline(df1['rspm'].median(), color='g')

ax[0].set_title('Mean Imputation')    

ax[0].legend(['rspm','Mean','Median'])



sns.kdeplot(df2['rspm'] , ax=ax[1])

ax[1].axvline(df2['rspm'].mean(), color='r')

ax[1].axvline(df2['rspm'].median(), color='g')

ax[1].set_title('Forward Fill')

ax[1].legend(['rspm','Mean','Median'])

                    

                    

plt.show()
df['rspm'] = df['rspm'].fillna(method='ffill')