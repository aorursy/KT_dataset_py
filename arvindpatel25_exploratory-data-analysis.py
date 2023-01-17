import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

sns.set()
df=pd.read_csv('../input/house-price-prediction-challenge/train.csv')

df.head()
df.info()
df.describe()
# taking out city from address 
df['CITY']=df['ADDRESS'].apply(lambda x :x.split(',')[1])
df.drop('ADDRESS',1,inplace=True)
df.head()
df.isnull().sum(axis=0)
df.nunique()
col_less_20=['POSTED_BY','UNDER_CONSTRUCTION','BHK_NO.','BHK_OR_RK','READY_TO_MOVE','RESALE','CITY']





for col in col_less_20:

    print(df[col].value_counts())

    print("----------------------------------------------")
# we have so much city we can make less occuring city as Other and maharastra as other
df['CITY'].value_counts()[:11]
# we can keep top 10 and fo reaming wi will make other
col_dict_city=dict(df['CITY'].value_counts()<600)
for i in col_dict_city.keys():

    if col_dict_city[i]==True:

        df['CITY']=df['CITY'].replace(i,'Other')

# maharastra is not a city       

df['CITY']=df['CITY'].replace('Maharashtra','Other')
df['BHK_NO.'].value_counts()
# converting dtype of BHK_NO to categorical



df['BHK_NO.']=df['BHK_NO.'].astype('O')
col_dict_bhk=dict(df['BHK_NO.'].value_counts()<1000)
for i in col_dict_bhk.keys():

    if col_dict_bhk[i]==True:

        df['BHK_NO.']=df['BHK_NO.'].replace(i,'4+')
df['BHK_NO.'].value_counts()
df['CITY'].value_counts()
#dropping BHK or RK as highly skewness

df.drop('BHK_OR_RK',1,inplace=True)
# with the help of sqft and price we can drive another feature price per sqft

df['Price_per_sqft(lac)']=df['TARGET(PRICE_IN_LACS)']/df['SQUARE_FT']
df.nunique()
df.head()
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)

sns.distplot(df['TARGET(PRICE_IN_LACS)'])



plt.subplot(2,2,2)

sns.distplot(df['SQUARE_FT'])

plt.subplot(2,2,3)

sns.distplot(df['LONGITUDE'])

plt.subplot(2,2,4)

sns.distplot(df['LATITUDE'])
# With the help of interquantile technique and soft boundary so we don't tempored much data we removed outliers





cols = ['SQUARE_FT', 'TARGET(PRICE_IN_LACS)'] # one or more



Q1 = df[cols].quantile(0.05)

Q3 = df[cols].quantile(0.90)

IQR = Q3 - Q1



df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)

sns.distplot(df['TARGET(PRICE_IN_LACS)'])



plt.subplot(2,2,2)

sns.distplot(df['SQUARE_FT'])

plt.subplot(2,2,3)

sns.distplot(df['LONGITUDE'])

plt.subplot(2,2,4)

sns.distplot(df['LATITUDE'])
# lets see outliers 
plt.figure(figsize=(10,10))

sns.scatterplot(x='TARGET(PRICE_IN_LACS)',y='SQUARE_FT',hue='POSTED_BY',data=df)
# let see house price wrt city
plt.figure(figsize=(10,10))

sns.barplot(x='CITY',y='TARGET(PRICE_IN_LACS)',data=df,estimator=np.median)
plt.figure(figsize=(20,15))

sns.barplot(x='CITY',y='TARGET(PRICE_IN_LACS)',hue='BHK_NO.',data=df,estimator=np.median)
plt.figure(figsize=(10,10))

sns.barplot(x='CITY',y='SQUARE_FT',data=df,estimator=np.median)
plt.figure(figsize=(20,15))

sns.barplot(x='CITY',y='SQUARE_FT',hue='BHK_NO.',data=df,estimator=np.median)
plt.figure(figsize=(10,10))

sns.barplot(x=df['CITY'],y=df['Price_per_sqft(lac)'],estimator=np.median)
plt.figure(figsize=(15,15))



plt.subplot(2,2,1)

sns.barplot('READY_TO_MOVE','TARGET(PRICE_IN_LACS)',data=df,estimator=np.median)



plt.subplot(2,2,2)

sns.barplot('RERA','TARGET(PRICE_IN_LACS)',data=df,estimator=np.median)

plt.subplot(2,2,3)

sns.barplot('RESALE','TARGET(PRICE_IN_LACS)',data=df,estimator=np.median)

plt.subplot(2,2,4)

sns.barplot('UNDER_CONSTRUCTION','TARGET(PRICE_IN_LACS)',data=df,estimator=np.median)
plt.figure(figsize=(10,10))

corr = df.corr()

corr = corr.where(np.triu(np.ones(corr.shape), k = 1).astype(np.bool))

sns.heatmap(corr,vmin=-1,vmax=1,annot=True)
# we will try to compare with mumbai,bangalore
df_lalit=df[df['CITY']=='Lalitpur']
df_mumbai=df[df['CITY']=='Mumbai']
df_bang=df[df['CITY']=='Bangalore']
df_lalit.head()
df_mumbai.head()
df_bang.head()
df_lalit.nunique()
df_lalit.describe([.99])
plt.figure(figsize=(10,10))

plt.subplot(2,1,1)

sns.distplot(df_lalit['TARGET(PRICE_IN_LACS)'],hist=False,color='r')

sns.distplot(df_mumbai['TARGET(PRICE_IN_LACS)'],hist=False,color='g')

sns.distplot(df_bang['TARGET(PRICE_IN_LACS)'],hist=False,color='b')



plt.subplot(2,1,2)

sns.distplot(df_lalit['SQUARE_FT'],hist=False,color='r')

sns.distplot(df_mumbai['SQUARE_FT'],hist=False,color='g')

sns.distplot(df_bang['SQUARE_FT'],hist=False,color='b')
