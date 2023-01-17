import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 
master_df = pd.read_csv("../input/startup-echosystem/startup_ecosystem_funds.csv")
master_df.head()
master_df.isnull().sum()
master_df.info()
#descarding column (SNo, Remark) for now as they may not add any value to model

fund_df = pd.DataFrame()

fund_df = master_df.drop(['SNo','Remarks'],axis=1)
fund_df.head()
fund_df.StartupName.isnull().sum()
fund_count  = fund_df.groupby('StartupName').size()
fund_count.head()
sort_count = fund_count.sort_values(ascending=False)
sort_count.head()
sort_df = sort_count.to_frame(name='count')
sort_df.head()
sort_df['StartUpName'] = sort_df.index
sort_df.head()
top_20 = sort_df.head(20)
top_20
plt.figure(figsize=(15,4))

sns.barplot(data=top_20,x='StartUpName', y='count',color='green')

plt.title('No of time startup get funds')

plt.xticks(rotation=70)
#creating sub df 

df_conti = pd.DataFrame()
fund_df.Date.isnull().sum()
fund_df['Date'] = fund_df.Date.str.replace('.',"/")

fund_df['Date'] = fund_df.Date.str.replace('//',"/")
#Lets take out year from date column to see which year has hightest amount has funded

fund_df['Date'] = pd.to_datetime(fund_df.Date)
fund_df['Year'] = fund_df['Date'].dt.year
fund_df['Year'].head()
plt.figure(figsize=(20,2))

sns.countplot(data=fund_df, y=fund_df.Year)

plt.title('Most number of funds allocated by year')
df_conti['Year'] = fund_df['Year']
fund_df.IndustryVertical.isnull().sum()
#we need to take care about missing values

fund_df.IndustryVertical.value_counts().head()
fund_df.IndustryVertical = fund_df.IndustryVertical.str.replace('eCommerce','ECommerce')
fund_df['IndustryVertical'] = fund_df['IndustryVertical'].fillna(method='ffill')
fund_df.IndustryVertical.value_counts().head()
fund_df.AmountInUSD.isnull().sum()
fund_df.AmountInUSD = fund_df.AmountInUSD.str.replace(',','')
fund_df.AmountInUSD = pd.to_numeric(fund_df.AmountInUSD)
fund_df.AmountInUSD = fund_df.AmountInUSD.fillna(fund_df.AmountInUSD.mean())
fund_df.AmountInUSD = pd.to_numeric(fund_df.AmountInUSD)
vertical_sort = fund_df.sort_values(['AmountInUSD'])

vertical_group = vertical_sort.groupby('IndustryVertical').sum()

vertical_group.sort_values(by='AmountInUSD',ascending=False,inplace=True)
top_20 = vertical_group.head(20)

top_20
plt.figure(figsize=(15,4))

sns.barplot(data=top_20,x=top_20.index, y='AmountInUSD')

plt.title('Total funding to Domains')

plt.xticks(rotation=90)
df_conti['IndustryVertical'] = fund_df['IndustryVertical']
df_conti.head()  #Sub dataframe for modeling 
fund_df.SubVertical.isnull().sum()
fund_df.SubVertical.value_counts().head(10)
fund_df['SubVertical'] = fund_df['SubVertical'].fillna(method='ffill')

fund_df.SubVertical.value_counts().head(10)
subvertical_group = vertical_sort.groupby('SubVertical').sum()

subvertical_group.sort_values(by='AmountInUSD',ascending=False,inplace=True)
top_20 = subvertical_group.head(20)

top_20.head()
plt.figure(figsize=(15,4))

sns.barplot(data=top_20,x=top_20.index, y='AmountInUSD')

plt.title('Total funding to Domains')

plt.xticks(rotation=90)
df_conti['SubVertical'] = fund_df['SubVertical']
fund_df.CityLocation.isnull().sum()
fund_df.CityLocation.value_counts().head()
fund_df['CityLocation'] = fund_df['CityLocation'].fillna('Bangalore')
fund_df.CityLocation.value_counts().head()
startup_count_by_cities  =  fund_df.CityLocation.value_counts()
cities_df = startup_count_by_cities.to_frame(name='count')

cities_df['cities'] = startup_count_by_cities.index
top_20 = cities_df.head(20)
#import squarify

#plt.figure(figsize=(15,8))

#count = fund_df['CityLocation'].value_counts()

#squarify.plot(sizes=count.values,label=count.index, value=count.values)

#plt.title('Distribution of Startups across Top cities')
plt.figure(figsize=(15,4))

sns.barplot(data=top_20,x='count', y='cities')

plt.title('Number of startups by cities')

plt.xlabel('Number of start ups')

plt.xticks(rotation=90)
df_conti['CityLocation'] = fund_df['CityLocation']
fund_df.InvestorsName.isnull().sum()
fund_df.InvestorsName = fund_df.InvestorsName.str.replace('Undisclosed investors','Undisclosed Investors')
fund_df.InvestorsName.value_counts().head()
investor_group = vertical_sort.groupby(['IndustryVertical','InvestorsName']).sum()

investor_group.sort_values(by='AmountInUSD',ascending=False,inplace=True)

investor_relation = investor_group.index

investor_relation = investor_relation.to_frame()
investor_relation.head()
fund_df.InvestmentType.isnull().sum()
fund_df['InvestmentType'] = fund_df['InvestmentType'].fillna(method='ffill')
fund_df['InvestmentType'].head()
plt.figure(figsize=(20,2))

sns.countplot(data=fund_df,y=fund_df.InvestmentType)

plt.title('Most number to funding type')
df_conti['InvestmentType'] = fund_df['InvestmentType']
df_conti.head()
fund_df.AmountInUSD.isnull().sum()
sns.distplot(fund_df.AmountInUSD)
sns.boxplot(fund_df.AmountInUSD)
fund_df.AmountInUSD.shape
fund_df[fund_df.AmountInUSD>10000000].shape
print(fund_df.AmountInUSD.mean())

print(fund_df.AmountInUSD.min())

print(fund_df.AmountInUSD.max())
fund_df.AmountInUSD.sort_values(ascending=False).head(5)
df_conti['AmountInUSD'] = fund_df['AmountInUSD']
#top 20 startups by word count
df_conti.head()
df_encod = pd.DataFrame()

df_encod = df_conti
cat = df_encod.iloc[:,0:4]
cat.head()
from sklearn.preprocessing import LabelEncoder
def one_hot_encode(df,columnName):

    top10 = df[columnName].value_counts().sort_values(ascending=False).head(10).index

    for label in top10:

        df[columnName+"_"+label] = np.where(df[columnName]==label,1,0)
one_hot_encode(cat,'IndustryVertical')

one_hot_encode(cat,'SubVertical')

one_hot_encode(cat,'CityLocation')
cat.head()
X = cat.iloc[:,4:]
X.head()
y = df_encod.iloc[:,-1]
y.head()
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import xgboost as xgb
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print('RMSE :  ',np.sqrt(mean_squared_error(y_test,y_pred)))
kn = KNeighborsRegressor(n_neighbors=5)
kn.fit(X_train, y_train)
kn.fit(X_train,y_train)
ky_pred = kn.predict(X_test)
print('RMSE :  ',np.sqrt(mean_squared_error(y_test,ky_pred)))
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print('RMSE :  ',np.sqrt(mean_squared_error(y_test,dt_pred)))
rf = RandomForestRegressor(n_estimators=200)
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)
print('RMSE :  ',np.sqrt(mean_squared_error(y_test,rf_pred)))
xgb = xgb.XGBRFRegressor(objective ='reg:linear')
xgb.fit(X_train,y_train)
xgb_pred = xgb.predict(X_test)
print('RMSE :  ',np.sqrt(mean_squared_error(y_test,xgb_pred)))