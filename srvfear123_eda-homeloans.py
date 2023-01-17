# importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pd.options.display.max_columns = 90
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
url1 = 'https://drive.google.com/file/d/1B8B6BK9T47e14BW8EWVXaRww1hdxoveZ/view?usp=sharing'
downloaded = drive.CreateFile({'id':'1B8B6BK9T47e14BW8EWVXaRww1hdxoveZ'}) 
downloaded.GetContentFile('app_data.csv')  
app_data = pd.read_csv('app_data.csv')

url2 = 'https://drive.google.com/file/d/1qw4aiFI69iO5nDoXuOzVAFRlDysWIzKx/view?usp=sharing'
downloaded = drive.CreateFile({'id':'1qw4aiFI69iO5nDoXuOzVAFRlDysWIzKx'}) 
downloaded.GetContentFile('prev_data.csv')  
prev_data = pd.read_csv('prev_data.csv')

# app_data contains all the information of the client at the time of application and prev_data contains all the past applications of the customer

app_data = pd.read_csv(r'C:\Users\Saurav\Desktop\Upgrad\Credit EDA Case study\application_data.csv')
prev_data = pd.read_csv(r'C:\Users\Saurav\Desktop\Upgrad\Credit EDA Case study\previous_application.csv')

# inspecting the columns of the app_data dataframe

app_data.columns
# inspecting the shape of the app_data dataframe

app_data.shape
# inspecting the variable types of the app_data dataframe

app_data.info()
# inspecting the datatypes of app_data dataframe

app_data.dtypes
# inspecting the central tendencies of app_data dataframe

app_data.describe()
# inspecting the percentage of missing values for all the columns in app_data dataframe
# rounding off the percentage to 3 decimal places

round(100*(app_data.isnull().sum()/len(app_data.index)),3)[:30]
round(100*(app_data.isnull().sum()/len(app_data.index)),3)[30:60]
round(100*(app_data.isnull().sum()/len(app_data.index)),3)[60:90]
round(100*(app_data.isnull().sum()/len(app_data.index)),3)[90:122]
(app_data.isnull().sum()*100/len(app_data)).sort_values(ascending=False).head(60)
# column OWN_CAR_AGE has got 65.991% of missing values
# dropping the column as the column hasn't got much use for our analysis

app_data.drop('OWN_CAR_AGE',axis = 1,inplace = True)
app_data.shape
# column EXT_SOURCE_1 has got 56.383% of missing values
# dropping the column EXT_SOURCE_1

app_data.drop('EXT_SOURCE_1',axis = 1,inplace = True)
app_data.shape
# column EXT_SOURCE_3 has got 19.840% of missing values
# dropping the column as EXT_SOURCE_3 has normalized score from external data source
# and we can analyse the data through EXT_SOURCE_2

app_data.drop('EXT_SOURCE_3',axis = 1,inplace = True)
app_data.shape
# the housing columns have got around 50% of missing values
# dropping all those columns

app_data.drop(['APARTMENTS_AVG','BASEMENTAREA_AVG','YEARS_BEGINEXPLUATATION_AVG','YEARS_BUILD_AVG','COMMONAREA_AVG','ELEVATORS_AVG','ENTRANCES_AVG','FLOORSMAX_AVG','FLOORSMIN_AVG','LANDAREA_AVG','LIVINGAPARTMENTS_AVG','LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_AVG','APARTMENTS_MODE','BASEMENTAREA_MODE','YEARS_BEGINEXPLUATATION_MODE','YEARS_BUILD_MODE','COMMONAREA_MODE','ELEVATORS_MODE','ENTRANCES_MODE','FLOORSMAX_MODE','FLOORSMIN_MODE','LANDAREA_MODE','LIVINGAPARTMENTS_MODE','LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE','NONLIVINGAREA_MODE','APARTMENTS_MEDI','BASEMENTAREA_MEDI','YEARS_BEGINEXPLUATATION_MEDI','YEARS_BUILD_MEDI','COMMONAREA_MEDI','ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI','FLOORSMIN_MEDI','LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI','LIVINGAREA_MEDI','NONLIVINGAPARTMENTS_MEDI','NONLIVINGAREA_MEDI','FONDKAPREMONT_MODE','HOUSETYPE_MODE','TOTALAREA_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE'],axis = 1,inplace = True)
app_data.shape
#We have an unknown entry in gender, lets make it null
app_data['CODE_GENDER'].unique()
app_data['CODE_GENDER'].replace('XNA',np.NaN,inplace=True)
print((app_data['CODE_GENDER'].isnull().sum()/len(app_data))*100)
# column AMT_ANNUITY has got 0.004% of missing values
# We can impute the missing values with the median of the data, since taking credits for all individuals can account massive outliers, hence the annuity payment would be affected too.
# But for now, we are dropping the rows where values are missing


app_data = app_data[~app_data['AMT_ANNUITY'].isnull()]
app_data['AMT_ANNUITY'].isnull().sum()
# column AMT_GOODS_PRICE has got 0.090% of missing values
# Again this column has massive outliers, imputing median is a good option

app_data = app_data[~app_data['AMT_GOODS_PRICE'].isnull()]
app_data['AMT_GOODS_PRICE'].isnull().sum()
# column NAME_TYPE_SUITE has got 0.420% of missing values
# We can impute the mode, ie. Unaccompanied, but again we are dropping it here

app_data = app_data[~app_data['NAME_TYPE_SUITE'].isnull()]
app_data['NAME_TYPE_SUITE'].isnull().sum()
# column OCCUPATION_TYPE has got 31.346% of missing values
# inspecting the column OCCUPATION_TYPE

app_data['OCCUPATION_TYPE'].value_counts()
# since column OCCUPATION_TYPE is useful for our analysis
# imputing the missing values as 'Self employed', #WE ARE IMPUTING because we feel that this variable needs to be present for analysis

app_data['OCCUPATION_TYPE'].fillna(value = 'Self employed',inplace = True)
app_data['OCCUPATION_TYPE'].value_counts()
# column CNT_FAM_MEMBERS has got 0.001% of missing values
# dropping the rows with the missing values

app_data = app_data[~app_data['CNT_FAM_MEMBERS'].isnull()]
app_data['CNT_FAM_MEMBERS'].isnull().sum()
# column EXT_SOURCE_2 has got 0.215% of missing values
# dropping the rows with the missing values

app_data = app_data[~app_data['EXT_SOURCE_2'].isnull()]
app_data['EXT_SOURCE_2'].isnull().sum()
#column CODE_GENDER has got 0.0013% of missing values
#dropping the rows with the missing values

app_data = app_data[~app_data['CODE_GENDER'].isnull()]
app_data['CODE_GENDER'].isnull().sum()
# column OBS_30_CNT_SOCIAL_CIRCLE has got 0.332% of missing values
# dropping the rows with the missing values

app_data = app_data[~app_data['OBS_30_CNT_SOCIAL_CIRCLE'].isnull()]
app_data['OBS_30_CNT_SOCIAL_CIRCLE'].isnull().sum()
# column DEF_30_CNT_SOCIAL_CIRCLE has got 0.332% of missing values
# dropping the rows with the missing values

app_data = app_data[~app_data['DEF_30_CNT_SOCIAL_CIRCLE'].isnull()]
app_data['DEF_30_CNT_SOCIAL_CIRCLE'].isnull().sum()
# column OBS_60_CNT_SOCIAL_CIRCLE has got 0.332% of missing values
# dropping the rows with the missing values

app_data = app_data[~app_data['OBS_60_CNT_SOCIAL_CIRCLE'].isnull()]
app_data['OBS_60_CNT_SOCIAL_CIRCLE'].isnull().sum()
# column DEF_60_CNT_SOCIAL_CIRCLE has got 0.332% of missing values
# dropping the rows with the missing values

app_data = app_data[~app_data['DEF_60_CNT_SOCIAL_CIRCLE'].isnull()]
app_data['DEF_60_CNT_SOCIAL_CIRCLE'].isnull().sum()
# column AMT_REQ_CREDIT_BUREAU_HOUR has got 13.502% of missing values
# inspecting the column AMT_REQ_CREDIT_BUREAU_HOUR

app_data['AMT_REQ_CREDIT_BUREAU_HOUR'].value_counts()
# since the most number of enquiries to Credit Bureau about the client one hour before application
# comes out to be 0, imputing the missing values with 0

app_data['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(value = 0,inplace = True)
app_data['AMT_REQ_CREDIT_BUREAU_HOUR'].isnull().sum()
# column AMT_REQ_CREDIT_BUREAU_DAY has got 13.502% of missing values
# inspecting the column AMT_REQ_CREDIT_BUREAU_DAY

app_data['AMT_REQ_CREDIT_BUREAU_DAY'].value_counts()
# since the most number of enquiries to Credit Bureau about the client one day before application
# comes out to be 0, imputing the missing values with 0

app_data['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(value = 0,inplace = True)
app_data['AMT_REQ_CREDIT_BUREAU_DAY'].isnull().sum()
# column AMT_REQ_CREDIT_BUREAU_WEEK has got 13.502% of missing values
# inspecting the column AMT_REQ_CREDIT_BUREAU_WEEK

app_data['AMT_REQ_CREDIT_BUREAU_WEEK'].value_counts()
# since the most number of enquiries to Credit Bureau about the client one week before application
# comes out to be 0, imputing the missing values with 0

app_data['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(value = 0,inplace = True)
app_data['AMT_REQ_CREDIT_BUREAU_WEEK'].isnull().sum()
# column AMT_REQ_CREDIT_BUREAU_MON has got 13.502% of missing values
# inspecting the column AMT_REQ_CREDIT_BUREAU_MON

app_data['AMT_REQ_CREDIT_BUREAU_MON'].value_counts()
# since the most number of enquiries to Credit Bureau about the client one month before application
# comes out to be 0, imputing the missing values with 0

app_data['AMT_REQ_CREDIT_BUREAU_MON'].fillna(value = 0,inplace = True)
app_data['AMT_REQ_CREDIT_BUREAU_MON'].isnull().sum()
# column AMT_REQ_CREDIT_BUREAU_QRT has got 13.502% of missing values
# inspecting the column AMT_REQ_CREDIT_BUREAU_QRT

app_data['AMT_REQ_CREDIT_BUREAU_QRT'].value_counts()
# since the most number of enquiries to Credit Bureau about the client 3 month before application
# comes out to be 0, imputing the missing values with 0

app_data['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(value = 0,inplace = True)
app_data['AMT_REQ_CREDIT_BUREAU_QRT'].isnull().sum()
# column AMT_REQ_CREDIT_BUREAU_YEAR has got 13.502% of missing values
# inspecting the column AMT_REQ_CREDIT_BUREAU_YEAR

app_data['AMT_REQ_CREDIT_BUREAU_YEAR'].value_counts()
app_data['AMT_REQ_CREDIT_BUREAU_YEAR'].median()
# since the number of times are heavily focussed around 0,1,2.. times and is skewed towards right, better to go for median
# comes out to be 1, imputing the missing values with 1

app_data['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(value = 1,inplace = True)
app_data['AMT_REQ_CREDIT_BUREAU_YEAR'].isnull().sum()
#removing the remaining missing values

app_data.dropna(inplace=True)
# inspecting the percentage of missing values for all the columns in app_data again

round(100*(app_data.isnull().sum()/len(app_data.index)),3)[:30]
round(100*(app_data.isnull().sum()/len(app_data.index)),3)[30:72]
# percentage of rows retained

round(100*(len(app_data.index)/307511),2)
# inspecting the datatypes of the columns in app_data dataframe

app_data.dtypes[:30]
app_data.dtypes[30:72]
# column DAYS_BIRTH shows client's age in days at the time of application in negative values
# changing the column DAYS_BIRTH to AGE with the respective age of clients in years

app_data['AGE'] = abs(app_data['DAYS_BIRTH'])//365
app_data.drop('DAYS_BIRTH',axis = 1,inplace = True)
app_data['AGE']
# column DAYS_EMPLOYED shows how many days before the application the person started current employment in negative values
# changing the neagtive values to positive values 

app_data['DAYS_EMPLOYED'] = abs(app_data['DAYS_EMPLOYED'])
app_data['DAYS_EMPLOYED']
# column DAYS_REGISTRATION shows how many days before the application did client change his registration in negative values
# changing the neagtive values to positive values and the datatype to int64

app_data['DAYS_REGISTRATION'] = abs(app_data['DAYS_REGISTRATION'])
app_data['DAYS_REGISTRATION'] = app_data['DAYS_REGISTRATION'].astype('int64')
app_data['DAYS_REGISTRATION']
# column DAYS_ID_PUBLISH shows how many days before the application did client change the identity document with which 
# he applied for the loan in negative values, changing the negative values to positive values

app_data['DAYS_ID_PUBLISH'] = abs(app_data['DAYS_ID_PUBLISH'])
app_data['DAYS_ID_PUBLISH']
# column FLAG_MOBIL shows whether or not the client provided mobile phone
# changing the continuous data into categorical data

app_data['FLAG_MOBIL'] = app_data['FLAG_MOBIL'].apply(lambda x: 'Y' if x == 1 else 'N')
app_data['FLAG_MOBIL'].value_counts()
# column FLAG_EMP_PHONE shows whether or not the client provided work phone
# changing the continuous data into categorical data

app_data['FLAG_EMP_PHONE'] = app_data['FLAG_EMP_PHONE'].apply(lambda x: 'Y' if x == 1 else 'N')
app_data['FLAG_EMP_PHONE'].value_counts()
# column FLAG_WORK_PHONE shows whether or not the client provided home phone
# changing the continuous data into categorical data

app_data['FLAG_WORK_PHONE'] = app_data['FLAG_WORK_PHONE'].apply(lambda x: 'Y' if x == 1 else 'N')
app_data['FLAG_WORK_PHONE'].value_counts()
# column FLAG_CONT_MOBILE shows whether or not the mobile phone reachable
# changing the continuous data into categorical data

app_data['FLAG_CONT_MOBILE'] = app_data['FLAG_CONT_MOBILE'].apply(lambda x: 'Y' if x == 1 else 'N')
app_data['FLAG_CONT_MOBILE'].value_counts()
# column FLAG_PHONE shows whether or not the client provided home phone 
# changing the continuous data into categorical data

app_data['FLAG_PHONE'] = app_data['FLAG_PHONE'].apply(lambda x: 'Y' if x == 1 else 'N')
app_data['FLAG_PHONE'].value_counts()
# column FLAG_EMAIL shows whether or not the client provided email 
# changing the continuous data into categorical data

app_data['FLAG_EMAIL'] = app_data['FLAG_EMAIL'].apply(lambda x: 'Y' if x == 1 else 'N')
app_data['FLAG_EMAIL'].value_counts()
# column REG_REGION_NOT_LIVE_REGION shows whether or not the client's permanent address matches contact address (at region level)
# changing the continuous data into categorical data

app_data['REG_REGION_NOT_LIVE_REGION'] = app_data['REG_REGION_NOT_LIVE_REGION'].apply(lambda x: 'Different' if x == 1 else 'Same')
app_data['REG_REGION_NOT_LIVE_REGION'].value_counts()
# column REG_REGION_NOT_WORK_REGION shows whether or not the client's permanent address matches work address (at region level)
# changing the continuous data into categorical data

app_data['REG_REGION_NOT_WORK_REGION'] = app_data['REG_REGION_NOT_WORK_REGION'].apply(lambda x: 'Different' if x == 1 else 'Same')
app_data['REG_REGION_NOT_WORK_REGION'].value_counts()
# column LIVE_REGION_NOT_WORK_REGION shows whether or not the client's contact address matches work address (at region level)
# changing the continuous data into categorical data

app_data['LIVE_REGION_NOT_WORK_REGION'] = app_data['LIVE_REGION_NOT_WORK_REGION'].apply(lambda x: 'Different' if x == 1 else 'Same')
app_data['LIVE_REGION_NOT_WORK_REGION'].value_counts()
# column REG_CITY_NOT_LIVE_CITY shows whether or not the client's permanent address matches contact address (at city level)
# changing the continuous data into categorical data

app_data['REG_CITY_NOT_LIVE_CITY'] = app_data['REG_CITY_NOT_LIVE_CITY'].apply(lambda x: 'Different' if x == 1 else 'Same')
app_data['REG_CITY_NOT_LIVE_CITY'].value_counts()
# column REG_CITY_NOT_WORK_CITY shows whether or not the client's permanent address matches work address (at city level)
# changing the continuous data into categorical data

app_data['REG_CITY_NOT_WORK_CITY'] = app_data['REG_CITY_NOT_WORK_CITY'].apply(lambda x: 'Different' if x == 1 else 'Same')
app_data['REG_CITY_NOT_WORK_CITY'].value_counts()
# column LIVE_CITY_NOT_WORK_CITY shows whether or not the client's contact address matches work address (at city level)
# changing the continuous data into categorical data

app_data['LIVE_CITY_NOT_WORK_CITY'] = app_data['LIVE_CITY_NOT_WORK_CITY'].apply(lambda x: 'Different' if x == 1 else 'Same')
app_data['LIVE_CITY_NOT_WORK_CITY'].value_counts()
# column DAYS_LAST_PHONE_CHANGE shows how many days before application did client change phone in negative values
# changing the neagtive values to positive values and the datatype to int64

app_data['DAYS_LAST_PHONE_CHANGE'] = abs(app_data['DAYS_LAST_PHONE_CHANGE'])
app_data['DAYS_LAST_PHONE_CHANGE'] = app_data['DAYS_LAST_PHONE_CHANGE'].astype('int64')
app_data['DAYS_LAST_PHONE_CHANGE']
# columns FLAG_DOCUMENTs show whether or not client provided the document and there are 20 such columns 
# changing the continuous data into categorical data

for i in range (2,22):
    app_data['FLAG_DOCUMENT_{}'.format(i)] = app_data['FLAG_DOCUMENT_{}'.format(i)].apply(lambda x: 'N' if x == 1 else 'Y')
# column AMT_REQ_CREDIT_BUREAU_HOUR shows the number of enquiries to Credit Bureau about the client one hour before application
# changing the datatype to int64

app_data['AMT_REQ_CREDIT_BUREAU_HOUR'] = app_data['AMT_REQ_CREDIT_BUREAU_HOUR'].astype('int64')
# column AMT_REQ_CREDIT_BUREAU_DAY shows the number of enquiries to Credit Bureau about the client one day before application
# changing the datatype to int64

app_data['AMT_REQ_CREDIT_BUREAU_DAY'] = app_data['AMT_REQ_CREDIT_BUREAU_DAY'].astype('int64')
# column AMT_REQ_CREDIT_BUREAU_WEEK shows the number of enquiries to Credit Bureau about the client one week before application
# changing the datatype to int64

app_data['AMT_REQ_CREDIT_BUREAU_WEEK'] = app_data['AMT_REQ_CREDIT_BUREAU_WEEK'].astype('int64')
# column AMT_REQ_CREDIT_BUREAU_MON shows the number of enquiries to Credit Bureau about the client one month before application
# changing the datatype to int64

app_data['AMT_REQ_CREDIT_BUREAU_MON'] = app_data['AMT_REQ_CREDIT_BUREAU_MON'].astype('int64')
# column AMT_REQ_CREDIT_BUREAU_QRT shows the number of enquiries to Credit Bureau about the client 3 months before application
# changing the datatype to int64

app_data['AMT_REQ_CREDIT_BUREAU_QRT'] = app_data['AMT_REQ_CREDIT_BUREAU_QRT'].astype('int64')
# column AMT_REQ_CREDIT_BUREAU_YEAR shows the number of enquiries to Credit Bureau about the client year before application
# changing the datatype to int64

app_data['AMT_REQ_CREDIT_BUREAU_YEAR'] = app_data['AMT_REQ_CREDIT_BUREAU_YEAR'].astype('int64')
# analysing the central tendencies of column AMT_INCOME_TOTAL

app_data['AMT_INCOME_TOTAL'].describe(percentiles = [0.75,0.95,0.99])
# visualizing the column AMT_INCOME_TOTAL

sns.boxplot(y = 'AMT_INCOME_TOTAL',data = app_data,palette = 'pastel')
plt.yscale('log')
# analysing the central tendencies of column AMT_CREDIT

app_data['AMT_CREDIT'].describe(percentiles = [0.75,0.95,0.99])
# visualizing the column AMT_CREDIT

sns.boxplot(y = 'AMT_CREDIT',data = app_data,palette = 'pastel')
# analysing the central tendencies of column AMT_ANNUITY

app_data['AMT_ANNUITY'].describe(percentiles = [0.75,0.95,0.99])
# visualizing the column AMT_ANNUITY

sns.boxplot(y = 'AMT_ANNUITY',data = app_data,palette = 'pastel')
# analysing the central tendencies of column AMT_GOODS_PRICE

app_data['AMT_GOODS_PRICE'].describe(percentiles = [0.75,0.95,0.99])
# visualizing the column AMT_GOODS_PRICE

sns.boxplot(y = 'AMT_GOODS_PRICE',data = app_data,palette = 'pastel')
# analysing the central tendencies of column AGE

app_data['AGE'].describe(percentiles = [0.75,0.95,0.99])
# visualizing the column AGE

sns.boxplot(y = 'AGE',data = app_data,palette = 'pastel')
print(app_data['AGE'].max())
print(app_data['AGE'].min())
# binning the AGE column into categories

app_data['AGE_GROUP'] = pd.cut(app_data['AGE'],bins = [10,18,25,44,70],labels = ['Young','Young Adult','Adult','Elderly'])
app_data['AGE_GROUP'].value_counts()
# binning the HOUR_APPR_PROCESS_START column into categories

app_data['HOUR_APPR_PROCESS_START'] = pd.cut(app_data['HOUR_APPR_PROCESS_START'],bins = [0,6,12,16,20,23],labels = ['Past Midnight','Morning','Noon','Evening','Night'])
app_data['HOUR_APPR_PROCESS_START'].value_counts()
# checking the imbalance percentage in the app_dataframe

print('Target 0: {}'.format(round(100*(len(app_data[app_data['TARGET'] == 0])/len(app_data.index)),2)))
print('Target 1: {}'.format(round(100*(len(app_data[app_data['TARGET'] == 1])/len(app_data.index)),2)))
sns.heatmap(app_data.corr())
app_data.head()
# dataset with target variable as 0 (clients who repaid their loan)

target_0 = app_data[app_data['TARGET'] == 0]
target_0
# dataset with target variable as 1 (clients who defaulted on their loan)

target_1 = app_data[app_data['TARGET'] == 1]
target_1
# analysing the column NAME_CONTRACT_TYPE for target_0 dataset

sns.countplot(x = 'NAME_CONTRACT_TYPE',data = target_0,palette = 'pastel')
app_data['CODE_GENDER'].value_counts()
# analysing the column NAME_CONTRACT_TYPE for target_1 dataset

sns.countplot(x = 'NAME_CONTRACT_TYPE',data = target_1,palette = 'pastel')
# analysing the column CODE_GENDER for target_0 dataset

sns.countplot(x = 'CODE_GENDER',data = target_0,palette = 'pastel')
# analysing the column CODE_GENDER for target_1 dataset

sns.countplot(x = 'CODE_GENDER',data = target_1,palette = 'pastel')
# analysing the column FLAG_OWN_CAR for target_0 dataset

sns.countplot(x = 'FLAG_OWN_CAR',data = target_0,palette = 'pastel')
# analysing the column FLAG_OWN_CAR for target_1 dataset

sns.countplot(x = 'FLAG_OWN_CAR',data = target_1,palette = 'pastel')
# analysing the column FLAG_OWN_REALTY for target_0 dataset

sns.countplot(x = 'FLAG_OWN_REALTY',data = target_0,palette = 'pastel')
# analysing the column FLAG_OWN_REALTY for target_1 dataset

sns.countplot(x = 'FLAG_OWN_REALTY',data = target_1,palette = 'pastel')
# analysing the column CNT_CHILDREN for target_0 dataset

fig = plt.figure(figsize = (12,6))
sns.countplot(x = 'CNT_CHILDREN',data = target_0,palette = 'pastel')
# analysing the column CNT_CHILDREN for target_1 dataset

fig = plt.figure(figsize = (12,6))
sns.countplot(x = 'CNT_CHILDREN',data = target_1,palette = 'pastel')
# analysing the column NAME_INCOME_TYPE for target_0 dataset

fig = plt.figure(figsize = (12,6))
sns.countplot(x = 'NAME_INCOME_TYPE',data = target_0,palette = 'pastel')
# analysing the column NAME_INCOME_TYPE for target_1 dataset

fig = plt.figure(figsize = (12,6))
sns.countplot(x = 'NAME_INCOME_TYPE',data = target_1,palette = 'pastel')
# analysing the column NAME_EDUCATION_TYPE for target_0 dataset

fig = plt.figure(figsize = (12,6))
sns.countplot(x = 'NAME_EDUCATION_TYPE',data = target_0,palette = 'pastel')
# analysing the column NAME_EDUCATION_TYPE for target_1 dataset

fig = plt.figure(figsize = (12,6))
sns.countplot(x = 'NAME_EDUCATION_TYPE',data = target_1,palette = 'pastel')
# analysing the column NAME_FAMILY_STATUS for target_0 dataset

fig = plt.figure(figsize = (12,6))
sns.countplot(x = 'NAME_FAMILY_STATUS',data = target_0,palette = 'pastel')
# analysing the column NAME_FAMILY_STATUS for target_1 dataset

fig = plt.figure(figsize = (12,6))
sns.countplot(x = 'NAME_FAMILY_STATUS',data = target_1,palette = 'pastel')
# analysing the column NAME_HOUSING_TYPE for target_0 dataset

fig = plt.figure(figsize = (12,6))
sns.countplot(x = 'NAME_HOUSING_TYPE',data = target_0,palette = 'pastel')
# analysing the column NAME_HOUSING_TYPE for target_1 dataset

fig = plt.figure(figsize = (12,6))
sns.countplot(x = 'NAME_HOUSING_TYPE',data = target_1,palette = 'pastel')
# analysing the column FLAG_MOBIL for app_data dataset

fig = plt.figure(figsize = (12,6))
sns.countplot(x = 'FLAG_MOBIL',hue = 'TARGET',data = app_data,palette = 'pastel')
# analysing the column FLAG_EMP_PHONE for app_data dataset

fig = plt.figure(figsize = (12,6))
sns.countplot(x = 'FLAG_EMP_PHONE',hue = 'TARGET',data = app_data,palette = 'pastel')
# analysing the column FLAG_WORK_PHONE for app_data dataset

fig = plt.figure(figsize = (12,6))
sns.countplot(x = 'FLAG_WORK_PHONE',hue = 'TARGET',data = app_data,palette = 'pastel')
# analysing the column FLAG_CONT_MOBILE for app_data dataset

fig = plt.figure(figsize = (12,6))
sns.countplot(x = 'FLAG_CONT_MOBILE',hue = 'TARGET',data = app_data,palette = 'pastel')
# analysing the column FLAG_EMAIL for app_data dataset

fig = plt.figure(figsize = (12,6))
sns.countplot(x = 'FLAG_EMAIL',hue = 'TARGET',data = app_data,palette = 'pastel')
# analysing the column OCCUPATION_TYPE for target_0 dataset

fig = plt.figure(figsize = (12,10))
sns.countplot(y = 'OCCUPATION_TYPE',data = target_0,palette = 'pastel')
plt.tight_layout()
# analysing the column OCCUPATION_TYPE for target_1 dataset

fig = plt.figure(figsize = (12,10))
sns.countplot(y = 'OCCUPATION_TYPE',data = target_1,palette = 'pastel')
plt.tight_layout()
# analysing the column CNT_FAM_MEMBERS for target_0 dataset

fig = plt.figure(figsize = (12,8))
sns.countplot(x = 'CNT_FAM_MEMBERS',data = target_0,palette = 'pastel')
plt.tight_layout()
# analysing the column CNT_FAM_MEMBERS for target_1 dataset

fig = plt.figure(figsize = (12,8))
sns.countplot(x = 'CNT_FAM_MEMBERS',data = target_1,palette = 'pastel')
plt.tight_layout()
# analysing the FLAG_DOCUMENTs columns for the app_data

for i in range(2,22):
    sns.countplot(x = 'FLAG_DOCUMENT_{}'.format(i),hue = 'TARGET',data = app_data,palette = 'pastel')
    plt.show()
# analysing the column AGE for target_0

fig = plt.figure(figsize = (10,8))
sns.countplot(x = 'AGE_GROUP',data = target_0,palette = 'pastel')
plt.tight_layout()
# analysing the column AGE for target_1

fig = plt.figure(figsize = (10,8))
sns.countplot(x = 'AGE_GROUP',data = target_1,palette = 'pastel')
plt.tight_layout()
# analysing the column AMT_INCOME_TOTAL for target_0 dataset

sns.distplot(target_0['AMT_INCOME_TOTAL'])
plt.xscale('log')
# analysing the column AMT_INCOME_TOTAL for target_1 dataset

sns.distplot(target_1['AMT_INCOME_TOTAL'])
plt.xscale('log')
# analysing the column AMT_CREDIT for target_0

sns.distplot(target_0['AMT_CREDIT'])
# analysing the column AMT_CREDIT for target_1

sns.distplot(target_1['AMT_CREDIT'])
# analysing the column AMT_ANNUITY for target_0

sns.distplot(target_0['AMT_ANNUITY'])
# analysing the column AMT_ANNUITY for target_1

sns.distplot(target_1['AMT_ANNUITY'])
# analysing the column AMT_GOODS_PRICE for target_0

sns.distplot(target_0['AMT_GOODS_PRICE'])
# analysing the column AMT_GOODS_PRICE for target_1

sns.distplot(target_0['AMT_GOODS_PRICE'])
# analysing the column REGION_POPULATION_RELATIVE for target_0

sns.distplot(target_0['REGION_POPULATION_RELATIVE'])
# analysing the column REGION_POPULATION_RELATIVE for target_1

sns.distplot(target_1['REGION_POPULATION_RELATIVE'])
# analysing the column DAYS_EMPLOYED for target_0

sns.distplot(target_0['DAYS_EMPLOYED'])
# analysing the column DAYS_EMPLOYED for target_1

sns.distplot(target_1['DAYS_EMPLOYED'])
# analysing the columns AMT_INCOME_TOTAL and AMT_CREDIT for target_0

fig = plt.figure(figsize = (10,6))
sns.scatterplot(y = 'AMT_INCOME_TOTAL',x = 'AMT_CREDIT',data = target_0)
plt.tight_layout()
# analysing the columns AMT_INCOME_TOTAL and AMT_CREDIT for target_1

fig = plt.figure(figsize = (10,6))
sns.scatterplot(y = 'AMT_INCOME_TOTAL',x = 'AMT_CREDIT',data = target_1)
plt.tight_layout()
# analysing the columns AMT_INCOME_TOTAL and AMT_ANNUITY for target_0

fig = plt.figure(figsize = (10,6))
sns.scatterplot(y = 'AMT_INCOME_TOTAL',x = 'AMT_ANNUITY',data = target_0)
plt.tight_layout()
# analysing the columns AMT_INCOME_TOTAL and AMT_ANNUITY for target_1

fig = plt.figure(figsize = (10,6))
sns.scatterplot(y = 'AMT_INCOME_TOTAL',x = 'AMT_ANNUITY',data = target_1)
plt.tight_layout()
plt.figure(figsize=(40,20))
sns.set_context("paper", font_scale=5)
sns.boxplot(x='AGE_GROUP',y='AMT_CREDIT',hue='TARGET',data=app_data,whis=1.5)
plt.figure(figsize=(20,15))
sns.set_context("paper", font_scale=1.5)
sns.catplot(x="AGE_GROUP", hue="TARGET", kind="count", data=app_data)
plt.show()
plt.figure(figsize=(10,7))
sns.boxplot(x='TARGET',y='EXT_SOURCE_2',data = app_data,whis=1.5)
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x = 'NAME_HOUSING_TYPE',hue = 'TARGET',data = app_data,palette = 'pastel')
plt.show()
prev_data.head()
prev_data.info()
prev_data.shape
(prev_data.isnull().sum()*100/len(prev_data)).sort_values(ascending=False).head(60)
#dropping the columns with high percentage values

prev_data.drop(['RATE_INTEREST_PRIVILEGED','RATE_INTEREST_PRIMARY','RATE_DOWN_PAYMENT','AMT_DOWN_PAYMENT','NAME_TYPE_SUITE','DAYS_TERMINATION','NFLAG_INSURED_ON_APPROVAL','DAYS_FIRST_DUE','DAYS_FIRST_DRAWING','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE'],axis=1,inplace=True)
#Make the unknown entry as null value and then removing the rows with missing values

prev_data['NAME_CLIENT_TYPE'].replace('XNA',np.NaN,inplace=True)
prev_data['NAME_CLIENT_TYPE'].isnull().sum()
prev_data = prev_data[~prev_data['NAME_CLIENT_TYPE'].isnull()]
prev_data['NAME_CLIENT_TYPE'].isnull().sum()
prev_data.head()
plt.figure(figsize=(20,15))
sns.set_context("paper", font_scale=1)
sns.catplot(x="NAME_CONTRACT_STATUS", hue="NAME_CLIENT_TYPE", kind="count", data=prev_data)
plt.show()
new_group = prev_data[['SK_ID_CURR','SK_ID_PREV']].groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index().rename(columns={'SK_ID_PREV':'PREV_APP_COUNT'})
new_group.head()
#Now merging the data set on app_data's current ID
merged_df = app_data.merge(new_group, on =['SK_ID_CURR'], how = 'left')
merged_df.head()
#Removing the null values if present in the new column
merged_df['PREV_APP_COUNT'] = merged_df['PREV_APP_COUNT'].fillna(0)
sns.distplot(merged_df['PREV_APP_COUNT'])
#Lets aggregate the numerical variables like mean for a given single applicant. 
#P.s- Categorical variables can be pre-processed with techniques like One hot, for variables like contract_status and then merged with app_data by mean() but for now we have merged continuous data

new_group = prev_data.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
prev_columns = ['PREV_'+column if column != 'SK_ID_CURR' else column for column in new_group.columns ]
new_group.columns = prev_columns
merged_df = merged_df.merge(new_group, on =['SK_ID_CURR'], how = 'left')
merged_df.head()

(merged_df.isnull().sum()*100/len(merged_df)).sort_values(ascending=False).head(60)
#Lets drop the rows with missing values  ,current shape of dataframe = 304531, 83
merged_df.dropna(inplace=True)
#retained rows in the dataset

(len(merged_df)/304531)*100
merged_df.head()
#check if the merge is correct
#We know sk_id_curr- 100003 has applied 2 times in the past from the above dataframe, Lets check

prev_data[prev_data['SK_ID_CURR']==100003]
#turning the days into positive

merged_df['PREV_DAYS_DECISION'] = abs(merged_df['PREV_DAYS_DECISION'])
#4 is the average number of times people have applied in the past

sns.distplot(merged_df['PREV_APP_COUNT'])
sns.heatmap(merged_df.corr())
sns.catplot(x="TARGET", y="PREV_APP_COUNT",hue='CODE_GENDER',  data=merged_df)
plt.figure(figsize=(10,7))
sns.boxplot(x='TARGET',y='PREV_DAYS_DECISION',hue='CODE_GENDER',data = merged_df,whis=1.5)
plt.show()
plt.figure(figsize=(11,7))
sns.lmplot(y='PREV_AMT_CREDIT',x='AMT_CREDIT',data=merged_df,hue='CODE_GENDER',
           col='TARGET',palette='Set1')