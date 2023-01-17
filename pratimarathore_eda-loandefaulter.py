# Importing the warnings
import warnings
warnings.filterwarnings("ignore")
# Importing all the Libraries

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Setting the display option

pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

pd.options.display.float_format = '{:.2f}'.format
# Reading the first Data set application_data.csv.

app_data=pd.read_csv("application_data.csv",encoding='unicode_escape')
#app_data=app_data.head(10000)
# Reading the Second Data set previous_application.csv

prev_app_data=pd.read_csv("previous_application.csv",encoding='unicode_escape')
prev_app_data.head()
# Reading the third Data set columns_description.csv

col_desc=pd.read_csv("columns_description.csv",encoding='unicode_escape')
col_desc
# Getting the shape of first Data frame app_data

app_data.shape

# Getting the shape of second Data frame prev_app_data

prev_app_data.shape
# Getting the shape of third Data frame col_desc

col_desc.shape
# Getting the info of first Data frame app_data

app_data.info()
# Getting the info of second Data frame prev_app_data

prev_app_data.info()
# Getting the info of third Data frame col_desc

col_desc.info()
#Print the head of the data frame.

app_data.head()

#Print the head of the data frame.

prev_app_data.head()
col_desc.head()
# Checking for duplicate records in the dataframe.

app_data.drop_duplicates()
app_data.shape

# There are no duplicate in app_data records.
# Checking for duplicate records in the dataframe.

prev_app_data.drop_duplicates()
prev_app_data.shape

# There are no duplicate in prev_app_data records.
# Checking for duplicate records in the dataframe.

col_desc.drop_duplicates()
col_desc.shape

# There are no duplicate in col_desc records.
# Remove columns where number of unique value is only 1

app_data_unique = app_data.nunique()
app_data_unique = app_data_unique[app_data_unique.values == 1]

app_data_unique

# No column with Single unique value
total = app_data.isnull().sum().sort_values(ascending = False)
percent = (app_data.isnull().sum()/app_data.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# verifying the null value of app_data data frame

app_data_null = app_data.isnull().sum()
app_data_null = app_data_null[app_data_null.values >0]
app_data_null

plt.figure(figsize=(20,4))
app_data_null.plot(kind='bar')
plt.title('Columns with Null values ')
plt.show()
app_data.shape
# Dropping all the columns with more than 60% Null Value.

def percentage(df):
     return round(100*(df.isnull().sum()/df.shape[0]),4).sort_values(ascending=False) 

app_data=app_data.loc[:,percentage(app_data) < 60.0000]

# 105 columns are left behind app_data data frame

app_data.shape

# Now showing only those columns which has less than 10% null values

app_data_null_2 = app_data_null[app_data_null.values <(0.15*len(app_data))]
app_data_null_2
# Since DAYS_LAST_PHONE_CHANGE has only 1 row with null value
#Deleting null value from app_data_analysis where DAYS_LAST_PHONE_CHANGE is null (i.e 1)

app_data = app_data[~np.isnan(app_data['DAYS_LAST_PHONE_CHANGE'])]
# Since CNT_FAM_MEMBERS has only 2 row with null value
#Deleting null value from app_data_analysis where CNT_FAM_MEMBERS is null (i.e 2)

app_data = app_data[~np.isnan(app_data['CNT_FAM_MEMBERS'])]
# Since AMT_GOODS_PRICE is "For consumer loans it is the price of the goods for which the loan is given".
# AMT_GOODS_PRICE has 278 missing values.

round(app_data.AMT_GOODS_PRICE.mean(),2)
plt.figure(figsize=(15,6))
sns.boxplot(app_data['AMT_GOODS_PRICE'])
app_data['AMT_GOODS_PRICE'] = app_data['AMT_GOODS_PRICE'].fillna((app_data['AMT_GOODS_PRICE'].median()))

# As there are outliers present filling na with median.
app_data.AMT_GOODS_PRICE.isnull().sum()
# since AMT_REQ_CREDIT_BUREAU_HOUR is Number of enquiries to Credit Bureau about the client one hour before application
# It has 5 Unique values from 0-4. As it defines as enquiries to Credit Bureau which would be almost same for all the clients
#we can safely impute the null values with the most frequestly used value i.e mode of the column.

app_data.AMT_REQ_CREDIT_BUREAU_HOUR.unique()
app_data.AMT_REQ_CREDIT_BUREAU_HOUR.mode()
app_data['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(app_data['AMT_REQ_CREDIT_BUREAU_HOUR'].mode()[0], inplace=True)
app_data.AMT_REQ_CREDIT_BUREAU_DAY.isnull().sum()
# Doing the same thing for AMT_REQ_CREDIT_BUREAU_HOUR,AMT_REQ_CREDIT_BUREAU_DAY,AMT_REQ_CREDIT_BUREAU_WEEK,
# AMT_REQ_CREDIT_BUREAU_MON , AMT_REQ_CREDIT_BUREAU_QRT, AMT_REQ_CREDIT_BUREAU_YEAR

app_data['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(app_data['AMT_REQ_CREDIT_BUREAU_DAY'].mode()[0], inplace=True)
app_data['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(app_data['AMT_REQ_CREDIT_BUREAU_WEEK'].mode()[0], inplace=True)
app_data['AMT_REQ_CREDIT_BUREAU_MON'].fillna(app_data['AMT_REQ_CREDIT_BUREAU_MON'].mode()[0], inplace=True)
app_data['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(app_data['AMT_REQ_CREDIT_BUREAU_QRT'].mode()[0], inplace=True)
app_data['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(app_data['AMT_REQ_CREDIT_BUREAU_YEAR'].mode()[0], inplace=True)

app_data.AMT_REQ_CREDIT_BUREAU_DAY.isnull().sum()
app_data.AMT_REQ_CREDIT_BUREAU_WEEK.isnull().sum()
app_data.AMT_REQ_CREDIT_BUREAU_MON.isnull().sum()
app_data.AMT_REQ_CREDIT_BUREAU_QRT.isnull().sum()
app_data.AMT_REQ_CREDIT_BUREAU_YEAR.isnull().sum()
# Removing XNA from all the rows for CODE_GENDER & ORGANIZATION_TYPE

app_data.drop(app_data[app_data['CODE_GENDER']=='XNA'].index , inplace =True)
app_data.drop(app_data[app_data['ORGANIZATION_TYPE']=='XNA'].index , inplace =True)
# Out of 105 columns selecting 28 columns for out analysis.
# Creating a new Data frame with these 28 columns

app_data_analysis = app_data[['SK_ID_CURR','TARGET','NAME_CONTRACT_TYPE','CODE_GENDER','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','OCCUPATION_TYPE','CNT_FAM_MEMBERS','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','ORGANIZATION_TYPE','TOTALAREA_MODE','EMERGENCYSTATE_MODE','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','DAYS_LAST_PHONE_CHANGE']].copy()
app_data_analysis.shape
app_data_analysis.head()
app_data_analysis.dtypes
# Lets change the dtype for DAYS_LAST_PHONE_CHANGE from float64 to int64. 

app_data_analysis.DAYS_LAST_PHONE_CHANGE.astype('int64')
# Lets change the dtype for DAYS_REGISTRATION from float64 to int64. 
#But before doing that we have to check the sum of null values in this column
# Its gives 0 null Value. 

app_data_analysis.DAYS_REGISTRATION.isnull().sum()

#changing the dtype for DAYS_LAST_PHONE_CHANGE from float64 to int64. 

app_data_analysis.DAYS_REGISTRATION.astype('int64')
app_data_analysis.head()
app_data_analysis.CNT_FAM_MEMBERS.isnull().sum()
#app_data_analysis.CNT_FAM_MEMBERS.astype(int)
#app_data_analysis['CNT_FAM_MEMBERS']=round(app_data_analysis['CNT_FAM_MEMBERS'],0)
app_data_analysis['CNT_FAM_MEMBERS']=app_data_analysis['CNT_FAM_MEMBERS'].apply(lambda x:int(x))
app_data_analysis.head()
# Removing negative from DAYS_BIRTH

app_data_analysis['DAYS_BIRTH'] = app_data_analysis['DAYS_BIRTH'].astype(str).str[1:].astype(np.int64)

# Removing negative from DAYS_EMPLOYED


app_data_analysis['DAYS_EMPLOYED']= app_data_analysis['DAYS_EMPLOYED'].apply(lambda x: re.findall('\d+', str(x))[0])
app_data_analysis['DAYS_EMPLOYED'] = app_data_analysis['DAYS_EMPLOYED'].apply(lambda x: pd.to_numeric(x))


# Removing negative from DAYS_REGISTRATION


app_data_analysis['DAYS_REGISTRATION']= app_data_analysis['DAYS_REGISTRATION'].apply(lambda x: re.findall('\d+', str(x))[0])
app_data_analysis['DAYS_REGISTRATION'] = app_data_analysis['DAYS_REGISTRATION'].apply(lambda x: pd.to_numeric(x))


# Removing negative from DAYS_LAST_PHONE_CHANGE


app_data_analysis['DAYS_LAST_PHONE_CHANGE']= app_data_analysis['DAYS_LAST_PHONE_CHANGE'].apply(lambda x: re.findall('\d+', str(x))[0])
app_data_analysis['DAYS_LAST_PHONE_CHANGE'] = app_data_analysis['DAYS_LAST_PHONE_CHANGE'].apply(lambda x: pd.to_numeric(x))

app_data_analysis.head()
# Identifying wheather the column is catagorical or not.

app_data_analysis.nunique()
# Taking AMT_INCOME_TOTAL and creating a boxplot for the same to identify if we have any outlier

sns.boxplot(app_data_analysis['AMT_INCOME_TOTAL'])

# According to the boxplot it has a outlier.
app_data_analysis1 = app_data_analysis[(app_data_analysis.AMT_INCOME_TOTAL>100000000)]
app_data_analysis1.head()

# As we can see there is a value with AMT_INCOME_TOTAL  more than 100 Million. 
# And it is much more than the average value of AMT_INCOME_TOTAL which is 168797.03
# This Value should be truncated in order to give us better analysis
app_data_analysis.drop(app_data_analysis[app_data_analysis.AMT_INCOME_TOTAL > 100000000].index, inplace=True)
app_data_analysis.head()

# Again verifying the boxplot. As we can see now there is no extreme outlier.
sns.boxplot(app_data_analysis['AMT_INCOME_TOTAL'])

# Taking AMT_INCOME_TOTAL and creating a boxplot for the same to identify if we have any outlier

sns.boxplot(app_data_analysis['CNT_FAM_MEMBERS'])

# According to the boxplot it has a outlier.
app_data_analysis.CNT_FAM_MEMBERS.unique()
app_data_analysis2 = app_data_analysis[(app_data_analysis.CNT_FAM_MEMBERS>16)]
app_data_analysis2.head()

#There are two rows with CNT_FAM_MEMBERS value as 20. 
#Since it is a outlier, we can cap this outlier with the 99th percentile value.
percentiles = app_data_analysis['CNT_FAM_MEMBERS'].quantile([.99]).values

app_data_analysis['CNT_FAM_MEMBERS'][app_data_analysis['CNT_FAM_MEMBERS'] > 16] = percentiles[0]
app_data_analysis.CNT_FAM_MEMBERS.unique()

# As we can see we have capped the CNT_FAM_MEMBERS with 99th percentile value . 
# Value 20 has been removed.
# Taking AMT_INCOME_TOTAL and creating a boxplot for the same to identify if we have any outlier

sns.boxplot(app_data_analysis['AMT_ANNUITY'])

# According to the boxplot it has a outlier.
app_data['AMT_ANNUITY'].describe()
app_data['AMT_ANNUITY'].quantile([0.25,0.5, 0.75, 0.90, 0.95, 0.99])
#This column has outliers along with missing values (12).

#Fixing Outliers 

Q1 = app_data_analysis['AMT_ANNUITY'].quantile(0.25)
Q3 = app_data_analysis['AMT_ANNUITY'].quantile(0.75)
IQR = Q3 - Q1
app_data_analysis = app_data_analysis.loc[(app_data_analysis['AMT_ANNUITY'] >  (Q1 - 1.5 * IQR))
                                             & (app_data_analysis['AMT_ANNUITY'] < (Q3 + 1.5 * IQR))]

sns.boxplot(app_data_analysis['AMT_ANNUITY'])

# As we can see from boxplot, outlier has been removed.
# Afer handeling the outliers ,treating the missing values present in the column 

app_data_analysis['AMT_ANNUITY'].describe()
sns.distplot(app_data_analysis['AMT_ANNUITY'])
# on observing the above disribution , median seems to be a beter soluion then mean 
app_data_analysis['AMT_ANNUITY'] = app_data_analysis['AMT_ANNUITY'].fillna((app_data_analysis['AMT_ANNUITY'].median()))
app_data_analysis['AMT_ANNUITY'].isnull().sum()
# Now we can see , the disribution remains the same after immuting null values with median
app_data_analysis['AMT_ANNUITY'].describe()
# Taking DAYS_EMPLOYED and creating a boxplot for the same to identify if we have any outlier.
# DAYS_EMPLOYED is How many days before the application the person started current employment
plt.figure(figsize=(20,4))

sns.boxplot(app_data_analysis['DAYS_EMPLOYED'])

# According to the boxplot it has a outlier.
app_data_analysis3 = app_data_analysis[(app_data_analysis.DAYS_EMPLOYED>20000) & (app_data_analysis.DAYS_EMPLOYED<360000)]
app_data_analysis3.shape

# As we can see there is no value for DAYS_EMPLOYED between 20000 days & 360000 days. 
# And there are 54486 values for more than 360000 days. 
# Since 360000 days makes approx 986 years, which is impossible for anyone to be employed. 
# So binning the DAYS_EMPLOYED column.
app_data_analysis['DAYS_EMPLOYED_BINS'] = pd.cut(x=app_data_analysis['DAYS_EMPLOYED'], bins=[0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,370000],labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])

app_data_analysis.head()
# So now all the values above 20000 days would fall in one bin with label 21.
# Checking the percentage of imbalance
100*(app_data_analysis['TARGET'].value_counts()/len(app_data))
#Target variable (1 - client with payment difficulties: he/she had late payment more than X days on at 
#least one of the first Y installments of the loan in our sample, 0 - all other cases)

plt.figure(figsize = (6,6))
plt.title('Target Variable in App_Data')
sns.countplot(app_data_analysis['TARGET'])

# So as we can see from barplot, approximately 275000 values are for Target 0 while only 25000 fro Target 1.
plt.figure(figsize = (6,4))
app_data_analysis['TARGET'].value_counts().plot.pie(autopct= " %1.0f%%",startangle=90)
# Creating 2 Data Frames for each Target Value

#Target0 = app_data.where[app_data['TARGET']==0,inplace = True)
#Target1 = app_data.where(app_data['TARGET']==1,inplace = True)
Target0 = app_data_analysis[app_data_analysis['TARGET']==0]                  
Target1 = app_data_analysis[app_data_analysis['TARGET']==1]                  

# Removing Target column from each of these two data frames.

del Target0['TARGET']
del Target1['TARGET']
Target0.head()
Target1.head()

plt.figure(figsize=(10, 20), dpi=100, facecolor='w', edgecolor='k') 
plt.subplot( 4,3, 1)
plt.title('Gender ratio of non defaulter -Target 0')
Target0['CODE_GENDER'].value_counts().plot.pie(autopct= " %1.0f%%",startangle=90)
plt.subplot( 4,3, 3)
Target1['CODE_GENDER'].value_counts().plot.pie(autopct= " %1.0f%%",startangle=90)
plt.title('Gender ratio of defaulter- Target 1')
plt.figure(figsize=(8,4))
plt.title("Distribution of AMT CREDIT for non-defaulter - Target 0")
ax = sns.distplot(Target0["AMT_CREDIT"])

plt.figure(figsize=(8,4))
plt.title("Distribution of AMT CREDIT for defaulter - Target 1")
ax = sns.distplot(Target1["AMT_CREDIT"])


plt.figure(figsize=(8,5))
Target0['AMT_ANNUITY'].plot.hist()
plt.title("Distribution of AMT_ANNUITY for non-defaulter - Target 0")

plt.figure(figsize=(8,5))
Target1['AMT_ANNUITY'].plot.hist()
plt.title("Distribution of AMT_ANNUITY for defaulter - Target 1")
plt.figure(figsize = (10,6))
plt.title('NAME_EDUCATION_TYPE for non-defaulter - Target 0')
sns.countplot(Target0['NAME_EDUCATION_TYPE'])

plt.figure(figsize = (10,6))
plt.title('NAME_EDUCATION_TYPE for defaulter - Target 1')
sns.countplot(Target1['NAME_EDUCATION_TYPE'])


plt.figure(figsize = (10,4))
plt.title('NAME_FAMILY_STATUS for non-defaulter - Target 0')
sns.countplot(Target0['NAME_FAMILY_STATUS'])



plt.figure(figsize = (10,4))
plt.title('NAME_FAMILY_STATUS for defaulter - Target 1')
sns.countplot(Target1['NAME_FAMILY_STATUS'])

plt.figure(figsize = (10,4))
sns.catplot(y="OCCUPATION_TYPE",kind="count", data=Target0)
plt.title('OCCUPATION_TYPE for non-defaulter - Target 0')

plt.figure(figsize = (10,4))
sns.catplot(y="OCCUPATION_TYPE",kind="count", data=Target1)
plt.title('OCCUPATION_TYPE for defaulter - Target 1')

plt.figure(figsize = (20,10))
Target0['ORGANIZATION_TYPE'].dropna().value_counts().plot( x='ORGANIZATION_TYPE', kind = 'bar')
plt.title('ORGANIZATION_TYPE for non-defaulter - Target 0')

plt.figure(figsize = (10,4))
Target1['ORGANIZATION_TYPE'].value_counts().plot( x='ORGANIZATION_TYPE', kind = 'bar')
plt.title('ORGANIZATION_TYPE for non-defaulter - Target 1')


g=sns.catplot(x="NAME_INCOME_TYPE", y="AMT_INCOME_TOTAL", hue="TARGET",
            kind="bar", data=app_data_analysis,palette="Set3");
g.fig.set_figwidth(15)
g=sns.catplot(x="NAME_EDUCATION_TYPE", y="AMT_CREDIT", hue="TARGET",kind="violin",
             data=app_data_analysis);
g.fig.set_figwidth(15)

plt.figure(figsize=(15,6))
sns.catplot(x="NAME_FAMILY_STATUS", y="AMT_ANNUITY", hue="TARGET",
            kind="bar",palette="YlGnBu", data=app_data_analysis)

df1= app_data[['CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE','CODE_GENDER','NAME_CONTRACT_TYPE','NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 
               'NAME_HOUSING_TYPE','NAME_FAMILY_STATUS','TARGET']]
T0 = df1[df1['TARGET']==0] 

T1 = df1[df1['TARGET']==1] 

del T0['TARGET']
del T1['TARGET']

corr=T0.corr()
ax= sns.heatmap(corr,annot=True,vmin=-1, vmax=1,center=0,cmap='RdYlGn')
plt.title('Correlation for non-defaulter')

corr=T1.corr()
ax= sns.heatmap(corr,annot=True,vmin=-1, vmax=1,center=0,cmap='RdYlGn')
plt.title('Correlation for defaulter')

sns.pairplot(df1, hue = 'TARGET')
# Merging the data app_data_analysis and prev_app_data for our analysis on column SK_ID_CURR. 
#We would be doing an inner join to makes sure that only the SK_ID_CURR ids present in both dfs are included in the result

merge_app_prev_data = pd.merge(app_data_analysis, prev_app_data, how='inner', on='SK_ID_CURR')

# Verifying the new data frame we created.

merge_app_prev_data.head()
# Verifying the description & info of our new data frame.

merge_app_prev_data.describe()
merge_app_prev_data.shape
# Identifying wheather the column is catagorical or not.

merge_app_prev_data.nunique().sort_values()
# There is a column named CHANNEL_TYPE in merge_app_prev_data.
# Description of the CHANNEL_TYPE column say "Through which channel we acquired the client on the previous application".
# Basicly it consists of data from where we acquired this client in prev_app_data.
# So lets Calculate the percentage of each categories in the "CHANNEL_TYPE" variable.

merge_app_prev_data.CHANNEL_TYPE.value_counts(normalize=True)
# Now lets make a bar plot with its value count.
merge_app_prev_data.CHANNEL_TYPE.value_counts(normalize=True).plot.bar()
# Now let's check how much of amount has been credited to each of these Channel Types.
# Making a vertical bar plot for the same.
plt.figure(figsize=(15,6))
sns.barplot(x='CHANNEL_TYPE', y='AMT_CREDIT_y', data=merge_app_prev_data)
plt.show()


# Now let's analyse what is the type of payment dificulties for these Channel Type. 
# Whether the clients are able to make the payment or not, as based on Target column.
# plotting the heat map :
plt.figure(figsize=(15,8))

res=pd.pivot_table(data=merge_app_prev_data,index='CHANNEL_TYPE',columns='OCCUPATION_TYPE',values='TARGET')
sns.heatmap(res,annot=True,cmap='RdYlGn')
# So now lets analyse of all the clients, who takes cash loan for a particular purpose are the defaulters.
# Before doing that lets see how many unique values  are there for NAME_CASH_LOAN_PURPOSE.
merge_app_prev_data.NAME_CASH_LOAN_PURPOSE.unique()
merge_app_prev_data.OCCUPATION_TYPE.unique()
# As we can see we have lots of XAP & XNA values in our NAME_CASH_LOAN_PURPOSE. So lets change these to null values.
merge_app_prev_data.NAME_CASH_LOAN_PURPOSE.replace("XAP",np.nan,inplace=True)
merge_app_prev_data.NAME_CASH_LOAN_PURPOSE.replace("XNA",np.nan,inplace=True)

merge_app_prev_data.NAME_CASH_LOAN_PURPOSE.unique()
#Lets create a count plot for all the different purposes of applying for a loan.
merge_app_prev_data.NAME_CASH_LOAN_PURPOSE.value_counts(normalize=True)
plt.figure(figsize=(15,6))
plt.xticks(rotation=90)
sns.countplot(merge_app_prev_data['NAME_CASH_LOAN_PURPOSE'])
# Now lets make a bar plot for analysing out of all these cash loan purpose what is the amount application, 
# i.e. for each of the cash loan purposes how much credit did client ask on the previous application

plt.figure(figsize=(20,8))
plt.xticks(rotation=90)
sns.barplot(x='NAME_CASH_LOAN_PURPOSE', y='AMT_APPLICATION', data=merge_app_prev_data)


# Now let's analyse for different cash purpose who are among the biggest defaulters in paying the loan back. 
# Whether the clients are able to make the payment or not, as based on Target column.
# plotting the heat map :
plt.figure(figsize=(15,8))

res=pd.pivot_table(data=merge_app_prev_data,index='OCCUPATION_TYPE',columns='NAME_CASH_LOAN_PURPOSE',values='TARGET')
sns.heatmap(res,annot=True,cmap='RdYlGn')

# So now lets analyse of all the clients, and see what type of educations clients have.
# Before doing that lets see how many unique values  are there for NAME_EDUCATION_TYPE.

merge_app_prev_data.NAME_EDUCATION_TYPE.value_counts()
#Creating a countplot to see the dustribution of education types.
plt.figure(figsize=(15,5))
sns.countplot(merge_app_prev_data['NAME_EDUCATION_TYPE'])
# Now lets make a bar plot for analysing out of all these cash loan purpose what is the amount application, 
# i.e. for each of the cash loan purposes how much credit did client ask on the previous application

plt.figure(figsize=(15,8))
plt.yscale('log')
sns.boxplot(x='NAME_EDUCATION_TYPE', y='AMT_APPLICATION', data=merge_app_prev_data)

# So we can see from the boxplot below irrespective of the education type AMT_APPLICATION
# i.e.For how much credit did client ask on the previous application is always the same.
# Lets analyse how much amoutn has been credited based on the eduction of the clients.

plt.figure(figsize=(15,8))
plt.yscale('log')
sns.boxplot(x='NAME_EDUCATION_TYPE', y='AMT_CREDIT_x', data=merge_app_prev_data)

# It means, most of the time while providing loan the client, education is not considered considerably.
# Now let's analyse for different clients based on their education how many times they are able to pay the 
# loan due back without any difficulties.
# Whether the clients are able to make the payment or not, as based on Target column.

plt.figure(figsize=(20,8))
#plt.xticks(rotation=90)
sns.barplot(x='NAME_EDUCATION_TYPE', y='AMT_APPLICATION',hue='TARGET', data=merge_app_prev_data)

# So now lets analyse of all the clients, and see what is the status of their contract.
# Before doing that lets see how many unique values  are there for NAME_EDUCATION_TYPE.

merge_app_prev_data.NAME_CONTRACT_STATUS.value_counts()
#Creating a countplot to see the dustribution of contract status.
plt.figure(figsize=(15,5))
sns.countplot(merge_app_prev_data['NAME_CONTRACT_STATUS'])
# Now lets make a bar plot for analysing the diversion of total income & contract status.
# i.e. for each of the contract type, what is the total income.

plt.figure(figsize=(10,5))
plt.xticks(rotation=90)
sns.barplot(x='NAME_CONTRACT_STATUS', y='AMT_INCOME_TOTAL', data=merge_app_prev_data)

# Buying a new car & Buying a home even though counts to a small percentage of cash loan purpose 
# but they are the purposes for which clients had asked for highest amount.
# Now let's analyse for different clients based on their annual income how many times they are able to pay the 
# loan due back without any difficulties. who's contract have been approved or not
# Whether the clients are able to make the payment or not, as based on Target column.

plt.figure(figsize=(10,5))
#plt.xticks(rotation=90)
sns.barplot(x='NAME_CONTRACT_STATUS', y='AMT_DOWN_PAYMENT',hue ='TARGET' ,data=merge_app_prev_data)

# Let make a new data frame from merge_app_prev_data with selected columns for our analysis.

merge_app_prev_data1=merge_app_prev_data[['TARGET','AMT_INCOME_TOTAL','AMT_CREDIT_x','OCCUPATION_TYPE','AMT_APPLICATION','AMT_ANNUITY_x','AMT_DOWN_PAYMENT','DAYS_EMPLOYED','REGION_RATING_CLIENT']].copy()


merge_app_prev_data1.head()
plt.figure(figsize=(10,10))
sns.heatmap(merge_app_prev_data1.corr(), annot = True,cmap='RdYlGn')
