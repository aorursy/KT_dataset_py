# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 200)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#reading the csv file

application_data = pd.read_csv('/kaggle/input/credit-card/application_data.csv')
#checking the data

application_data.head()
#checking the shape of the data

application_data.shape
#checking the info, dtypes

application_data.info(verbose=True)
application_data.describe()
# not much information is recieved through this.. we will check for percentages

application_data.isnull().sum()
#checking the percentage of null values in columns and filtering the columns with

#more than or equal to 40% NULL values.

null_data_percentage = application_data.isnull().sum()*100/len(application_data)

major_missing_data_columns = null_data_percentage[null_data_percentage>=40]

major_missing_data_columns
#dropping the above columns from dataframe for further analysis

application_data_df = application_data.drop(columns=major_missing_data_columns.index)
application_data_df.shape
len(application_data_df)
#checking if there is if NaN values in rows is greater than 50%

# we see that none of the rows have more than 50% nan values. so we will proceed with further checks.

missing_rows = application_data_df.isnull().sum(axis=1)/application_data_df.shape[1]

missing_rows[missing_rows>50]
minor_missing_data_columns = null_data_percentage[(null_data_percentage<=15) & (null_data_percentage>0)].sort_values(ascending=False)

minor_missing_data_columns
application_data_df[['AMT_REQ_CREDIT_BUREAU_YEAR','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_QRT']].info()
application_data_df[['EXT_SOURCE_3','AMT_REQ_CREDIT_BUREAU_YEAR','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_QRT']]
# since these columns are numeric type, we will chack the number of unique values each column contain.

print("AMT_REQ_CREDIT_BUREAU_YEAR unique values:", application_data_df['AMT_REQ_CREDIT_BUREAU_YEAR'].nunique())

print("AMT_REQ_CREDIT_BUREAU_MON unique values:", application_data_df['AMT_REQ_CREDIT_BUREAU_MON'].nunique())

print("AMT_REQ_CREDIT_BUREAU_WEEK unique values:", application_data_df['AMT_REQ_CREDIT_BUREAU_WEEK'].nunique())

print("AMT_REQ_CREDIT_BUREAU_DAY unique values:", application_data_df['AMT_REQ_CREDIT_BUREAU_DAY'].nunique())

print("AMT_REQ_CREDIT_BUREAU_HOUR unique values:", application_data_df['AMT_REQ_CREDIT_BUREAU_HOUR'].nunique())

print("AMT_REQ_CREDIT_BUREAU_QRT unique values:", application_data_df['AMT_REQ_CREDIT_BUREAU_QRT'].nunique())
application_data_df['AMT_REQ_CREDIT_BUREAU_YEAR'].value_counts()
application_data_df['AMT_REQ_CREDIT_BUREAU_MON'].value_counts()
application_data_df['AMT_REQ_CREDIT_BUREAU_WEEK'].value_counts()
application_data_df['AMT_REQ_CREDIT_BUREAU_HOUR'].value_counts()
application_data_df['AMT_REQ_CREDIT_BUREAU_DAY'].value_counts()
application_data_df['AMT_REQ_CREDIT_BUREAU_QRT'].value_counts()
# we will check for unwanted columns

application_data_df.columns
application_data_df.head()
# we found these cols to be unwanted/ not required so we will drop them for further analysis

unwanted_cols = ['FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE','FLAG_EMAIL',

          'REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY', 'FLAG_EMAIL','DAYS_LAST_PHONE_CHANGE',

          'FLAG_DOCUMENT_2','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION',

          'LIVE_REGION_NOT_WORK_REGION','FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4','FLAG_DOCUMENT_5',

          'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10',

          'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',

          'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18',

          'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']



application_data_df.drop(columns=unwanted_cols, inplace=True)
application_data_df.shape
#now we have the final dataframe for analysis.

# we will now dive deep into details of variables to find out insights

application_data_df.info()
# we will describe to have a better look at variables

application_data_df.describe()
application_data_df.nunique().sort_values()
# we notice that till variable ORGANIZATION_TYPE all variables are categorical

# so we will get their index and convert them to categorical columns

application_data_df.nunique().sort_values().index
categorical_cols = ['FLAG_MOBIL', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',

                    'FLAG_OWN_REALTY', 'REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY',

                    'REG_CITY_NOT_LIVE_CITY','AMT_REQ_CREDIT_BUREAU_HOUR', 'NAME_EDUCATION_TYPE',

                    'NAME_HOUSING_TYPE', 'NAME_FAMILY_STATUS', 'WEEKDAY_APPR_PROCESS_START',

                    'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'AMT_REQ_CREDIT_BUREAU_DAY',

                    'AMT_REQ_CREDIT_BUREAU_WEEK', 'DEF_60_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE',

                    'AMT_REQ_CREDIT_BUREAU_QRT', 'CNT_CHILDREN','CNT_FAM_MEMBERS','OCCUPATION_TYPE',

                    'HOUR_APPR_PROCESS_START','AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_YEAR',

                    'OBS_60_CNT_SOCIAL_CIRCLE','OBS_30_CNT_SOCIAL_CIRCLE','ORGANIZATION_TYPE']

for col in categorical_cols:

    application_data_df[col] = application_data_df[col].astype('category')
# SK_ID_CURR is int data type, but it holds id number of customers, and this variable cannot be manipulated

# so we will convert it to object datatype

application_data_df['SK_ID_CURR'] = application_data_df['SK_ID_CURR'].astype('object')
# we will check the dtypes again to confirm

application_data_df.info()
#we notice that "DAYS_BIRTH"  'DAYS_EMPLOYED' 'DAYS_REGISTRATION' 'DAYS_ID_PUBLISH' columns have negative values,which is not not possible.

# so we will try to correct this

application_data_df[['DAYS_BIRTH' ,'DAYS_EMPLOYED' ,'DAYS_REGISTRATION' ,'DAYS_ID_PUBLISH']]
application_data_df[['DAYS_BIRTH' ,'DAYS_EMPLOYED' ,'DAYS_REGISTRATION' ,'DAYS_ID_PUBLISH']].describe()
'''we will now convert negative values to positve values using abs() 

and then convert days into years for better understanding'''



days_cols = ['DAYS_BIRTH' ,'DAYS_EMPLOYED' ,'DAYS_REGISTRATION' ,'DAYS_ID_PUBLISH']

application_data_df[days_cols] = application_data_df[days_cols].abs()

application_data_df[days_cols] = application_data_df[days_cols]/365

application_data_df[days_cols].describe()
application_data_df.rename(columns={'DAYS_BIRTH':'YEARS_BIRTH' ,'DAYS_EMPLOYED':'YEARS_EMPLOYED' ,

    'DAYS_REGISTRATION':'YEARS_REGISTRATION' ,'DAYS_ID_PUBLISH':'YEARS_ID_PUBLISH'}, inplace=True)
# we will now check gendere column

application_data_df['CODE_GENDER'].value_counts()
# since gender varible contains categorical value, so we will replace XNA with F based on mode value

application_data_df.loc[application_data_df['CODE_GENDER']=='XNA', 'CODE_GENDER'] = 'F'

application_data_df['CODE_GENDER'].value_counts()
# we will now check the'AMT_INCOME_TOTAL' and 'AMT_CREDIT' variables

application_data_df[['AMT_INCOME_TOTAL', 'AMT_CREDIT']].describe()
# it is hard to determine the type of customer based on these values as these are continous.

# we will make make 2 new columns for these respectively  dividing them into categories for easy understanding

bins = [0,100000,250000,500000,750000,1000000, 1250000, 1500000, 1750000, 2000000, 2250000,2500000,

        2750000,3000000,3250000,3500000,3750000,4000000,4250000,4500000,4750000,5000000,150000000]

ranges = ['0-100000','100000-250000','250000-500000','500000-750000','750000-1000000', '1000000-1250000',

          '1250000-1500000','1500000-1750000','1750000-2000000','2000000-2250000','2250000-2500000',

          '2500000-2750000','2750000-3000000','3000000-3250000','3250000-3500000','3500000-3750000',

          '3750000-4000000','4000000-4250000','4250000-4500000','4500000-4750000','4750000-5000000',

          '5000000 and above']



application_data_df['AMT_INCOME_RANGE'] = pd.cut(application_data_df['AMT_INCOME_TOTAL'],bins,labels=ranges)

application_data_df['AMT_CREDIT_RANGE'] = pd.cut(application_data_df['AMT_CREDIT'],bins,labels=ranges)

application_data_df.head()
plt.figure(figsize=(15,6)) 

sns.countplot(data=application_data_df,x='AMT_CREDIT_RANGE', hue='CODE_GENDER')

plt.xticks(rotation=90)

plt.legend(loc='upper right')

plt.show()
application_data_df.describe()
'''this variable indiactes Number of children the client has.

as we see from the plot some values are as high as 19, which is not possible in general case scenario. 

hence an outlier'''



sns.boxplot(application_data_df['CNT_CHILDREN'])

plt.show()
'''this  varibale indictes the Income of the client.

as we can see from the plot there is one value which is too high compared to others.

hence it is an outlier.

'''



sns.boxplot(application_data_df['AMT_INCOME_TOTAL'])

plt.show()
#this output proves that it is an outlier since the person here has occupation type as labourer, and her target variable is 1. 

application_data_df[application_data_df['AMT_INCOME_TOTAL'] == application_data_df['AMT_INCOME_TOTAL'].max()]
'''

this variable indicates Credit amount of the loan

as we can see from the graph there are few outliers.

we will check these values to confirm.



'''



sns.boxplot(application_data_df['AMT_CREDIT'])

plt.show()
# as we can see from tehe values below, the AMT_CRDIT  is greater than AMT_INCOME_TOTAL in all the cases

#and then its greater than most values

application_data_df[application_data_df['AMT_CREDIT']> 3.5*1e6]
'''

this variable indicates How many years before the application the person started current employment?

as we can see from the plot below the outlier value is 1000 yrs. which makes the case for it being an outlier



'''

sns.boxplot(application_data_df['YEARS_EMPLOYED'])

plt.show() 
'''

this variable indicates the  Number of enquiries to Credit Bureau about the client 3 month

before application (excluding one month before application)

as we can see from the plot below there is one outlier.



'''

sns.boxplot(application_data_df['AMT_REQ_CREDIT_BUREAU_QRT'])

plt.show() 
#checking the distribution of target variable

sns.countplot(application_data['TARGET'])

plt.xlabel("TARGET Value")

plt.ylabel("Count of TARGET value")

plt.title("Distribution of TARGET Variable")

plt.show()
application_data_df['TARGET'].value_counts()
# creating new datadrame for target=0

appli_data_target0 = application_data_df[application_data_df['TARGET']==0]

appli_data_target0.head()
#checking the shape of new dataframe

appli_data_target0.shape
# creating new datadrame for target=0

appli_data_target1 = application_data_df[application_data_df['TARGET']==1]

appli_data_target1.head()
#checking the shape of the new dataframe

appli_data_target1.shape
# to get the ratio of appli_data_target0 : appli_data_target1

ratio = appli_data_target0.shape[0]/appli_data_target1.shape[0]

ratio
# for target variable=0

plt.figure(figsize=(12,8)) 

sns.heatmap(appli_data_target0.corr(), annot=True, cmap="coolwarm")

plt.title('Correlation matrix for target variable 0')

plt.show()
# now we need to find top 10 correlations

corr0 = appli_data_target0.corr()

corr_df0 = corr0.where(np.triu(np.ones(corr0.shape), k=1).astype(np.bool))

corr_df0 = corr_df0.unstack().reset_index().dropna(subset = [0])

corr_df0.columns = ['VAR1', 'VAR2', 'Correlation_Value']

corr_df0['Corr_abs'] = abs(corr_df0['Correlation_Value'])

corr_df0.sort_values(by = "Corr_abs", ascending =False, inplace = True)

corr_df0.head(10)
# for target variable=1

plt.figure(figsize=(12,8)) 

sns.heatmap(appli_data_target1.corr(), annot=True, cmap="coolwarm")

plt.title('Correlation matrix for target variable 1')

plt.show()
# now we need to find top 10 correlations

corr1 = appli_data_target1.corr()

corr_df1 = corr1.where(np.triu(np.ones(corr1.shape), k=1).astype(np.bool))

corr_df1 = corr_df1.unstack().reset_index().dropna(subset = [0])

corr_df1.columns = ['VAR1', 'VAR2', 'Correlation_Value']

corr_df1['Corr_abs'] = abs(corr_df1['Correlation_Value'])

corr_df1.sort_values(by = "Corr_abs", ascending =False, inplace = True)

corr_df1.head(10)
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.boxplot(appli_data_target0['YEARS_BIRTH'])

plt.title('Customer without payment difficulties')



plt.subplot(1,2,2)

ax = sns.boxplot(appli_data_target1['YEARS_BIRTH'])

plt.title('Customer with payment difficulties')

plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.boxplot(appli_data_target0[appli_data_target0['YEARS_EMPLOYED']<1000]['YEARS_EMPLOYED'])

plt.title('Customer without payment difficulties')



plt.subplot(1,2,2)

ax = sns.boxplot(appli_data_target1[appli_data_target1['YEARS_EMPLOYED']<1000]['YEARS_EMPLOYED'])

plt.title('Customer with payment difficulties')

plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.boxplot(appli_data_target0['AMT_GOODS_PRICE'])

plt.title('Customer without payment difficulties')



plt.subplot(1,2,2)

ax = sns.boxplot(appli_data_target1['AMT_GOODS_PRICE'])

plt.title('Customer with payment difficulties')

plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.boxplot(appli_data_target0['YEARS_ID_PUBLISH'])

plt.title('Customer without payment difficulties')



plt.subplot(1,2,2)

ax = sns.boxplot(appli_data_target1['YEARS_ID_PUBLISH'])

plt.title('Customer with payment difficulties')

plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.boxplot(appli_data_target0['AMT_ANNUITY'])

plt.title('Customer without payment difficulties')



plt.subplot(1,2,2)

ax = sns.boxplot(appli_data_target1['AMT_ANNUITY'])

plt.title('Customer with payment difficulties')

plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.countplot(appli_data_target0['NAME_CONTRACT_TYPE'])

plt.title('Customer without payment difficulties')



plt.subplot(1,2,2)

ax = sns.countplot(appli_data_target1['NAME_CONTRACT_TYPE'])

plt.title('Customer with payment difficulties')

plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.countplot(appli_data_target0['CODE_GENDER'])

plt.title('Customer without payment difficulties')



plt.subplot(1,2,2)

ax = sns.countplot(appli_data_target1['CODE_GENDER'])

plt.title('Customer with payment difficulties')

plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.countplot(appli_data_target0['NAME_EDUCATION_TYPE'])

plt.title('Customer without payment difficulties')

plt.xticks(rotation=90)



plt.subplot(1,2,2)

ax = sns.countplot(appli_data_target1['NAME_EDUCATION_TYPE'])

plt.title('Customer with payment difficulties')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.countplot(appli_data_target0['NAME_HOUSING_TYPE'])

plt.title('Customer without payment difficulties')

plt.xticks(rotation=90)



plt.subplot(1,2,2)

ax = sns.countplot(appli_data_target1['NAME_HOUSING_TYPE'])

plt.title('Customer with payment difficulties')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.countplot(appli_data_target0['OCCUPATION_TYPE'])

plt.title('Customer without payment difficulties')

plt.xticks(rotation=90)



plt.subplot(1,2,2)

ax = sns.countplot(appli_data_target1['OCCUPATION_TYPE'])

plt.title('Customer with payment difficulties')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.scatterplot(data=appli_data_target0[appli_data_target0['YEARS_EMPLOYED']<1000], x='YEARS_EMPLOYED',y='AMT_INCOME_TOTAL')

plt.title('Customer without payment difficulties')



plt.subplot(1,2,2)

ax = sns.scatterplot(data=appli_data_target1[appli_data_target1['YEARS_EMPLOYED']<1000], x='YEARS_EMPLOYED',y='AMT_INCOME_TOTAL')

plt.title('Customer with payment difficulties')

plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.scatterplot(data=appli_data_target0,x='AMT_CREDIT',y='AMT_GOODS_PRICE')

plt.title('Customer without payment difficulties')





plt.subplot(1,2,2)

ax = sns.scatterplot(data=appli_data_target1,x='AMT_CREDIT',y='AMT_GOODS_PRICE')

plt.title('Customer with payment difficulties')



plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.scatterplot(data=appli_data_target0,x='AMT_CREDIT',y='AMT_ANNUITY')

plt.title('Customer without payment difficulties')





plt.subplot(1,2,2)

ax = sns.scatterplot(data=appli_data_target1,x='AMT_CREDIT',y='AMT_ANNUITY')

plt.title('Customer with payment difficulties')



plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.countplot(data=appli_data_target0,x='NAME_CONTRACT_TYPE',hue='AMT_CREDIT_RANGE')

plt.title('Customer without payment difficulties')

plt.legend(loc='upper right')





plt.subplot(1,2,2)

ax = sns.countplot(data=appli_data_target1,x='NAME_CONTRACT_TYPE',hue='AMT_CREDIT_RANGE')

plt.title('Customer with payment difficulties')

plt.legend(loc='upper right')

plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.countplot(data=appli_data_target0,x='CODE_GENDER',hue='AMT_INCOME_RANGE')

plt.title('Customer without payment difficulties')

plt.legend(loc='upper right')





plt.subplot(1,2,2)

ax = sns.countplot(data=appli_data_target1,x='CODE_GENDER',hue='AMT_INCOME_RANGE')

plt.title('Customer with payment difficulties')

plt.legend(loc='upper right')

plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.boxplot(data=appli_data_target0,y='AMT_CREDIT',x='NAME_EDUCATION_TYPE')

plt.title('Customer without payment difficulties')

plt.xticks(rotation=90)



plt.subplot(1,2,2)

ax = sns.boxplot(data=appli_data_target1,y='AMT_CREDIT',x='NAME_EDUCATION_TYPE')

plt.title('Customer with payment difficulties')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.boxplot(data=appli_data_target0[appli_data_target0['AMT_INCOME_TOTAL']<5000000],y='AMT_INCOME_TOTAL',x='NAME_EDUCATION_TYPE')

plt.title('Customer without payment difficulties')

plt.xticks(rotation=90)



plt.subplot(1,2,2)

ax = sns.boxplot(data=appli_data_target1[appli_data_target1['AMT_INCOME_TOTAL']<5000000],y='AMT_INCOME_TOTAL',x='NAME_EDUCATION_TYPE')

plt.title('Customer with payment difficulties')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,8)) 



plt.subplot(1,2,1)

ax = sns.boxplot(data=appli_data_target0,y='AMT_CREDIT',x='OCCUPATION_TYPE')

plt.title('Customer without payment difficulties')

plt.xticks(rotation=90)



plt.subplot(1,2,2)

ax = sns.boxplot(data=appli_data_target1,y='AMT_CREDIT',x='OCCUPATION_TYPE')

plt.title('Customer with payment difficulties')

plt.xticks(rotation=90)

plt.show()
#reading the previous application file

previous_application_df = pd.read_csv('/kaggle/input/credit-card/previous_application.csv')
#checking the data

previous_application_df.head()
#checking the shape of the file

previous_application_df.shape
#checking the info about the file

previous_application_df.info()
#checking the percentiles, min values for the file

previous_application_df.describe()
application_data.shape
previous_application_df.shape
#merging the application_data with previous application data

all_data_df = pd.merge(left=application_data, right=previous_application_df,how='inner', on='SK_ID_CURR',suffixes='_x')
#checking the new dataframe's shape

all_data_df.shape
all_data_df.head()
# we will check the percentages of each type of contract status

all_data_df['NAME_CONTRACT_STATUS'].value_counts()*100/len(all_data_df)
sns.countplot(all_data_df['NAME_CONTRACT_STATUS'])

plt.xlabel("Contract Status")

plt.ylabel("Count of Contract Status")

plt.title("Distribution of Contract Status")

plt.show()
approved_df = all_data_df[all_data_df['NAME_CONTRACT_STATUS']=='Approved']

refused_df = all_data_df[all_data_df['NAME_CONTRACT_STATUS']=='Refused']

canceled_df = all_data_df[all_data_df['NAME_CONTRACT_STATUS']=='Canceled']

unused_df = all_data_df[all_data_df['NAME_CONTRACT_STATUS']=='Unused offer']
all_data_df['NAME_CONTRACT_TYPEx'].value_counts()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2,ncols=2, figsize=(15,10),sharey=True)



ax1 = sns.countplot(ax=ax1,data=approved_df,x='NAME_CONTRACT_TYPEx')

ax1.set_title("Refused", fontsize=10)

ax1.set_xlabel('NAME_CONTRACT_TYPEx')

ax1.set_ylabel("Number of Loans")

# ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)



ax2 = sns.countplot(ax=ax2,data=refused_df,x='NAME_CONTRACT_TYPEx')

ax2.set_title("Approved", fontsize=10)

ax2.set_xlabel('NAME_CONTRACT_TYPEx')

ax2.set_ylabel("Number of Loans")

# ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)



ax3 = sns.countplot(ax=ax3,data=canceled_df,x='NAME_CONTRACT_TYPEx')

ax3.set_title("Canceled", fontsize=10)

ax3.set_xlabel('NAME_CONTRACT_TYPEx')

ax3.set_ylabel("Number of Loans")

# ax3.set_xticklabels(ax3.get_xticklabels(),rotation=90)



ax4 = sns.countplot(ax=ax4,data=unused_df,x='NAME_CONTRACT_TYPEx')

ax4.set_title("Unused", fontsize=10)

ax4.set_xlabel('NAME_CONTRACT_TYPEx')

ax4.set_ylabel("Number of Loans")

# ax4.set_xticklabels(ax4.get_xticklabels(),rotation=90)

plt.show()
def multi_plot(variable_name):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2,ncols=2, figsize=(15,12), sharey='all')

    fig.tight_layout(pad=10.0)



    ax1 = sns.countplot(ax=ax1,data=approved_df,x=variable_name)

    ax1.set_title("Refused", fontsize=10)

    ax1.set_ylabel("Number of Loans")

    ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)



    ax2 = sns.countplot(ax=ax2,data=refused_df,x=variable_name)

    ax2.set_title("Approved", fontsize=10)

    ax2.set_ylabel("Number of Loans")

    ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)



    ax3 = sns.countplot(ax=ax3,data=canceled_df,x=variable_name)

    ax3.set_title("Canceled", fontsize=10)

    ax3.set_xlabel(variable_name)

    ax3.set_ylabel("Number of Loans")

    ax3.set_xticklabels(ax3.get_xticklabels(),rotation=90)



    ax4 = sns.countplot(ax=ax4,data=unused_df,x=variable_name)

    ax4.set_title("Unused", fontsize=10)

    ax4.set_xlabel(variable_name)

    ax4.set_ylabel("Number of Loans")

    ax4.set_xticklabels(ax4.get_xticklabels(),rotation=90)

    

    plt.show()
multi_plot('NAME_CLIENT_TYPE')
multi_plot('CODE_GENDER')
multi_plot('NAME_EDUCATION_TYPE')
multi_plot('NAME_INCOME_TYPE')
multi_plot('NAME_FAMILY_STATUS')
multi_plot('NAME_PAYMENT_TYPE')
multi_plot('NAME_PORTFOLIO')
multi_plot('OCCUPATION_TYPE')
multi_plot('NAME_GOODS_CATEGORY')
multi_plot('PRODUCT_COMBINATION')