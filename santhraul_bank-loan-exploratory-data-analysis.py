# Import libraries

import numpy as np

print('numpy version\t:', np.__version__)

import pandas as pd

print('pandas version\t:', pd.__version__)

import matplotlib.pyplot as plt

import seaborn as sns

print('seaborn version\t:', sns.__version__)

from scipy import stats



import os



pd.set_option('display.max_columns', 200) # to display all the columns

pd.set_option('display.max_rows',150) # to display all rows of df series

pd.options.display.float_format = '{:.4f}'.format #set it to convert scientific noations such as 4.225108e+11 to 422510842796.00



import warnings

warnings.filterwarnings('ignore') # if there are any warning due to version mismatch, it will be ignored



import random
# # Sample data to overcome Memory Error

# # Less RAM: Reduce the data: It's completely fine to take a sample of the data to work on this case study

# # Random Sampling to get a random sample of data from the complete data

# filename = "application_data.csv"# This file is available is the same location as the jupyter notebook



# # Count the number of rows in my file

# num_lines = sum(1 for i in open(filename))

# # The number of rows that I wanted to load

# size = num_lines//2



# # Create a random indices between these two numbers



# random.seed(10)

# skip_id = random.sample(range(1, num_lines), num_lines-size)



# df_app = pd.read_csv(filename, skiprows = skip_id)
# read data

df_app = pd.read_csv('../input/credit-card/application_data.csv')
# get shape of data (rows, columns)

print(df_app.shape)
df_app.dtypes.value_counts()
# get some insights of data

df_app.head()
df_app.info()
# get the count, size and Unique value in each column of application data

df_app.agg(['count','size','nunique'])
# funcion to get null value

def column_wise_null_percentage(df):

    output = round(df.isnull().sum()/len(df.index)*100,2)

    return output
# get missign values of all columns

NA_col = column_wise_null_percentage(df_app)

NA_col
# identify columns only with null values

NA_col = NA_col[NA_col>0]

NA_col
# grafical representation of columns having % null values

plt.figure(figsize= (20,4),dpi=300)

NA_col.plot(kind = 'bar')

plt.title (' columns having null values')

plt.ylabel('% null values')

plt.show()

# plt.savefig('filename.png', dpi=300)
# Get the column with null values more than 50%

NA_col_50 = NA_col[NA_col>50]

print("Number of columns having null value more than 50% :", len(NA_col_50.index))

print(NA_col_50)
# removed 41 columns having null percentage more than 50%.

df_app = df_app.drop(NA_col_50.index, axis =1)

df_app.shape
# Get columns having <15% null values

NA_col_15 = NA_col[NA_col<15]

print("Number of columns having null value less than 15% :", len(NA_col_15.index))

print(NA_col_15)
NA_col_15.index
# understand the insight of missing columns having <15% null values

df_app[NA_col_15.index].describe()
# identify unique values in the colums having <15% null value 

df_app[NA_col_15.index].nunique().sort_values(ascending=False)
# Box plot for continuious variable

plt.figure(figsize=(12,4))

sns.boxplot(df_app['EXT_SOURCE_2'])

plt.show()
plt.figure(figsize=(12,4))

sns.boxplot(df_app['AMT_GOODS_PRICE'])

plt.show()
# identify maximum frequency values

print('Maximum Frequncy categorical values are,')

print('NAME_TYPE_SUITE: ',df_app['NAME_TYPE_SUITE'].mode()[0])

print('OBS_30_CNT_SOCIAL_CIRCLE:', df_app['OBS_30_CNT_SOCIAL_CIRCLE'].mode()[0])

print('DEF_30_CNT_SOCIAL_CIRCLE:', df_app['DEF_30_CNT_SOCIAL_CIRCLE'].mode()[0])

print('OBS_60_CNT_SOCIAL_CIRCLE:', df_app['OBS_60_CNT_SOCIAL_CIRCLE'].mode()[0])

print('DEF_60_CNT_SOCIAL_CIRCLE:', df_app['DEF_60_CNT_SOCIAL_CIRCLE'].mode()[0])

# Remove unwanted columns from application dataset for better analysis.



unwanted=['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE','FLAG_PHONE', 'FLAG_EMAIL',

          'REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','FLAG_EMAIL','CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',

          'REGION_RATING_CLIENT_W_CITY','FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4',

          'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10',

          'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',

          'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',

          'FLAG_DOCUMENT_21','EXT_SOURCE_2','EXT_SOURCE_3','YEARS_BEGINEXPLUATATION_AVG','FLOORSMAX_AVG','YEARS_BEGINEXPLUATATION_MODE',

          'FLOORSMAX_MODE','YEARS_BEGINEXPLUATATION_MEDI','FLOORSMAX_MEDI','TOTALAREA_MODE','EMERGENCYSTATE_MODE']



df_app.drop(labels=unwanted,axis=1,inplace=True)
df_app.shape
df_app.head()
# For Code Gender column



print('CODE_GENDER: ',df_app['CODE_GENDER'].unique())

print('No of values: ',df_app[df_app['CODE_GENDER']=='XNA'].shape[0])



XNA_count = df_app[df_app['CODE_GENDER']=='XNA'].shape[0]

per_XNA = round(XNA_count/len(df_app.index)*100,3)



print('% of XNA Values:',  per_XNA)



print('maximum frequency data :', df_app['CODE_GENDER'].describe().top)
# Dropping the XNA value in column 'CODE_GENDER' with "F" for the dataset



df_app = df_app.drop(df_app.loc[df_app['CODE_GENDER']=='XNA'].index)

df_app[df_app['CODE_GENDER']=='XNA'].shape
# For Organization column

print('No of XNA values: ', df_app[df_app['ORGANIZATION_TYPE']=='XNA'].shape[0])



XNA_count = df_app[df_app['ORGANIZATION_TYPE']=='XNA'].shape[0]

per_XNA = round(XNA_count/len(df_app.index)*100,3)



print('% of XNA Values:',  per_XNA)



df_app['ORGANIZATION_TYPE'].describe()

# # Dropping the rows have 'XNA' values in the organization type column



# df_app = df_app.drop(df_app.loc[df_app['ORGANIZATION_TYPE']=='XNA'].index)

# df_app[df_app['ORGANIZATION_TYPE']=='XNA'].shape
df_app.head()
# Casting variable into numeric in the dataset



numeric_columns=['TARGET','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','REGION_POPULATION_RELATIVE',

                 'DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','HOUR_APPR_PROCESS_START',

                 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',

                'DAYS_LAST_PHONE_CHANGE']



df_app[numeric_columns]=df_app[numeric_columns].apply(pd.to_numeric)

df_app.head(5)

# Converting '-ve' values into '+ve' Values

df_app['DAYS_BIRTH'] = df_app['DAYS_BIRTH'].abs()

df_app['DAYS_EMPLOYED'] = df_app['DAYS_EMPLOYED'].abs()

df_app['DAYS_REGISTRATION'] = df_app['DAYS_REGISTRATION'].abs()

df_app['DAYS_ID_PUBLISH'] = df_app['DAYS_ID_PUBLISH'].abs()

df_app['DAYS_LAST_PHONE_CHANGE'] = df_app['DAYS_LAST_PHONE_CHANGE'].abs()
# describe numeric columns

df_app[numeric_columns].describe()
# Box plot for selected columns

features = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','DAYS_EMPLOYED', 'DAYS_REGISTRATION']



plt.figure(figsize = (20, 15), dpi=300)

for i in enumerate(features):

    plt.subplot(3, 2, i[0]+1)

    sns.boxplot(x = i[1], data = df_app)

plt.show()
bins = [0,100000,200000,300000,400000,500000,10000000000]

slot = ['<100000', '100000-200000','200000-300000','300000-400000','400000-500000', '500000 and above']



df_app['AMT_INCOME_RANGE']=pd.cut(df_app['AMT_INCOME_TOTAL'],bins,labels=slot)
bins = [0,100000,200000,300000,400000,500000,600000,700000,800000,900000,10000000000]

slot = ['<100000', '100000-200000','200000-300000','300000-400000','400000-500000', '500000-600000',

        '600000-700000','700000-800000','850000-900000','900000 and above']



df_app['AMT_CREDIT_RANGE']=pd.cut(df_app['AMT_CREDIT'],bins,labels=slot)
# Dividing the dataset into two dataset of  target=1(client with payment difficulties) and target=0(all other)



target0_df=df_app.loc[df_app["TARGET"]==0]

target1_df=df_app.loc[df_app["TARGET"]==1]
# insights from number of target values



percentage_defaulters= round(100*len(target1_df)/(len(target0_df)+len(target1_df)),2)



percentage_nondefaulters=round(100*len(target0_df)/(len(target0_df)+len(target1_df)),2)



print('Count of target0_df:', len(target0_df))

print('Count of target1_df:', len(target1_df))





print('Percentage of people who paid their loan are: ', percentage_nondefaulters, '%' )

print('Percentage of people who did not paid their loan are: ', percentage_defaulters, '%' )
# Calculating Imbalance percentage

    

# Since the majority is target0 and minority is target1



imb_ratio = round(len(target0_df)/len(target1_df),2)



print('Imbalance Ratio:', imb_ratio)
# Count plotting in logarithmic scale



def uniplot(df,col,title,hue =None):

    

    sns.set_style('whitegrid')

    sns.set_context('talk')

    plt.rcParams["axes.labelsize"] = 14

    plt.rcParams['axes.titlesize'] = 16

    plt.rcParams['axes.titlepad'] = 14

    

    

    temp = pd.Series(data = hue)

    fig, ax = plt.subplots()

    width = len(df[col].unique()) + 7 + 4*len(temp.unique())

    fig.set_size_inches(width , 8)

    plt.xticks(rotation=45)

    plt.yscale('log')

    plt.title(title)

    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue) 

        

    plt.show()
# Categoroical Univariate Analysis in logarithmic scale



features = ['AMT_INCOME_RANGE', 'AMT_CREDIT_RANGE','NAME_INCOME_TYPE','NAME_CONTRACT_TYPE']

plt.figure(figsize = (20, 15))



for i in enumerate(features):

    plt.subplot(2, 2, i[0]+1)

    plt.subplots_adjust(hspace=0.5)

    sns.countplot(x = i[1], hue = 'TARGET', data = df_app)

    

    plt.rcParams['axes.titlesize'] = 16

    

    plt.xticks(rotation = 45)

    plt.yscale('log')

    
# Categoroical Univariate Analysis in Value scale



features = ['CODE_GENDER','FLAG_OWN_CAR']

plt.figure(figsize = (20, 10))



for i in enumerate(features):

    plt.subplot(2, 2, i[0]+1)

    plt.subplots_adjust(hspace=0.5)

    sns.countplot(x = i[1], hue = 'TARGET', data = df_app)

     

    plt.rcParams['axes.titlesize'] = 16

    plt.xticks(rotation = 45)

#     plt.yscale('log')
# Univariate Analysis for continous variable



features = ['AMT_ANNUITY','AMT_GOODS_PRICE','DAYS_BIRTH','DAYS_EMPLOYED','DAYS_LAST_PHONE_CHANGE','DAYS_ID_PUBLISH']

plt.figure(figsize = (15, 20))



for i in enumerate(features):

    plt.subplot(3, 2, i[0]+1)

    plt.subplots_adjust(hspace=0.5)

    sns.boxplot(x = 'TARGET', y = i[1], data = df_app)

    
# Box plotting for Credit amount



plt.figure(figsize=(16,12))

plt.xticks(rotation=45)

sns.boxplot(data =target0_df, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient='v')

plt.title('Credit amount vs Education Status')

plt.show()
# Box plotting for Income amount in logarithmic scale



plt.figure(figsize=(16,12))

plt.xticks(rotation=45)

plt.yscale('log')

sns.boxplot(data =target0_df, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v')

plt.title('Income amount vs Education Status')

plt.show()
# Box plotting for credit amount



plt.figure(figsize=(15,10))

plt.xticks(rotation=45)

sns.boxplot(data =target0_df, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient='v')

plt.title('Credit Amount vs Education Status')

plt.show()
# Box plotting for Income amount in logarithmic scale



plt.figure(figsize=(16,12))

plt.xticks(rotation=45)

plt.yscale('log')

sns.boxplot(data =target0_df, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v')

plt.title('Income amount vs Education Status')

plt.show()
# Top 10 correlated variables: target 0 dataaframe



corr = target0_df.corr()

corrdf = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

corrdf = corrdf.unstack().reset_index()

corrdf.columns = ['Var1', 'Var2', 'Correlation']

corrdf.dropna(subset = ['Correlation'], inplace = True)

corrdf['Correlation'] = round(corrdf['Correlation'], 2)

corrdf['Correlation'] = abs(corrdf['Correlation'])

corrdf.sort_values(by = 'Correlation', ascending = False).head(10)
# Top 10 correlated variables: target 1 dataaframe



corr = target1_df.corr()

corrdf = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

corrdf = corrdf.unstack().reset_index()

corrdf.columns = ['Var1', 'Var2', 'Correlation']

corrdf.dropna(subset = ['Correlation'], inplace = True)

corrdf['Correlation'] = round(corrdf['Correlation'], 2)

corrdf['Correlation'] = abs(corrdf['Correlation'])

corrdf.sort_values(by = 'Correlation', ascending = False).head(10)
# Reading the dataset of previous application



df_prev=pd.read_csv('../input/credit-card/previous_application.csv')
#explore the dataset

df_prev.columns
# get shape of data (rows, columns)

df_prev.shape
# get the type of dataset

df_prev.dtypes
# displaying the informtion of previous application dataset

df_prev.info()
# Describing the previous application dataset

df_prev.describe()
# Finding percentage of null values columns

NA_col_pre = column_wise_null_percentage(df_prev)
# identify columns only with null values

NA_col_pre = NA_col_pre[NA_col_pre>0]

NA_col_pre
# grafical representation of columns having % null values

plt.figure(figsize= (20,4),dpi=300)

NA_col_pre.plot(kind = 'bar')

plt.title (' columns having null values')

plt.ylabel('% null values')

plt.show()
# Get the column with null values more than 50%

NA_col_pre = NA_col_pre[NA_col_pre>50]

print("Number of columns having null value more than 50% :", len(NA_col_pre.index))

print(NA_col_pre)
# removed 4 columns having null percentage more than 50%.

df_prev = df_prev.drop(NA_col_pre.index, axis =1)

df_prev.shape
# Merging the Application dataset with previous appliaction dataset



df_comb = pd.merge(left=df_app,right=df_prev,how='inner',on='SK_ID_CURR',suffixes='_x')

df_comb.shape
df_comb.head()
# Renaming the column names after merging from combined df



df_comb = df_comb.rename({'NAME_CONTRACT_TYPE_' : 'NAME_CONTRACT_TYPE','AMT_CREDIT_':'AMT_CREDIT','AMT_ANNUITY_':'AMT_ANNUITY',

                         'WEEKDAY_APPR_PROCESS_START_' : 'WEEKDAY_APPR_PROCESS_START',

                         'HOUR_APPR_PROCESS_START_':'HOUR_APPR_PROCESS_START','NAME_CONTRACT_TYPEx':'NAME_CONTRACT_TYPE_PREV',

                         'AMT_CREDITx':'AMT_CREDIT_PREV','AMT_ANNUITYx':'AMT_ANNUITY_PREV',

                         'WEEKDAY_APPR_PROCESS_STARTx':'WEEKDAY_APPR_PROCESS_START_PREV',

                         'HOUR_APPR_PROCESS_STARTx':'HOUR_APPR_PROCESS_START_PREV'}, axis=1)

# Removing unwanted columns from cmbined df for analysis



df_comb.drop(['SK_ID_CURR','WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION', 

              'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',

              'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY','WEEKDAY_APPR_PROCESS_START_PREV',

              'HOUR_APPR_PROCESS_START_PREV', 'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY'],axis=1,inplace=True)
# Distribution of contract status in logarithmic scale

# Distribution of contract status in logarithmic scale



sns.set_style('whitegrid')

sns.set_context('talk')



plt.figure(figsize=(10,25),dpi = 300)

plt.rcParams["axes.labelsize"] = 20

plt.rcParams['axes.titlesize'] = 22

plt.rcParams['axes.titlepad'] = 30

plt.xticks(rotation=90)

plt.xscale('log')

plt.title('Distribution of contract status with purposes')

ax = sns.countplot(data = df_comb, y= 'NAME_CASH_LOAN_PURPOSE', 

                   order=df_comb['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'NAME_CONTRACT_STATUS') 
# Distribution of contract status



sns.set_style('whitegrid')

sns.set_context('talk')



plt.figure(figsize=(10,30),dpi = 300)

plt.rcParams["axes.labelsize"] = 20

plt.rcParams['axes.titlesize'] = 22

plt.rcParams['axes.titlepad'] = 30

plt.xticks(rotation=90)

plt.xscale('log')

plt.title('Distribution of purposes with target ')

ax = sns.countplot(data = df_comb, y= 'NAME_CASH_LOAN_PURPOSE', 

                   order=df_comb['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'TARGET') 
# Box plotting for Credit amount in logarithmic scale



plt.figure(figsize=(20,15),dpi = 300)

plt.xticks(rotation=90)

plt.yscale('log')

sns.boxplot(data =df_comb, x='NAME_CASH_LOAN_PURPOSE',hue='NAME_INCOME_TYPE',y='AMT_CREDIT_PREV',orient='v')

plt.title('Prev Credit amount vs Loan Purpose')

plt.show()
# Box plotting for Credit amount prev vs Housing type in logarithmic scale



plt.figure(figsize=(15,15),dpi = 150)

plt.xticks(rotation=90)

sns.barplot(data =df_comb, y='AMT_CREDIT_PREV',hue='TARGET',x='NAME_HOUSING_TYPE')

plt.title('Prev Credit amount vs Housing type')

plt.show()