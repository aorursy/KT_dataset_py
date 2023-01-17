import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500) 
pd.set_option('display.max_rows', 500) 

# Supress Warnings

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns 
df_application_backup = pd.read_csv("application_data.csv")
df_Previous_backup = pd.read_csv("previous_application.csv")
df_application = pd.read_csv("application_data.csv")
df_application.head()
df_previous = pd.read_csv("previous_application.csv")
df_previous.head()
print('Size of application_data', df_application.shape)
df_application.columns.values
df_application.info(verbose= True)
df_application.describe()
100*df_application.isnull().sum()/len(df_application)

df_application.drop(df_application.columns[(100*df_application.isnull().sum()/len(df_application))>=50], axis=1, inplace= True)
df_application = df_application.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21','EXT_SOURCE_3',
       'YEARS_BEGINEXPLUATATION_AVG', 'FLOORSMAX_AVG','YEARS_BEGINEXPLUATATION_MODE', 'FLOORSMAX_MODE','YEARS_BEGINEXPLUATATION_MEDI', 'FLOORSMAX_MEDI', 'TOTALAREA_MODE',
       'EMERGENCYSTATE_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE',
       'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',
       'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY', 
       'LIVE_CITY_NOT_WORK_CITY'],axis=1)
df_application.head()
# Checking the column name and shape after dropping columns
print(df_application.columns)
print('Size of application_data', df_application.shape)
print(df_application.columns[100*df_application.isnull().sum()/len(df_application) > 0].tolist())
# Columns having <=8 and OCCUPATION_TYPE,ORGANIZATION_TYPE are categorical columns others are continuous columns
df_application.nunique().sort_values()
#plotting Box plot to get outliers in order to get the values for outliers in order get imputing values for variables.

plt.figure(1,figsize=(15,8)) 

# create 1st subplot:
plt.subplot(2,2,1) 
plt.title('Annuity Amount')
sns.boxplot(y=df_application["AMT_ANNUITY"])

# cretae 2nd subplot:
plt.subplot(2,2,2) 
plt.title('Loan Amount')
sns.boxplot(y=df_application["AMT_GOODS_PRICE"])

# cretae 3rd subplot:
plt.subplot(2,2,3)
plt.title('Family Members')
sns.boxplot(y=df_application["CNT_FAM_MEMBERS"])


# Getting mean and medial for all 4 variables 
print (df_application["AMT_ANNUITY"].aggregate(['mean', 'median']))
print(df_application["AMT_GOODS_PRICE"].aggregate(['mean', 'median']))
print(df_application["CNT_FAM_MEMBERS"].aggregate(['mean', 'median']))

# NAME_TYPE_SUITE column imputation
plt.figure(figsize=(10,5))
sns.countplot(x='NAME_TYPE_SUITE',data=df_application)

print(df_application.NAME_TYPE_SUITE.mode())
# OCCUPATION_TYPE column imputation
plt.figure(figsize=(10,5))
plt.xticks(rotation=90)
sns.countplot(x='OCCUPATION_TYPE',data=df_application)

print(df_application.OCCUPATION_TYPE.mode())
df_application.dtypes
# converting DAYS_REGISTRATION and CNT_FAM_MEMBERS column datatype from Float64 to int64 as these cannot be float
df_application['DAYS_REGISTRATION']= df_application['DAYS_REGISTRATION'].astype('int64')

# Removing rows for which CNT_FAM_MEMBERS values are missing, as these are very less(.000650%),beacuse for missing values while converting datatype it is throwing error.
df_application.dropna(subset=['CNT_FAM_MEMBERS'], inplace= True)
df_application['CNT_FAM_MEMBERS']= df_application['CNT_FAM_MEMBERS'].astype('int64')

# Removing rows for which DAYS_LAST_PHONE_CHANGE values are missing, as these are very less(0.000325%),beacuse for missing values while converting datatype it is throwing error.
df_application.dropna(subset=['DAYS_LAST_PHONE_CHANGE'], inplace= True)
df_application['DAYS_LAST_PHONE_CHANGE']= df_application['DAYS_LAST_PHONE_CHANGE'].astype('int64')
df_application.dtypes
# Converting negative values of DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH as it should be in positve
cols_negative = ['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']

for i in cols_negative:
    df_application[i] = df_application[i].apply(lambda x: round(abs(x)))
df_application.head()

plt.figure(figsize=(15,8))
plt.subplot(2,2,1)
sns.boxplot(df_application["AMT_INCOME_TOTAL"])


plt.subplot(2,2,2) 
sns.distplot(df_application["AMT_INCOME_TOTAL"])


# After removing outliers
plt.subplot(2,2,3)
sns.boxplot(df_application[df_application["AMT_INCOME_TOTAL"]<800000]["AMT_INCOME_TOTAL"])
plt.show()

plt.figure(figsize=(15,8))
plt.subplot(2,2,1)
sns.boxplot(df_application["AMT_CREDIT"])
plt.title('Credit Amount')


plt.subplot(2,2,2) 
sns.distplot(df_application["AMT_CREDIT"])


# After removing outliers
plt.subplot(2,2,3)
sns.boxplot(df_application[df_application["AMT_CREDIT"]<2300000]["AMT_CREDIT"])
plt.show()
plt.figure(figsize=(15,8))
plt.subplot(2,2,1)
sns.boxplot(y=df_application["AMT_ANNUITY"])
plt.title('Annuity Amount')

# Imputimg Null values with median to draw distplot
plt.subplot(2,2,2)
df_application['AMT_ANNUITY'].fillna((df_application['AMT_ANNUITY'].median()), inplace=True) 
sns.distplot(df_application["AMT_ANNUITY"])

# After removing outliers
plt.subplot(2,2,3)
sns.boxplot(df_application[df_application["AMT_ANNUITY"]<80000]["AMT_ANNUITY"])
plt.title('Annuity Amount')
plt.show()
plt.figure(figsize=(15,8))
plt.subplot(2,2,1)
sns.boxplot(y=df_application["AMT_GOODS_PRICE"])
plt.title('Loan Amount')

# Imputimg Null values with median to draw distplot
plt.subplot(2,2,2)
df_application['AMT_GOODS_PRICE'].fillna((df_application['AMT_GOODS_PRICE'].median()), inplace=True) 
sns.distplot(df_application["AMT_GOODS_PRICE"])

# After removing outliers

plt.subplot(2,2,3)
sns.boxplot(df_application[df_application["AMT_GOODS_PRICE"]<1850000]["AMT_GOODS_PRICE"])
plt.title('Loan Amount')
plt.show()
sns.boxplot(x=df_application['DAYS_BIRTH'])
df_application['AMT_INCOME_TOTAL'].describe()
# Binning based quantiles
df_application['Income_lable']= pd.cut(df_application['AMT_INCOME_TOTAL'],[25649.999,112500.0,147150.0,202500.0,117000000.0],labels = ['Poor','Low', 'medium', 'High'])
df_application
# Visualization for Binning column
x,y = 'Income_lable', 'TARGET'

df1 = df_application.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1,aspect=15/8.27)
g.ax.set_ylim(0,100)

plt.xticks(rotation=90)
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    g.ax.text(txt_x,txt_y,txt)
df_application['DAYS_BIRTH']= df_application['DAYS_BIRTH']/356
df_application['DAYS_BIRTH']= df_application['DAYS_BIRTH'].astype('int64')
df_application['Age']= pd.cut(df_application['DAYS_BIRTH'],[0,30,50,70],labels = ['Young','Adult','Old'])
df_application.Age
x,y = 'Age', 'TARGET'

df1 = df_application.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1,aspect=15/8.27)
g.ax.set_ylim(0,100)

plt.xticks(rotation=90)
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    g.ax.text(txt_x,txt_y,txt)
df1
temp = df_application["TARGET"].value_counts()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
ax.pie(temp.values, labels= temp.index,autopct='%1.2f%%')
plt.title('Loan Repaid or not')
plt.show()
df_0= df_application[df_application['TARGET']==0]
df_1 = df_application[df_application['TARGET']==1]
df_0.head()
df_1.head()

x,y = 'NAME_INCOME_TYPE', 'TARGET'

df1 = df_application.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1,aspect=15/8.27)
g.ax.set_ylim(0,100)

plt.xticks(rotation=90)
plt.title('Income Type')
plt.show()

df1

x,y = 'NAME_FAMILY_STATUS', 'TARGET'

df1 = df_application.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1,aspect=15/8.27)
g.ax.set_ylim(0,100)

plt.xticks(rotation=90)
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    g.ax.text(txt_x,txt_y,txt)
x,y = 'OCCUPATION_TYPE', 'TARGET'

df1 = df_application.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1,aspect=15/8.27)
g.ax.set_ylim(0,100)

plt.xticks(rotation=90)
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    g.ax.text(txt_x,txt_y,txt)
x,y = 'CODE_GENDER', 'TARGET'

df1 = df_application.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1,aspect=15/8.27)
g.ax.set_ylim(0,100)

plt.xticks(rotation=90)
plt.show()
df1
x,y = 'NAME_CONTRACT_TYPE', 'TARGET'

df1 = df_application.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1,aspect=15/8.27)
g.ax.set_ylim(0,100)

plt.xticks(rotation=90)
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    g.ax.text(txt_x,txt_y,txt)
x,y = 'NAME_EDUCATION_TYPE', 'TARGET'

df1 = df_application.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1,aspect=15/8.27)
g.ax.set_ylim(0,100)

plt.xticks(rotation=90)
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    g.ax.text(txt_x,txt_y,txt)
# Owning Car
x,y = 'FLAG_OWN_CAR', 'TARGET'

df1 = df_application.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1,aspect=20/8.27)
g.ax.set_ylim(0,100)

plt.xticks(rotation=90)
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    g.ax.text(txt_x,txt_y,txt)

# Owning Realty

x,y = 'FLAG_OWN_REALTY', 'TARGET'

df1 = df_application.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1,aspect=20/8.27)
g.ax.set_ylim(0,100)

plt.xticks(rotation=90)
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    g.ax.text(txt_x,txt_y,txt)
income_type=['Maternity leave','Unemployed','Working','Commercial associate']

for i in income_type:
    g = sns.catplot(x='NAME_FAMILY_STATUS', hue='TARGET', col= 'Age',col_wrap=3,kind='count',data=df_application[df_application['NAME_INCOME_TYPE']==i],aspect=1)
    plt.xticks(rotation=90)
    plt.show()
    
income_type=np.delete(df_application['OCCUPATION_TYPE'].unique(), 4)
genders=['M']
for i in income_type:
    for gender in genders:
        temp_dataframe = df_application[df_application['OCCUPATION_TYPE']==i][df_application['CODE_GENDER']==gender]
        if len(temp_dataframe) > 0:
            g = sns.catplot(x='NAME_FAMILY_STATUS',margin_titles=True , hue='TARGET', col= 'Age',col_wrap=3,orient='v' ,kind='count',data=temp_dataframe,aspect=1)
            plt.title(i + '_' + gender)
            plt.xticks(rotation= 90)
            plt.show()
    
income_type=np.delete(df_application['OCCUPATION_TYPE'].unique(), 4)
genders=['F']
for i in income_type:
    for gender in genders:
        temp_dataframe = df_application[df_application['OCCUPATION_TYPE']==i][df_application['CODE_GENDER']==gender]
        if len(temp_dataframe) > 0:
            g = sns.catplot(x='NAME_FAMILY_STATUS',margin_titles=True , hue='TARGET', col= 'Age',col_wrap=3,orient='v' ,kind='count',data=temp_dataframe,aspect=1)
            plt.title(i + '_' + gender)
            plt.xticks(rotation=90)
            plt.show()
    
income_type=np.delete(df_application['OCCUPATION_TYPE'].unique(), 4)
genders=['XNA']
for i in income_type:
    for gender in genders:
        temp_dataframe = df_application[df_application['OCCUPATION_TYPE']==i][df_application['CODE_GENDER']==gender]
        if len(temp_dataframe) > 0:
            g = sns.catplot(x='NAME_FAMILY_STATUS',margin_titles=True , hue='TARGET', col= 'Age',col_wrap=3,orient='v' ,kind='count',data=temp_dataframe,aspect=1)
            plt.title(i + '_' + gender)
            plt.xticks(rotation=90)
            plt.show()
income_label=df_application.Income_lable.astype('str').unique()
for i in income_label:
    g = sns.catplot(x='NAME_FAMILY_STATUS', hue='TARGET', col= 'Age',col_wrap=3,orient='v' ,kind='count',data=df_application[df_application.Income_lable==i],aspect=1)
    plt.title(i )
    plt.xticks(rotation=90)
    plt.show()
# Analysis for Male customers
REGION_RATING_CLIENT=df_application['REGION_RATING_CLIENT'].unique()
REGION_RATING_CLIENT
genders=['M']
for i in REGION_RATING_CLIENT:
    for gender in genders:
        temp_dataframe = df_application[df_application['REGION_RATING_CLIENT']==i][df_application['CODE_GENDER']==gender]
        if len(temp_dataframe) > 0:
            g = sns.catplot(x='NAME_FAMILY_STATUS',margin_titles=True , hue='TARGET', col= 'Age',col_wrap=3,kind='count',data=temp_dataframe,aspect=1)
            plt.title(str(i) + '_' + gender)
            plt.xticks(rotation=90)
            plt.show()
# Analysis for Female customers

REGION_RATING_CLIENT=df_application['REGION_RATING_CLIENT'].unique()
REGION_RATING_CLIENT
# df_application[df_application['REGION_RATING_CLIENT']==3] REGION_RATING_CLIENT_W_CITY
genders=['F']
for i in REGION_RATING_CLIENT:
    for gender in genders:
        temp_dataframe = df_application[df_application['REGION_RATING_CLIENT']==i][df_application['CODE_GENDER']==gender]
        if len(temp_dataframe) > 0:
            g = sns.catplot(x='NAME_FAMILY_STATUS',margin_titles=True , hue='TARGET', col= 'Age',col_wrap=3,kind='count',data=temp_dataframe,aspect=1)
            plt.title(str(i) + '_' + gender)
            plt.xticks(rotation=90)
            plt.show()
# Analysis for XNA customers

REGION_RATING_CLIENT=df_application['REGION_RATING_CLIENT'].unique()
REGION_RATING_CLIENT
genders=['XNA']
for i in REGION_RATING_CLIENT:
    for gender in genders:
        temp_dataframe = df_application[df_application['REGION_RATING_CLIENT']==i][df_application['CODE_GENDER']==gender]
        if len(temp_dataframe) > 0:
            g = sns.catplot(x='NAME_FAMILY_STATUS',margin_titles=True , hue='TARGET', col= 'Age',col_wrap=3,kind='count',data=temp_dataframe,aspect=1)
            plt.title(str(i) + '_' + gender)
            plt.xticks(rotation=90)
            plt.show()
# df_application

REGION_RATING_CLIENT_W_CITY=df_application['REGION_RATING_CLIENT_W_CITY'].unique()
REGION_RATING_CLIENT_W_CITY

for i in REGION_RATING_CLIENT_W_CITY:
    temp_dataframe = df_application[df_application['REGION_RATING_CLIENT']==i]
    if len(temp_dataframe) > 0:
        g = sns.catplot(x='NAME_FAMILY_STATUS',margin_titles=True , hue='TARGET', col= 'Age',col_wrap=3,kind='count',data=temp_dataframe,aspect=1)
        plt.title(i)
        plt.xticks(rotation=90)
        plt.show()
var_numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
corr_1 = (df_1.select_dtypes(include=var_numerics)).corr().abs()
corr_1 = pd.DataFrame(corr_1.unstack()).reset_index()
corr_1.columns = ['FEATURE_1', 'FEATURE_2', 'CORRELATION']
dup = (corr_1[['FEATURE_1', 'FEATURE_2']].apply(frozenset, axis=1).duplicated()) | (corr_1['FEATURE_1']==corr_1['FEATURE_2']) 
corr_1 = corr_1[~dup]
print(corr_1.nlargest(10,['CORRELATION']))
var_numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
corr_1 = (df_0.select_dtypes(include=var_numerics)).corr().abs()
corr_1 = pd.DataFrame(corr_1.unstack()).reset_index()
corr_1.columns = ['FEATURE_1', 'FEATURE_2', 'CORRELATION']
dup = (corr_1[['FEATURE_1', 'FEATURE_2']].apply(frozenset, axis=1).duplicated()) | (corr_1['FEATURE_1']==corr_1['FEATURE_2']) 
corr_1 = corr_1[~dup]
print(corr_1.nlargest(10,['CORRELATION']))
plt.figure(figsize = (15, 30))
plt.subplot(5,2,1)
sns.scatterplot(x='DAYS_EMPLOYED', y='FLAG_EMP_PHONE', data=df_0)

plt.subplot(5,2,2)
sns.scatterplot(x='AMT_CREDIT', y='AMT_GOODS_PRICE', data=df_0)
plt.xticks(rotation=90)

plt.subplot(5,2,3)
sns.scatterplot(x='REGION_RATING_CLIENT', y='REGION_RATING_CLIENT_W_CITY', data=df_0)
plt.xticks(rotation=90)

plt.subplot(5,2,4)
sns.scatterplot(x='CNT_CHILDREN', y='CNT_FAM_MEMBERS', data=df_0)
plt.xticks(rotation=90)

plt.subplot(5,2,5)
sns.scatterplot(x='AMT_ANNUITY', y='AMT_GOODS_PRICE', data=df_0)
plt.xticks(rotation=90)

plt.subplot(5,2,6)
sns.scatterplot(x='AMT_CREDIT', y='AMT_ANNUITY', data=df_0)
plt.xticks(rotation=90)

plt.subplot(5,2,7)
sns.scatterplot(x='DAYS_BIRTH', y='DAYS_EMPLOYED', data=df_0)
plt.xticks(rotation=90)

plt.subplot(5,2,8)
sns.scatterplot(x='DAYS_BIRTH', y='FLAG_EMP_PHONE', data=df_0)
plt.xticks(rotation=90)

plt.subplot(5,2,9)
sns.scatterplot(x='REGION_POPULATION_RELATIVE', y='REGION_RATING_CLIENT', data=df_0)
plt.xticks(rotation=90)

plt.subplot(5,2,10)
sns.scatterplot(x='REGION_POPULATION_RELATIVE', y='REGION_RATING_CLIENT_W_CITY', data=df_0)
plt.xticks(rotation=90)

plt.figure(figsize = (15, 30))
plt.subplot(5,2,1)
sns.scatterplot(x='DAYS_EMPLOYED', y='FLAG_EMP_PHONE', data=df_1)

plt.subplot(5,2,2)
sns.scatterplot(x='AMT_CREDIT', y='AMT_GOODS_PRICE', data=df_1)
plt.xticks(rotation=90)

plt.subplot(5,2,3)
sns.scatterplot(x='REGION_RATING_CLIENT', y='REGION_RATING_CLIENT_W_CITY', data=df_1)
plt.xticks(rotation=90)

plt.subplot(5,2,4)
sns.scatterplot(x='CNT_CHILDREN', y='CNT_FAM_MEMBERS', data=df_1)
plt.xticks(rotation=90)

plt.subplot(5,2,5)
sns.scatterplot(x='AMT_ANNUITY', y='AMT_GOODS_PRICE', data=df_1)
plt.xticks(rotation=90)

plt.subplot(5,2,6)
sns.scatterplot(x='AMT_CREDIT', y='AMT_ANNUITY', data=df_1)
plt.xticks(rotation=90)

plt.subplot(5,2,7)
sns.scatterplot(x='DAYS_BIRTH', y='DAYS_EMPLOYED', data=df_1)
plt.xticks(rotation=90)

plt.subplot(5,2,8)
sns.scatterplot(x='DAYS_BIRTH', y='FLAG_EMP_PHONE', data=df_1)
plt.xticks(rotation=90)

plt.subplot(5,2,9)
sns.scatterplot(x='REGION_POPULATION_RELATIVE', y='REGION_RATING_CLIENT', data=df_1)
plt.xticks(rotation=90)

plt.subplot(5,2,10)
sns.scatterplot(x='REGION_POPULATION_RELATIVE', y='REGION_RATING_CLIENT_W_CITY', data=df_1)
plt.xticks(rotation=90)

f, axs = plt.subplots(1,2,figsize=(15,8))
plt.subplot(2,1,1)
sns.distplot(df_1['AMT_CREDIT'].dropna(), hist = False, label = "Defaulter", color = 'red')
plt.subplot(2,1,1)
sns.distplot(df_0['AMT_CREDIT'].dropna(), hist = False, label = "Non Defaulter", color = 'green')
plt.show()

f, axs = plt.subplots(1,2,figsize=(15,8))
plt.subplot(2,1,1)
sns.distplot(df_1['AMT_ANNUITY'].dropna(), hist = False, label = "Defaulter", color = 'red')
plt.subplot(2,1,1)
sns.distplot(df_0['AMT_ANNUITY'].dropna(), hist = False, label = "Non Defaulter", color = 'green')
plt.show()

f, axs = plt.subplots(1,2,figsize=(15,8))
plt.subplot(2,1,1)
sns.distplot(df_1['AMT_GOODS_PRICE'].dropna(), hist = False, label = "Defaulter", color = 'red')
plt.subplot(2,1,1)
sns.distplot(df_0['AMT_GOODS_PRICE'].dropna(), hist = False, label = "Non Defaulter", color = 'green')
plt.show()
f, axs = plt.subplots(1,2,figsize=(15,8))
plt.subplot(2,1,1)
sns.distplot(df_1['DAYS_BIRTH'].dropna(), hist = False, label = "Defaulter", color = 'red')
plt.subplot(2,1,1)
sns.distplot(df_0['DAYS_BIRTH'].dropna(), hist = False, label = "Non Defaulter", color = 'green')
plt.show()

sns.boxplot(x='TARGET', y='DAYS_BIRTH', data=df_application)
plt.show()
sns.boxplot(x='TARGET', y = 'EXT_SOURCE_2', data=df_application)
plt.show()
sns.boxplot(x='TARGET', y='REGION_POPULATION_RELATIVE', data=df_application)
plt.show()
# Getting size of previous application
print('Size of application_data', df_previous.shape)
# Getting column info
df_previous.info()
df_previous.describe()
# Getting missing values
100*df_previous.isnull().sum()/len(df_previous)
# Dropping column having >= 20 % missing values
df_previous.drop(df_previous.columns[(100*df_previous.isnull().sum()/len(df_previous))>=20], axis=1, inplace= True)
df_previous.columns
df_previous.shape
100*df_previous.isnull().sum()/len(df_previous)
df_previous.info()
# Impute method for PRODUCT_COMBINATION
print(df_previous['PRODUCT_COMBINATION'].mode())

plt.figure(figsize=(15,8))
sns.countplot(x='PRODUCT_COMBINATION',data=df_previous)
plt.xticks(rotation=90)
plt.show()
# Impute method for AMT_CREDIT
plt.figure(figsize=(15,8))
sns.boxplot(y=df_previous["AMT_CREDIT"])
plt.title('Approved Loan Amount')
plt.show()

df_previous['AMT_CREDIT'].aggregate(['mean', 'median'])
plt.figure(figsize=(15,8))
plt.subplot(2,2,1)
sns.boxplot(y=df_previous["AMT_APPLICATION"])
plt.title('Applied Loan Amount')

plt.subplot(2,2,2)
sns.distplot(df_previous["AMT_APPLICATION"])


# After removing outliers

plt.subplot(2,2,3)
sns.boxplot(df_previous[df_previous["AMT_APPLICATION"]<2200000]["AMT_APPLICATION"])
plt.title('Net Price')
plt.show()
plt.figure(figsize=(15,8))
plt.subplot(2,2,1)
#plt.figure(figsize=(15,8))
sns.boxplot(y=df_previous["AMT_CREDIT"])
plt.title('Approved Loan Amount')

# Imputimg Null values with median to draw distplot
plt.subplot(2,2,2)
df_previous['AMT_CREDIT'].fillna((df_application['AMT_CREDIT'].median()), inplace=True) 
sns.distplot(df_previous["AMT_CREDIT"])

# After removing outliers
plt.subplot(2,2,3)
sns.boxplot(df_previous[df_previous["AMT_CREDIT"]<2000000]["AMT_CREDIT"])
plt.title('Approved Price')
plt.show()
# for Previous application having only required Columns
df_previous = df_previous[['SELLERPLACE_AREA','NFLAG_LAST_APPL_IN_DAY','SK_ID_CURR','NAME_CONTRACT_STATUS','CODE_REJECT_REASON','NAME_CLIENT_TYPE','CHANNEL_TYPE','NAME_YIELD_GROUP','PRODUCT_COMBINATION']]
df_final = pd.merge(df_application,df_previous,how='inner',on='SK_ID_CURR')
df_final.head()
print(df_final.shape)
x,y =  'TARGET', 'NAME_CONTRACT_STATUS'

df1 = df_final.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1,aspect=20/10)
g.ax.set_ylim(0,100)

plt.xticks(rotation=90)
plt.title('Application status in terms of Loan paid or not')
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    g.ax.text(txt_x,txt_y,txt)
    
x,y =  'NAME_CLIENT_TYPE','NAME_CONTRACT_STATUS'

df1 = df_final.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1,aspect=20/10)
g.ax.set_ylim(0,100)

plt.xticks(rotation=90)
plt.title('Application status in terms of Loan paid or not')
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    g.ax.text(txt_x,txt_y,txt)
    

x,y = 'NAME_INCOME_TYPE', 'NAME_CONTRACT_STATUS'

df1 = df_final.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1,aspect=15/8.27)
g.ax.set_ylim(0,100)

plt.xticks(rotation=90)

plt.show()
df1
x,y = 'CODE_GENDER', 'NAME_CONTRACT_STATUS'

df1 = df_final.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1,aspect=15/8.27)
g.ax.set_ylim(0,100)

plt.xticks(rotation=90)

plt.show()
df1