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
#importing data from CSV file into pandas dataframe



import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

 

application_data = pd.read_csv('../input/credit-card/application_data.csv')

application_data.head()
application_data.shape
application_data.dtypes.value_counts()
# Get the count,size and unique values in each column of application data

application_data.agg(['count','size','nunique'])
defaulters=application_data[application_data.TARGET==1]

nondefaulters=application_data[application_data.TARGET==0]
sns.countplot(application_data.TARGET)

plt.xlabel("TARGET Value")

plt.ylabel("Count of TARGET")

plt.title("Distribution of TARGET Variable")

plt.show()

percentage_defaulters=(len(defaulters)*100)/len(application_data)

percentage_nondefaulters=(len(nondefaulters)*100)/len(application_data)



print("The Percentage of people who have paid their loan is:",round(percentage_nondefaulters,2),"%")

print("The Percentage of people who have NOT paid their loan is:",round(percentage_defaulters,2),"%")

print("The Ratio of Data Imbalance is:",round(len(nondefaulters)/len(defaulters),2))
#Function to calculate meta-data to identify % of data is missing in each column

def meta_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    unique = data.nunique()

    datatypes = data.dtypes

    return pd.concat([total, percent, unique, datatypes], axis=1, keys=['Total', 'Percent', 'Unique', 'Data_Type']).sort_values(by="Percent", ascending=False)
#calculating meta-data for application_data

app_meta_data=meta_data(application_data)

app_meta_data.head(20)
#dropping columns with more than 57% missing values 

#Selected 57% because we don't want to drop EXT_SOURCE_1 which is an important variable

cols_to_keep=list(app_meta_data[(app_meta_data.Percent<57)].index)

application_data=application_data[cols_to_keep]

application_data.describe()
#Checking columns with very less missing values

low_missing=pd.DataFrame(app_meta_data[(app_meta_data.Percent>0)&(app_meta_data.Percent<15)])

low_missing
application_data.select_dtypes('object').columns
application_data.select_dtypes('float64').columns
application_data.select_dtypes('int64').columns
application_data.select_dtypes('int64').apply(pd.Series.nunique, axis = 0)
#columns to convert

cols_to_convert=list(app_meta_data[(app_meta_data.Unique==2)&(app_meta_data.Data_Type=="int64")].index)



#function to conver columns

def convert_data(application_data, cols_to_convert):

    for y in cols_to_convert:

        application_data.loc[:,y].replace((0, 1), ('N', 'Y'), inplace=True)

    return application_data



#calling the function for application_data

convert_data(application_data, cols_to_convert)

application_data.TARGET.replace(('N', 'Y'), (0, 1), inplace=True)

application_data.dtypes.value_counts()
defaulters=application_data[application_data.TARGET==1]



nondefaulters=application_data[application_data.TARGET==0]
application_data.select_dtypes('object').columns
## FUNCTION TO PLOT CHARTS



def plot_charts(var, label_rotation,horizontal_layout):

    if(horizontal_layout):

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5))

    else:

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15,20))

    

    s1=sns.countplot(ax=ax1,x=defaulters[var], data=defaulters, order= defaulters[var].value_counts().index,)

    ax1.set_title('Distribution of '+ '%s' %var +' for Defaulters', fontsize=10)

    ax1.set_xlabel('%s' %var)

    ax1.set_ylabel("Count of Loans")

    if(label_rotation):

        s1.set_xticklabels(s1.get_xticklabels(),rotation=90)

    s2=sns.countplot(ax=ax2,x=nondefaulters[var], data=nondefaulters, order= nondefaulters[var].value_counts().index,)

    if(label_rotation):

        s2.set_xticklabels(s2.get_xticklabels(),rotation=90)

    ax2.set_xlabel('%s' %var)

    ax2.set_ylabel("Count of Loans")

    ax2.set_title('Distribution of '+ '%s' %var +' for Non-Defaulters', fontsize=10)

    plt.show()
plot_charts('NAME_CONTRACT_TYPE', label_rotation=False,horizontal_layout=True)
plot_charts('CODE_GENDER', label_rotation=False,horizontal_layout=True)
plot_charts('FLAG_OWN_REALTY', label_rotation=False,horizontal_layout=True)

plot_charts('FLAG_OWN_CAR', label_rotation=False,horizontal_layout=True)
plot_charts('REG_CITY_NOT_LIVE_CITY', label_rotation=False,horizontal_layout=True)

plot_charts('REG_CITY_NOT_WORK_CITY', label_rotation=True,horizontal_layout=True)
plot_charts('NAME_HOUSING_TYPE', label_rotation=True,horizontal_layout=True)
plot_charts('NAME_FAMILY_STATUS', label_rotation=True,horizontal_layout=True)
plot_charts('NAME_EDUCATION_TYPE', label_rotation=True,horizontal_layout=True)
plot_charts('NAME_INCOME_TYPE', label_rotation=True,horizontal_layout=True)
plot_charts('WALLSMATERIAL_MODE', label_rotation=True,horizontal_layout=True)
plot_charts('ORGANIZATION_TYPE', label_rotation=True,horizontal_layout=False)
plot_charts('FLAG_WORK_PHONE', label_rotation=True,horizontal_layout=True)
plot_charts('NAME_INCOME_TYPE', label_rotation=True,horizontal_layout=True)
plot_charts('OCCUPATION_TYPE', label_rotation=True,horizontal_layout=True)
application_data.select_dtypes('float64').columns
application_data.select_dtypes('int64').columns
## FUNCTION FOR PLOTTING BOX PLOT AND HISTOGRAM



def plot_boxhist(var):

    plt.figure(figsize=(15,5))

    plt.subplot(1, 2, 1)

    s=sns.boxplot(y=defaulters[var]);

    plt.title('Box Plot of '+ '%s' %var +' for Defaulters', fontsize=10)

    plt.xlabel('%s' %var)

    plt.ylabel("Count of Loans")

    plt.subplot(1, 2, 2)

    s=plt.hist(x=defaulters[var]);

    plt.xlabel('%s' %var)

    plt.ylabel("Count of Loans")

    plt.title('Histogram of '+ '%s' %var +' for Defaulters', fontsize=10)

plt.show()
plot_boxhist('AMT_INCOME_TOTAL')
#Removing all entries above 99 percentile

application_data=application_data[application_data.AMT_INCOME_TOTAL<np.nanpercentile(application_data['AMT_INCOME_TOTAL'], 99)]



#update dataframes

defaulters=application_data[application_data.TARGET==1] 

nondefaulters=application_data[application_data.TARGET==0]



plot_boxhist('AMT_INCOME_TOTAL')
plot_boxhist('AMT_CREDIT')
#Removing all entries above 99 percentile

application_data=application_data[application_data.AMT_CREDIT<np.nanpercentile(application_data['AMT_CREDIT'], 99)]



#update dataframes

defaulters=application_data[application_data.TARGET==1] 

nondefaulters=application_data[application_data.TARGET==0]



plot_boxhist('AMT_CREDIT')
plot_boxhist('AMT_ANNUITY')
#Removing all entries above 99 percentile

application_data=application_data[application_data.AMT_ANNUITY<np.nanpercentile(application_data['AMT_ANNUITY'], 90)]



#update dataframes

defaulters=application_data[application_data.TARGET==1] 

nondefaulters=application_data[application_data.TARGET==0]



plot_boxhist('AMT_ANNUITY')
#Deriving new metric Age from Days Birth

application_data['AGE'] = application_data['DAYS_BIRTH'] / -365

plt.hist(application_data['AGE']);

plt.title('Histogram of age in years.');
sns.boxplot(y=application_data['DAYS_EMPLOYED']);

plt.title('Length of days employed before loan.');
application_data['DAYS_EMPLOYED'].describe()
application_data['DAYS_EMPLOYED']=application_data['DAYS_EMPLOYED'].replace(365243, np.nan)

application_data['DAYS_EMPLOYED'].describe()
#Deriving variable "Years Employed" from days employed

application_data['YEARS_EMPLOYED'] = (application_data['DAYS_EMPLOYED']/-365)



#update dataframes

defaulters=application_data[application_data.TARGET==1] 

nondefaulters=application_data[application_data.TARGET==0]
plot_boxhist('YEARS_EMPLOYED')
application_data.groupby(['NAME_INCOME_TYPE']).agg({'YEARS_EMPLOYED': ['mean', 'median', 'count', 'max'], 'AGE': ['median']})
application_data.groupby(['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE']).agg({'AMT_INCOME_TOTAL': ['mean', 'median', 'count', 'max']})
application_data['AMT_INCOME_TOTAL'].describe()
defaulters.loc[:,'INCOME_BRACKET']=pd.qcut(application_data.loc[:,'AMT_INCOME_TOTAL'],q=[0,0.10,0.35,0.50,0.90,1], labels=['Very_low','Low','Medium','High','Very_high'])

nondefaulters.loc[:,'INCOME_BRACKET']=pd.qcut(application_data.loc[:,'AMT_INCOME_TOTAL'],q=[0,0.10,0.35,0.50,0.90,1], labels=['Very_low','Low','Medium','High','Very_high'])
plot_charts('INCOME_BRACKET', label_rotation=True,horizontal_layout=True)
defaulters.loc[:,'Rating1']=pd.cut(application_data.loc[:,'EXT_SOURCE_1'],[0,0.10,0.35,0.50,0.80,1], labels=['Very_low','Low','Medium','High','Very_high'])

nondefaulters.loc[:,'Rating1']=pd.cut(application_data.loc[:,'EXT_SOURCE_1'],[0,0.10,0.35,0.50,0.80,1], labels=['Very_low','Low','Medium','High','Very_high'])
plot_charts('Rating1', label_rotation=True,horizontal_layout=True)
defaulters.loc[:,'Rating2']=pd.cut(application_data.loc[:,'EXT_SOURCE_2'],[0,0.10,0.35,0.50,0.80,1], labels=['Very_low','Low','Medium','High','Very_high'])

nondefaulters.loc[:,'Rating2']=pd.cut(application_data.loc[:,'EXT_SOURCE_2'],[0,0.10,0.35,0.50,0.80,1], labels=['Very_low','Low','Medium','High','Very_high'])
plot_charts('Rating2', label_rotation=True,horizontal_layout=True)
defaulters.loc[:,'Rating3']=pd.cut(application_data.loc[:,'EXT_SOURCE_3'], [0,0.10,0.35,0.50,0.80,1], labels=['Very_low','Low','Medium','High','Very_high'])

nondefaulters.loc[:,'Rating3']=pd.cut(application_data.loc[:,'EXT_SOURCE_3'], [0,0.10,0.35,0.50,0.80,1], labels=['Very_low','Low','Medium','High','Very_high'])
plot_charts('Rating3', label_rotation=True,horizontal_layout=True)
defaulters.loc[:,'AMT_ANNUITY_BINS']=pd.qcut(application_data.loc[:,'AMT_ANNUITY'], [0,0.30,0.50,0.85,0.1], labels=['Low','Medium','High','Very_High'])

nondefaulters.loc[:,'AMT_ANNUITY_BINS']=pd.qcut(application_data.loc[:,'AMT_ANNUITY'], [0,0.30,0.50,0.85,1], labels=['Low','Medium','High','Very_High'])
plot_charts('AMT_ANNUITY_BINS', label_rotation=False,horizontal_layout=True)
age_data = application_data.loc[:,['TARGET', 'DAYS_BIRTH']]

age_data.loc[:,'YEARS_BIRTH'] = application_data.loc[:,'DAYS_BIRTH']/ -365

# Bin the age data

age_data.loc[:,'YEARS_BINNED'] = pd.cut(age_data.loc[:,'YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))

age_data.head(10)
age_groups  = age_data.groupby('YEARS_BINNED').mean()

age_groups
plt.figure(figsize = (8, 8))



# Graph the age bins and the average of the target as a bar plot

plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])



# Plot labeling

plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')

plt.title('Failure to Repay by Age Group');
#selecting columns for correlation, removing cols for floor and house ec



cols=['EXT_SOURCE_1', 'EXT_SOURCE_3', 'EXT_SOURCE_2',

       'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'CNT_FAM_MEMBERS',

       'DAYS_LAST_PHONE_CHANGE', 'AMT_CREDIT', 'AMT_INCOME_TOTAL',

       'DAYS_REGISTRATION', 'REGION_POPULATION_RELATIVE','CNT_CHILDREN', 'HOUR_APPR_PROCESS_START',

       'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT',

       'DAYS_ID_PUBLISH', 'DAYS_EMPLOYED', 'DAYS_BIRTH']
defaulters_1=defaulters[cols]

defaulters_correlation = defaulters_1.corr()

round(defaulters_correlation, 3)
defaulters_correlation.head(10).index
c1=defaulters_correlation.unstack()

c1.sort_values(ascending=False).drop_duplicates().head(10)
c1.sort_values(ascending=False).drop_duplicates().tail(10)
# figure size

plt.figure(figsize=(30,20))



# heatmap

sns.heatmap(defaulters_correlation, cmap="YlGnBu", annot=True)

plt.show()
nondefaulters_1=nondefaulters[cols]

nondefaulters_correlation = nondefaulters_1.corr()

round(nondefaulters_correlation, 3)
nondefaulters_correlation.head(10).index
c2=nondefaulters_correlation.unstack()

c2.sort_values(ascending=False).drop_duplicates().head(10)
c2.sort_values(ascending=False).drop_duplicates().tail(10)
# figure size

plt.figure(figsize=(30,20))



# heatmap

sns.heatmap(nondefaulters_correlation, cmap="YlGnBu", annot=True)

plt.show()
#importing data from CSV file into pandas dataframe



import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt



 

previous_data = pd.read_csv('../input/credit-card/previous_application.csv')

previous_data.head()
previous_data.shape
application_data.dtypes.value_counts()
previous_data.columns
previous_data.info()
previous_data.NAME_CONTRACT_STATUS.unique()
import matplotlib

sns.countplot(previous_data.NAME_CONTRACT_STATUS)

plt.xlabel("Contract Status")

plt.ylabel("Count of Contract Status")

plt.title("Distribution of Contract Status")

plt.show()
prev_meta_data=meta_data(previous_data)

prev_meta_data.reset_index(drop=False).head(20)
#dropping columns with more than 55% missing values 

cols_to_keep=list(prev_meta_data[(prev_meta_data.Percent<55)].index)

previous_data=previous_data[cols_to_keep]

previous_data.describe()
#Checking columns with very less missing values

low_missing=pd.DataFrame(prev_meta_data[(prev_meta_data.Percent>0)&(prev_meta_data.Percent<15)])

low_missing
cols_to_convert=list(prev_meta_data[(prev_meta_data.Unique==2)&((prev_meta_data.Data_Type=="int64")|(prev_meta_data.Data_Type=="float64"))].index)

cols_to_convert
def convert_data(previous_data, cols_to_convert):

    for y in cols_to_convert:

        previous_data.loc[:,y].replace((0, 1), ('N', 'Y'), inplace=True)

    return previous_data

convert_data(previous_data, cols_to_convert)

previous_data.dtypes.value_counts()
approved=previous_data[previous_data.NAME_CONTRACT_STATUS=='Approved']

refused=previous_data[previous_data.NAME_CONTRACT_STATUS=='Refused']

canceled=previous_data[previous_data.NAME_CONTRACT_STATUS=='Canceled']

unused=previous_data[previous_data.NAME_CONTRACT_STATUS=='Unused Offer']
percentage_approved=(len(approved)*100)/len(previous_data)

percentage_refused=(len(refused)*100)/len(previous_data)

percentage_canceled=(len(canceled)*100)/len(previous_data)

percentage_unused=(len(unused)*100)/len(previous_data)



print("The Percentage of people whose loans have been Approved is:",round(percentage_approved,2),"%")

print("The Percentage of people whose loans have been Refused is:",round(percentage_refused,2),"%")

print("The Percentage of people whose loans have been Canceled is:",round(percentage_canceled,2),"%")

print("The Percentage of people whose loans have been Unused is:",round(percentage_unused,2),"%")
def plot_3charts(var, label_rotation,horizontal_layout):

    if(horizontal_layout):

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,5))

    else:

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(15,30))

    

    s1=sns.countplot(ax=ax1,x=refused[var], data=refused, order= refused[var].value_counts().index,)

    ax1.set_title("Refused", fontsize=10)

    ax1.set_xlabel('%s' %var)

    ax1.set_ylabel("Count of Loans")

    if(label_rotation):

        s1.set_xticklabels(s1.get_xticklabels(),rotation=90)

    

    s2=sns.countplot(ax=ax2,x=approved[var], data=approved, order= approved[var].value_counts().index,)

    if(label_rotation):

        s2.set_xticklabels(s2.get_xticklabels(),rotation=90)

    ax2.set_xlabel('%s' %var)

    ax2.set_ylabel("Count of Loans")

    ax2.set_title("Approved", fontsize=10)

    

    

    s3=sns.countplot(ax=ax3,x=canceled[var], data=canceled, order= canceled[var].value_counts().index,)

    ax3.set_title("Canceled", fontsize=10)

    ax3.set_xlabel('%s' %var)

    ax3.set_ylabel("Count of Loans")

    if(label_rotation):

        s3.set_xticklabels(s3.get_xticklabels(),rotation=90)

    plt.show()
previous_data.select_dtypes('object').columns
plot_3charts('PRODUCT_COMBINATION', label_rotation=True,horizontal_layout=True)
plot_3charts('NAME_YIELD_GROUP', label_rotation=True,horizontal_layout=True)
plot_3charts('NAME_PORTFOLIO', label_rotation=True,horizontal_layout=True)
plot_3charts('CHANNEL_TYPE', label_rotation=True,horizontal_layout=True)
plot_3charts('NAME_PRODUCT_TYPE', label_rotation=True,horizontal_layout=True)
plot_3charts('NAME_PAYMENT_TYPE', label_rotation=True,horizontal_layout=True)
plot_3charts('NAME_CONTRACT_TYPE', label_rotation=True,horizontal_layout=True)
plot_3charts('NAME_CLIENT_TYPE', label_rotation=True,horizontal_layout=True)
sns.countplot(x=approved['NAME_CLIENT_TYPE'], data=previous_data)
fig, ax = plt.subplots(figsize = (30, 8))

plt.subplot(1, 2, 1)

sns.boxplot(y=approved['AMT_ANNUITY']);

plt.subplot(1, 2, 2)

plt.hist(approved['AMT_ANNUITY'])

plt.title('AMT_ANNUITY')

plt.show()
approved=approved[approved.AMT_ANNUITY<np.nanpercentile(approved['AMT_ANNUITY'], 99)]

fig, ax = plt.subplots(figsize = (30, 8))

ax.set_title('AMT_ANNUITY boxplot on data within 99 percentile');

plt.subplot(1, 2, 1)

sns.boxplot(y=approved['AMT_ANNUITY']);

plt.subplot(1, 2, 2)

plt.hist(approved['AMT_ANNUITY'])

ax.set_title('AMT_ANNUITY')
fig, ax = plt.subplots(figsize = (30, 8))

plt.subplot(1, 2, 1)

sns.boxplot(y=approved['AMT_CREDIT']);

plt.subplot(1, 2, 2)

plt.hist(approved['AMT_CREDIT'])

plt.title('AMT_CREDIT')

plt.show()
approved=approved[approved.AMT_CREDIT<np.nanpercentile(approved['AMT_CREDIT'], 90)]

fig, ax = plt.subplots(figsize = (30, 8))

ax.set_title('AMT_CREDIT boxplot on data within 99 percentile');

plt.subplot(1, 2, 1)

sns.boxplot(y=approved['AMT_CREDIT']);

plt.subplot(1, 2, 2)

plt.hist(approved['AMT_CREDIT'])

ax.set_title('AMT_CREDIT')
fig, ax = plt.subplots(figsize = (30, 8))

plt.subplot(1, 2, 1)

sns.boxplot(y=approved['AMT_GOODS_PRICE']);

plt.subplot(1, 2, 2)

plt.hist(approved['AMT_GOODS_PRICE'])

plt.title('AMT_GOODS_PRICE')

plt.show()
approved=approved[approved.AMT_GOODS_PRICE<np.nanpercentile(approved['AMT_GOODS_PRICE'], 90)]

fig, ax = plt.subplots(figsize = (30, 8))

ax.set_title('AMT_GOODS_PRICE boxplot on data within 99 percentile');

plt.subplot(1, 2, 1)

sns.boxplot(y=approved['AMT_GOODS_PRICE']);

plt.subplot(1, 2, 2)

plt.hist(approved['AMT_GOODS_PRICE'])

ax.set_title('AMT_GOODS_PRICE')
cols_approved=['AMT_DOWN_PAYMENT', 'RATE_DOWN_PAYMENT', 'DAYS_TERMINATION', 'DAYS_LAST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_FIRST_DUE', 'DAYS_FIRST_DRAWING', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'CNT_PAYMENT', 'AMT_CREDIT', 'DAYS_DECISION', 'AMT_APPLICATION']

approved_num=approved[cols_approved]
cols_refused=['AMT_DOWN_PAYMENT', 'RATE_DOWN_PAYMENT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'CNT_PAYMENT', 'AMT_CREDIT', 'DAYS_DECISION', 'AMT_APPLICATION']

refused_num=refused[cols_refused]
#calculating correlation for approved

approved_correlation = approved_num.corr()

round(approved_correlation, 3)
c1=approved_correlation.unstack()

c1.sort_values(ascending=False).drop_duplicates().head(10)
c1.sort_values(ascending=False).drop_duplicates().tail(10)
# figure size

plt.figure(figsize=(30,20))



# heatmap

sns.heatmap(approved_correlation, cmap="YlGnBu", annot=True)

plt.show()
#calculating correlation for approved

refused_correlation = refused_num.corr()

round(refused_correlation, 3)
# figure size

plt.figure(figsize=(30,20))



# heatmap

sns.heatmap(refused_correlation, cmap="YlGnBu", annot=True)

plt.show()
c2=refused_correlation.unstack()

c2.sort_values(ascending=False).drop_duplicates().head(10)
c2.sort_values(ascending=False).drop_duplicates().tail(10)
def has_terminated(x):

    if x < 0:

        return 'Loan Terminated'

    else:

        return 'Loan Open'

    

approved['CURRENT_STATUS'] = approved['DAYS_TERMINATION'].apply(has_terminated)
plt.figure(figsize=(5,5))

sns.countplot(x=approved['CURRENT_STATUS'], data=approved)

plt.show()
new_df= previous_data.pivot_table(values = 'NAME_CONTRACT_STATUS', index = 'SK_ID_CURR', aggfunc = 'count')

new_df=new_df.reset_index(drop=False)

new_df.rename(columns = {'NAME_CONTRACT_STATUS':'Count of Refused Loans'}, inplace = True)
merged_df1=pd.merge(new_df, pd.DataFrame(application_data[['SK_ID_CURR','TARGET']]), how='inner', on='SK_ID_CURR')
merged_df1=merged_df1[merged_df1['Count of Refused Loans']!=0]
merged_df1.head()
merged_df1[merged_df1['TARGET']==0].head()