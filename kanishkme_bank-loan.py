# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from pandas_profiling import ProfileReport

import seaborn as sns

import matplotlib.ticker as ticks

import random as rd
pd.set_option('max_rows',None)

pd.set_option('display.max_columns',200)

sns.set_style('whitegrid')
app = pd.read_csv('/kaggle/input/bank-loan-risk-analysis/application_data.csv')

app.head()
app.info(verbose=True)
missing_val_app = app.isnull().sum()
# Converting the missing values into percentage and wrap them into data frame



missing_perc_app = pd.DataFrame({'Columns':missing_val_app.index,'Percentage':(missing_val_app.values/app.shape[0])*100})

missing_perc_app
# Plot the missing value percentage



plt.figure(figsize=(20,7))

sns.pointplot(data=missing_perc_app,x='Columns',y='Percentage')

plt.axhline(50,color='r',linestyle='--')

plt.title('Missing % in Application Dataset',fontsize=30)

plt.xticks(rotation=90)

plt.show()
# Columns with more than 50% missing values



missing_more_50 = missing_perc_app[missing_perc_app['Percentage']>=50]

missing_more_50
app1 = app.drop(columns=missing_more_50.Columns.to_list())

app1.head()
# Remove the columns which are not necessary



app1 = app1.drop(columns=['FLAG_DOCUMENT_2',

       'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',

       'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',

       'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',

       'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',

       'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',

       'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',

       'FLAG_DOCUMENT_21','FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE',

       'FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','DAYS_LAST_PHONE_CHANGE',

       'REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION',

       'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY','EXT_SOURCE_2',

       'EXT_SOURCE_3','REGION_RATING_CLIENT_W_CITY','YEARS_BEGINEXPLUATATION_AVG','FLOORSMAX_AVG',

       'YEARS_BEGINEXPLUATATION_MODE','FLOORSMAX_MODE','YEARS_BEGINEXPLUATATION_MEDI',

       'FLOORSMAX_MEDI','TOTALAREA_MODE','EMERGENCYSTATE_MODE'])

app1.head()
# Check for the missing values in the columns where percentage is less than 50

less_50_missing = (app1.isnull().sum()/app1.shape[0])*100

less_50_missing[less_50_missing>0]
app1.AMT_ANNUITY.describe()
app1.AMT_GOODS_PRICE.describe()
app1.NAME_TYPE_SUITE.value_counts(normalize=True)
# Check the OCCUPATION_TYPE variable



app1.OCCUPATION_TYPE.value_counts(normalize=True)
# Checking the percentage of the missing data for OCCUPATION_TYPE variable

(app1.OCCUPATION_TYPE.isna().sum() / app1.shape[0])*100
app1.CNT_FAM_MEMBERS.value_counts()
# To check client ID for duplicate values



app1.SK_ID_CURR.nunique()
# Again checking the summary statistics for the numerical columns

app1.describe()
# Convert the Days_Birth column



app1.DAYS_BIRTH = app1.DAYS_BIRTH.abs()
# Convert the DAYS_EMPLOYED column



app1.DAYS_EMPLOYED = app1.DAYS_EMPLOYED.abs()
# Convert the DAYS_REGISTRATION column



app1.DAYS_REGISTRATION = app1.DAYS_REGISTRATION.abs()
# Convert the DAYS_ID_PUBLISH column



app1.DAYS_ID_PUBLISH = app1.DAYS_ID_PUBLISH.abs()
# Now to see if we can calculate the age of the client 



app1['Age_Range'] = (app1.DAYS_BIRTH / 365).round(2)

app1.head()
# Taking the bins and the labels

bins = [0,30,40,50,60,100]

labels = ['<30','30-40','40-50','50-60','60+']
# Binnig the age column

app1['AGE_RANGE'] = pd.cut(app1.Age_Range,bins=bins,labels=labels)
# Removing Age_Range column after making bins out of it

app1.drop(columns='Age_Range',inplace=True)
# Checking the Gender Column

app1.CODE_GENDER.value_counts()
app1.CODE_GENDER.replace('XNA',np.nan,inplace=True)
app1['ORGANIZATION_TYPE'].value_counts()
# Setting the seaborn style

sns.set_style('white')

app1.describe()
fig = plt.figure(figsize=(20,5))

ax = sns.boxplot(data=app1.AMT_INCOME_TOTAL,orient='h',color='r')

ax.set_xscale('linear')

ax.xaxis.set_major_locator(ticks.MultipleLocator(2000000))

ax.set_title('Income Total Amount',fontsize=50)

plt.xticks(rotation=90)

plt.show()
app1.AMT_INCOME_TOTAL.describe()
app1.AMT_INCOME_TOTAL.quantile([0.33,.34,.66,0.90,0.92,0.94,0.95,0.96,0.98,0.99,1])
bins = [0, 100000, 200000, 300000, 400000,200000000]

labels = ['Very Low','Low','Medium','High','Very High']
app1['INCOME_RANGE'] = pd.cut(app1.AMT_INCOME_TOTAL,bins=bins,labels=labels)
# Checking the AMT_ANNUITY variable



fig = plt.figure(figsize=(20,5))

ax = sns.boxplot(data=app1.AMT_ANNUITY,orient='h')

ax.set_xscale('linear')

ax.xaxis.set_major_locator(ticks.MultipleLocator(10000))

ax.set_title('Annuity Amount',fontsize=50)

plt.xticks(rotation=90)

plt.show()
app1.AMT_ANNUITY.describe()
app1['AMT_ANNUITY'].quantile([0.33,0.66,0.9,0.92,0.94,0.95,0.96,0.98,0.99,1.0])
# Setting up bins and labels for AMT_ANNUITY



bins = [0,10000,20000,30000,40000,500000]

labels = ['Very Low','Low','Medium','High','Very High']



app1['ANNUITY_RANGE'] = pd.cut(app1['AMT_ANNUITY'],bins=bins,labels=labels)
fig = plt.figure(figsize=(20,5))

ax = sns.boxplot(data=app1.AMT_GOODS_PRICE,orient='h',color='brown')

ax.set_xscale('linear')

ax.xaxis.set_major_locator(ticks.MultipleLocator(100000))

ax.set_title('Goods Amount',fontsize=50)

plt.xticks(rotation = 90)

plt.show()
app1.AMT_GOODS_PRICE.describe()
app1.AMT_GOODS_PRICE.quantile([0.75,0.80,0.85,0.90,0.95,0.96,0.97,0.98,0.99,1])
app1.AMT_CREDIT.describe()
fig = plt.figure(figsize=(20,5))

ax = sns.boxplot(data=app1.AMT_CREDIT,orient='h',color='orange')

ax.set_xscale('linear')

ax.xaxis.set_major_locator(ticks.MultipleLocator(100000))

ax.set_title('Credit Amount',fontsize=50)

plt.show()
app1.AMT_CREDIT.quantile([0.20,.33,.66,.90,.92,.94,.95,.96,.98,.99,1])
# Creating bins and labels for AMT_CREDIT column

bins = [0,300000,600000,900000,1200000,5000000]

labels = ['Very Low','Low','Medium','High','Very High']
# Binning the AMT_CREDIT column

app1['CREDIT_RANGE'] = pd.cut(x=app1.AMT_CREDIT,bins=bins,labels=labels)
fig = plt.figure(figsize=(20,5))

ax = sns.boxplot(data=app1.DAYS_BIRTH,orient='h',color='darkcyan')

ax.set_xscale('linear')

ax.xaxis.set_major_locator(ticks.MultipleLocator(1000))

ax.set_title('Age',fontsize=50)

plt.show()
app1.DAYS_EMPLOYED.describe()
fig = plt.figure(figsize=(20,5))

ax = sns.boxplot(app1.DAYS_EMPLOYED,orient='h',color='yellow')

ax.set_title('Days Employed',fontsize=50)

ax.xaxis.set_major_locator(ticks.MultipleLocator(10000))

plt.xticks(rotation=90)

plt.show()
years = 365243 / 365

years
app1['DAYS_EMPLOYED'].quantile([.8,0.82,0.95,0.98,0.99,1.0])
app1['DAYS_EMPLOYED'].value_counts().sort_index().tail(5)
fig = plt.figure(figsize=(20,5))



ax = sns.boxplot(app1.DAYS_ID_PUBLISH,orient='h',color='olive')



ax.set_title('Days Id Publish',fontsize=50)



ax.xaxis.set_major_locator(ticks.MultipleLocator(1000))



plt.xticks(rotation=90)



plt.show()
# SOCIAL_CIRCLE Variables



app1.describe().loc[:,'OBS_30_CNT_SOCIAL_CIRCLE':'DEF_60_CNT_SOCIAL_CIRCLE']
# Checking for the SOCIAL CIRCLE columns 



l=['OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']

color=['red','yellow','black','blue','darkcyan','orange','green','olive','brown']



fig=plt.figure(figsize=(15,20))

fig.subplots_adjust(hspace = .2, wspace=.2)



for i in enumerate(l):

    plt.subplot(4,2,i[0]+1)

    sns.boxplot(y=app1[i[1]],color=rd.choice(color))

    plt.title(i[1],fontsize=10)

plt.show()
app1[app1.DEF_60_CNT_SOCIAL_CIRCLE==24]
# Checking the TARGET column for imbalance data in percentage

app1.TARGET.value_counts(normalize=True)*100
# Plotting the graphs



fig, (ax1,ax2) = plt.subplots(1,2,figsize =(20,8))



ax = sns.countplot(app1.TARGET,ax=ax1)



ax1.set_title('TARGET',fontsize=20)



plt.setp(ax1.xaxis.get_majorticklabels(),fontsize=18)



ax2 = plt.pie(x=app1.TARGET.value_counts(normalize=True),autopct='%.2f',textprops={'fontsize':15},shadow=True,labels=['No Payment Issues','Payment Issues'],wedgeprops = {'linewidth': 5}) 



plt.title('Distribution of the Target Variable',fontsize=20)



plt.show()
# Check the Imbalance Percentage



print('Imbalance Percentage is : %.2f'%(app1.TARGET.value_counts(normalize=True)[0]/app1.TARGET.value_counts(normalize=True)[1]))
app1.head()
# Setting the style for the plots

sns.set_style(style = 'whitegrid',rc={"grid.linewidth": 5})
app1.nunique().sort_values()
# Function for univariate analysis

def plots(l,rows=1,cols=1,rot=90):

        

    if cols>1:

        fig, (ax1,ax2) = plt.subplots(nrows=rows,ncols=cols,figsize=(30,10))

        fig.subplots_adjust(hspace = .2, wspace=.2)

    

    else:

        fig, (ax1,ax2) = plt.subplots(nrows=rows,ncols=cols,figsize=(30,30))

        fig.subplots_adjust(hspace = .5, wspace=.1)

    

    

    # Subplot 1 : countplot 

    first = sns.countplot(data = app1 , hue = 'TARGET', palette='inferno',x=l,ax=ax1)

    first.set_title(l,fontsize=30)

    first.set_yscale('log')

    first.legend(labels=['Loan Repayers','Loan Defaulters'],fontsize=20)

    plt.setp(first.xaxis.get_majorticklabels(), rotation=rot,fontsize=25)

    plt.setp(first.yaxis.get_majorticklabels(),fontsize=18)





    # Percentage of the mean values for defaulters

    default_percentage = (app1.groupby(by=l)['TARGET'].mean()*100).sort_values()

    

    # Subplot 2 : barplot

    sec = sns.barplot(x=default_percentage.index,y=default_percentage,ax=ax2)

    sec.set_title(f'Default % in {l}',fontsize=30)

    sec.set_yscale('linear')

    plt.setp(sec.xaxis.get_majorticklabels(), rotation=rot,fontsize=25)

    plt.setp(sec.yaxis.get_majorticklabels(),fontsize=18)

    return None

list_Cat_num = ['NAME_EDUCATION_TYPE','CREDIT_RANGE','INCOME_RANGE','ANNUITY_RANGE']



for i in list_Cat_num:

    plots(i,1,2,rot=50)
list_categories = ['AGE_RANGE','REGION_RATING_CLIENT','CNT_FAM_MEMBERS','CNT_CHILDREN']



for i in list_categories:

    plots(i,1,2,rot=0)
list_categories = ['FLAG_OWN_CAR','FLAG_OWN_REALTY','CODE_GENDER','NAME_CONTRACT_TYPE']



for val in list_categories:

    plots(val,1,2,rot=0)
list_categories = ['NAME_INCOME_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE']



for i in list_categories:

    plots(i,1,2,rot=50)
plots('OCCUPATION_TYPE',1,2)
# Plotting the ORGANIZATION_TYPE separately 



plots('ORGANIZATION_TYPE',rows=2)
# Dividing the data set into 2 parts target_0 : Loan Repayers and target_1 : Loan Defaulters



target_0 = app1[app1['TARGET']==0]



target_1 = app1[app1['TARGET']==1]
# Creating a function for bivariate analysis



def cat_num_bivar(x=None,y=None,hue=None,est=np.mean,rot=90):

    plt.style.use('dark_background')

    fig ,(ax1,ax2) = plt.subplots(1,2,figsize=(20,10))

    first = sns.barplot(data=target_0,x=x,y=y,hue=hue,palette='Set3',estimator=est,ax=ax1)

    first.set_title(f'Target 0 : {x} vs {y}',fontsize=20)

    plt.setp(first.xaxis.get_majorticklabels(),rotation=rot,fontsize=25)

    plt.setp(first.yaxis.get_majorticklabels(),fontsize=18)



    sec  = sns.barplot(data=target_1,x=x,y=y,hue=hue,palette='Set3',estimator=est,ax=ax2)

    sec.set_title(f'Target 1 : {x} vs {y}',fontsize=20)

    plt.setp(sec.xaxis.get_majorticklabels(),rotation=rot,fontsize=25)

    plt.setp(sec.yaxis.get_majorticklabels(),fontsize=18)



    plt.show()

    return None
cat_num_bivar(x='NAME_EDUCATION_TYPE',y='AMT_CREDIT')
# Plotting the graphs with estimator as median



cat_num_bivar('OCCUPATION_TYPE','AMT_INCOME_TOTAL',est=np.median)
# Plotting the graphs



cat_num_bivar('NAME_INCOME_TYPE','AMT_CREDIT')
# Plotting the graph with estimator as median



cat_num_bivar('NAME_HOUSING_TYPE','AMT_INCOME_TOTAL',est=np.median)
cat_num_bivar('CODE_GENDER','AMT_INCOME_TOTAL',est=np.median)
def cat_cat(data1,data2,x,hue,scale='linear',order=None,rot=90):

    plt.style.use('dark_background')

    fig= plt.figure(figsize=(20,10))

    plt.subplot(1,2,1)



    sns.countplot(data = data1,x=x,hue=hue,hue_order=order,palette='Spectral')

    plt.title(f'target 0 :{x} vs {hue}',fontsize=20)

    plt.yscale(scale)

    plt.xticks(rotation=rot,fontsize=25)

    plt.legend(fontsize=16)

    

    plt.subplot(1,2,2)

    sns.countplot(data = data2,x=x,hue=hue,hue_order=order,palette='Spectral')

    plt.title(f'target 1 :{x} vs {hue}',fontsize=20)

    plt.yscale(scale)

    plt.xticks(rotation=rot,fontsize=25)

    plt.legend(fontsize=16)

    plt.show()

    return None

    
cat_cat(target_0,target_1,'NAME_HOUSING_TYPE','CODE_GENDER','log',['M','F'])
cat_cat(target_0,target_1,'OCCUPATION_TYPE','NAME_FAMILY_STATUS',scale='log')
cat_cat(target_0,target_1,'NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','log')
cat_cat(target_0,target_1,'NAME_FAMILY_STATUS',hue='FLAG_OWN_REALTY',scale='log',order=['Y','N'],rot=70)
cat_cat(target_0,target_1,'INCOME_RANGE',hue='CODE_GENDER',scale='log',order=['M','F'],rot=0)
list_numerics=['AMT_ANNUITY','AMT_CREDIT','AMT_INCOME_TOTAL','AMT_GOODS_PRICE','DAYS_BIRTH']



def plots_numeric_univ(l):

        sns.set_style(style='whitegrid')

        fig=plt.figure(figsize=(50,5))

        plt.subplot(1,5,1)

        sns.distplot(target_0[l],hist=False,label='Loan Repayers',color='green')

        sns.distplot(target_1[l],hist=False,label='Loan Defaulters',color='red')

        plt.title(l,fontsize=20)

        plt.show()

        

for i in list_numerics:

    plots_numeric_univ(i)
# Creating the function for the numeric bivariate analysis



def plots_numeric_biv(x,y):

    sns.set_style(style='whitegrid')

    fig=plt.figure(figsize=(15,6))

    sns.scatterplot(data=target_0, y = y, x = x,label='Loan Repayers',color='darkcyan')

    sns.scatterplot(data=target_1, y = y, x = x,label='Loan Defaulters',color='red')

    plt.title(f'{x} vs {y}',fontsize=20)

    plt.show()
plots_numeric_biv('AMT_ANNUITY','AMT_GOODS_PRICE')
plots_numeric_biv('AMT_ANNUITY','AMT_CREDIT')
plots_numeric_biv('AMT_GOODS_PRICE','AMT_CREDIT')
# PLotting the pairplot 

l=['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']

sns.pairplot(data = app1, vars=l,hue='TARGET',palette=['b','orange'])

plt.xticks(fontsize=25)

plt.yticks(fontsize=25)

plt.show()
# Dropping the columns which are not required

cols_drop = ['SK_ID_CURR','AMT_REQ_CREDIT_BUREAU_HOUR',

       'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',

       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',

       'AMT_REQ_CREDIT_BUREAU_YEAR','OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',

       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']
# Correlation matrix

target_0 = target_0.drop(columns=cols_drop)



corr = target_0.corr().abs().round(3)

corr
# Visualise the correlation using heatmaps



fig = plt.figure(figsize=(20,15))



sns.heatmap(data=corr,annot=True,cmap='RdYlGn_r',linewidths=.5,center=0.1)



plt.show()
# Unstacking the TARGET_0 variable

c = corr.abs()

s=c.unstack()
# Finding top 10 correlation among the people with no payment issues



target_0_corr = s[s.index.get_level_values(0)!= s.index.get_level_values(1)].sort_values(ascending=False,kind='quicksort').drop_duplicates()



df = pd.DataFrame(target_0_corr)



df = df.reset_index().rename(columns={'level_0':'Var1','level_1':'Var2',0:'Correlation'}).dropna()



df.head(10)
# Drop the columns which are not required

target_1 = target_1.drop(columns=cols_drop)
# Creating the correlation matrix for the Loan defaulter data frame

corr_t1 = target_1.corr().abs().round(3)



corr_t1
# PLotting the heatmap 

fig = plt.figure(figsize=(20,15))



sns.heatmap(data=corr_t1,annot=True,cmap='RdYlGn_r',linewidths=0.5,center=0.1)



plt.show()
c1 = corr_t1



s1 = c1.unstack()
# Top 10 Correlations from target_1 : Loan Defaulter data frame



target_1_corr = s1[s1.index.get_level_values(0)!= s1.index.get_level_values(1)].sort_values(ascending=False,kind='quicksort').drop_duplicates()



df = pd.DataFrame(target_1_corr)



df = df.reset_index().rename(columns={'level_0':'Var1','level_1':'Var2',0:'Correlation'}).dropna()



df.head(10)
# Fetching the dataset



## 'app' variable takes the dataset application_data.csv as Data Frame



f = '/kaggle/input/bank-loan-risk-analysis/previous_application.csv'

# Count the lines

num_lines = sum(1 for l in open(f))

# Sample size - in this case ~50%

size = int(num_lines // 2)

# The row indices to skip - make sure 0 is not included to keep the header!

rd.seed(100)

skip_idx = rd.sample(range(1, num_lines), num_lines - size)

# Read the data

prev_app = pd.read_csv(f, skiprows=skip_idx)



prev_app.head()
# Shape of the dataset

prev_app.shape
# Missing values check

prev_app.isna().sum()
# Setting the style for the plots

sns.set_style(style='whitegrid')
# Converting the absolute missing values into percentage



missing_prev = ((prev_app.isna().sum() / prev_app.shape[0])*100)



cols1 = missing_prev.index.to_list()



vals1 = missing_prev.to_list()



missing_prev_app = pd.DataFrame({'Columns':cols1,'Missing_prev_Percentage':vals1})



missing_prev_app.sort_values(by='Missing_prev_Percentage')
# Plot for missing values in previous application data frame

fig = plt.figure(figsize=(20,8))



ax = sns.pointplot(data = missing_prev_app, x='Columns', y = missing_prev_app.Missing_prev_Percentage,color='blue')



ax.axhline(50,color='red',linestyle='--')



ax.set_title('Missing Values % for Previous Applications ',fontsize=20)



ax.set_xlabel('Columns',fontsize=15)



ax.set_ylabel('Missing Percentage',fontsize=15)



ax.set_xticklabels(labels=cols1,rotation=90)





plt.show()

# Dropping the columns with more than 50 % missing values

prev_app.drop(columns=missing_prev[missing_prev>=50].index,inplace=True)
prev_app.info()
prev_app.describe()
# Merge the datasets

merged_df = app1.merge(right=prev_app,on='SK_ID_CURR')
# Check the shape of the data

merged_df.shape
merged_df.head()
merged_df['DAYS_DECISION'] = merged_df['DAYS_DECISION'].abs()
merged_df.nunique().sort_values()
# Dividing the merged dataset into two parts on the basis of Loan Approval and Loan Refusal



df1= merged_df[merged_df['NAME_CONTRACT_STATUS']=='Approved']



df2= merged_df[merged_df['NAME_CONTRACT_STATUS']=='Refused']



df3= merged_df[merged_df['NAME_CONTRACT_STATUS']=='Canceled']
# Function to analyze the categorical columns



def cat_1(x,hue,scale='linear',order=None,rot=90,hspc=0.3):

    plt.figure(figsize=(25,25))

    

    plt.subplots_adjust(wspace=0.3,hspace=hspc)

    plt.subplot(2,2,1)

    sns.countplot(data = df1,x=x,hue=hue,hue_order=order,palette=['darkcyan','darkgrey'])

    plt.title(f'Approved :{x} ',fontsize=20)

    plt.yscale(scale)

    plt.xticks(rotation=rot,fontsize=18)

    plt.yticks(fontsize=15)

    plt.legend(labels=['Loan Repayers','Loan Defaulters'],fontsize=16)

        

    plt.subplot(2,2,2)

    sns.countplot(data = df2,x=x,hue=hue,hue_order=order,palette=['darkcyan','darkgrey'])

    plt.title(f'Refused :{x} ',fontsize=20)

    plt.yscale(scale)

    plt.xticks(rotation=rot,fontsize=18)

    plt.yticks(fontsize=15)

    plt.legend(labels=['Loan Repayers','Loan Defaulters'],fontsize=16)

    

    # default % in approval

    ax=(df1.groupby(by=x)['TARGET'].mean()*100).sort_values()

    

    

    plt.subplot(2,2,3)

    a = sns.barplot(x=ax.index,y=ax)

    a.set_title(f'Default % in Approval : {x}',fontsize=20)

    a.set_yscale('linear')

    plt.setp(a.xaxis.get_majorticklabels(), rotation=rot,fontsize=18)

    plt.setp(a.yaxis.get_majorticklabels(),fontsize=18)

    

    

    # default % in refused

    ax1=(df2.groupby(by=x)['TARGET'].mean()*100).sort_values()

    

    plt.subplot(2,2,4)

    a = sns.barplot(x=ax1.index,y=ax1)

    a.set_title(f'Default % in Refusal : {x}',fontsize=20)

    a.set_yscale('linear')

    plt.setp(a.xaxis.get_majorticklabels(), rotation=rot,fontsize=18)

    plt.setp(a.yaxis.get_majorticklabels(),fontsize=18)

    

    plt.show()

    return None
cat_1(x='NAME_CLIENT_TYPE',hue='TARGET',scale='log',rot=50,hspc=0.4)
cat_1(x='PRODUCT_COMBINATION',hue='TARGET',scale='log',hspc=0.7)
cat_1(x='NAME_SELLER_INDUSTRY',hue='TARGET',scale='log',hspc=0.5)
cat_1(x='NAME_PRODUCT_TYPE',hue='TARGET',scale='log',rot=50)
cat_1(x='NAME_CASH_LOAN_PURPOSE',hue='TARGET',scale='log',hspc=0.7)
cat_1(x='NAME_CONTRACT_TYPE_y',hue='TARGET',scale='log',rot=50)
merged_df.head()
# Funciton for numerical - categorical analysis



def cat_num_bivar_merged(x=None,y=None,hue=None,est=np.median,rot=90):

    plt.style.use('dark_background')

    

    fig = plt.figure(figsize=(30,13))

    plt.subplots_adjust(wspace=0.3)

    

    plt.subplot(1,3,1)

    first = sns.barplot(data=df1,x=x,y=y,hue=hue,palette='Set3',estimator=est)

    first.set_title(f'Approval : {x} vs {y}',fontsize=18)

    plt.setp(first.xaxis.get_majorticklabels(),rotation=rot,fontsize=25)

    plt.setp(first.yaxis.get_majorticklabels(),fontsize=18)

    



    plt.subplot(1,3,2)

    sec = sns.barplot(data=df2,x=x,y=y,hue=hue,palette='Set3',estimator=est)

    sec.set_title(f'Refusal : {x} vs {y}',fontsize=18)

    plt.setp(sec.xaxis.get_majorticklabels(),rotation=rot,fontsize=25)

    plt.setp(sec.yaxis.get_majorticklabels(),fontsize=18)

    

    plt.subplot(1,3,3)

    third = sns.barplot(data=df3,x=x,y=y,hue=hue,palette='Set3',estimator=est)

    third.set_title(f'Cancelled : {x} vs {y}',fontsize=18)

    plt.setp(third.xaxis.get_majorticklabels(),rotation=rot,fontsize=25)

    plt.setp(third.yaxis.get_majorticklabels(),fontsize=18) 

    

    plt.show()

    return None
cat_num_bivar_merged(x='NAME_PRODUCT_TYPE',y='AMT_GOODS_PRICE_y',rot=0)
cat_num_bivar_merged(x='NAME_SELLER_INDUSTRY',y='AMT_GOODS_PRICE_y',rot=90)
cat_num_bivar_merged(x='PRODUCT_COMBINATION',y='AMT_GOODS_PRICE_y',rot=90)
cat_num_bivar_merged(x='NAME_GOODS_CATEGORY',y='AMT_GOODS_PRICE_y',rot=90)
cat_num_bivar_merged(x='NAME_CASH_LOAN_PURPOSE',y='AMT_GOODS_PRICE_y',rot=90)
# Plotting the numeric variables 



list_cat = ['AMT_ANNUITY_y','AMT_GOODS_PRICE_y']



def plots_numeric_univ_merged(l):

        sns.set_style('whitegrid')

    

        fig=plt.figure(figsize=(30,6))

        plt.subplot(1,2,1)

        ax=sns.distplot(df1[l],hist=False,label='Approved',color='green')

        ax=sns.distplot(df2[l],hist=False,label='Refused',color='red')

        ax=sns.distplot(df3[l],hist=False,label='Canceled',color='orange')

        plt.title(l,fontsize=20)

        plt.xticks(fontsize=15) 

        plt.show()

        return None

        

for i in list_cat:

    plots_numeric_univ_merged(i)
ord1= ['Approved','Refused','Canceled','Unused offer']



def plots_numeric_biv_merged(x,y,order=None):

    

    fig=plt.figure(figsize=(20,10))

    

    plt.subplot(1,2,1)

    

    sns.scatterplot(data=merged_df, y = y, x = x,hue='NAME_CONTRACT_STATUS',palette=['g','r','yellow','white'],hue_order=order)

    

    plt.title(f'{x} vs {y}',fontsize=20)

    

    plt.xticks(fontsize=18)

    

    plt.yticks(fontsize=18)

    

    plt.show()

    

    return None
plots_numeric_biv_merged('AMT_ANNUITY_y','AMT_CREDIT_y',order=ord1)
plots_numeric_biv_merged('AMT_GOODS_PRICE_y','AMT_CREDIT_y',order=ord1)
plots_numeric_biv_merged('AMT_ANNUITY_y','AMT_GOODS_PRICE_y',order=ord1)
plots_numeric_biv_merged('AMT_GOODS_PRICE_y','AMT_APPLICATION',order=ord1)
(merged_df.groupby('NAME_CONTRACT_STATUS')['TARGET'].value_counts(normalize=True)*100)
fig = plt.figure(figsize=(20,8))

sns.countplot(data=merged_df,x='NAME_CONTRACT_STATUS',hue='TARGET',palette='rocket')

plt.legend(['Loan Repayers','Loan Defaulters'])

plt.yscale('log')

plt.xticks(fontsize=25)

plt.show()