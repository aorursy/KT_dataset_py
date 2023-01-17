#Import the required Libraries.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import date, timedelta

pd.set_option('display.max_rows', 1000)

pd.set_option('display.max_columns', 1000)

pd.set_option('display.width', 5000)



#Read the data in a dataframe

inp1= pd.read_csv(r"../input/loan-defaulter/previous_application.csv")

inp2= pd.read_csv(r"../input/loan-defaulter/application_data.csv")

cols_data= pd.read_csv(r"../input/loan-defaulter/columns_description.csv",encoding= 'unicode_escape')
cols_data.head()
cols_data.rename(columns = {'ï»¿':'Serial No'}, inplace = True) 

cols_data.set_index('Serial No')

# To know the Shape of the Dataset we are going to explore

inp2.shape
#Lets, now see the columns we have and their dataypes and stats 

inp2.info()
#View sample data to see how the data set look like 

inp2.head(10)
inp2.describe()
# Cleaning the data 

# Exlpore for null values 

nullcolumns=inp2.isnull().sum()

nullcolumns
# To find the percentage of null values in the above columns we have the null counts displayed 

##To find the columns having more than 50% null values 

nullcolumns=inp2.isnull().sum()

nullcolumns=nullcolumns[nullcolumns.values>(0.5*len(nullcolumns))]

nullcolumns
#Drop the Null values 



nullcolumns = list(nullcolumns[nullcolumns.values>=0.3].index)

inp2.drop(labels=nullcolumns,axis=1,inplace=True)

print(len(nullcolumns))
#Check for percantage of null values again to ensure we have no NaN's in data set 



print((100*(inp2.isnull().sum()/len(inp2))))
#Box Plot check for Outliers 

sns.boxplot(inp2.AMT_ANNUITY)

plt.show()

plt.savefig('sample.jpg')
#Plot to see outliers in AMT_CREDIT 

sns.distplot(inp2.AMT_CREDIT)

plt.show()
#Plot to see outliers in AMT_INCOME_TOTAL 

plt.figure(figsize=[8,2])

sns.boxplot(inp2.AMT_INCOME_TOTAL)

plt.show()



# make boxplot with Seaborn

bplot=sns.boxplot(inp2.AMT_INCOME_TOTAL, 

                 width=0.5,

                 palette="colorblind")

 

# add stripplot to boxplot with Seaborn

bplot=sns.stripplot(inp2.AMT_INCOME_TOTAL,  

                   jitter=True, 

                   marker='o', 

                   alpha=0.5,

                   color='black')

#To find the median for the fiel AMT_ANNUITY

values=inp2['AMT_ANNUITY'].median()



values
# Fill the above value 24903 for all the missing values in AMT_ANNUITY

inp2.loc[inp2['AMT_ANNUITY'].isnull(),'AMT_ANNUITY']=values
#Check for percantage of null values again to ensure we have no NaN's in data set 



print((100*(inp2.isnull().sum()/len(inp2))))


# Removing rows having null values greater than or equal to 50%



nullrows=inp2.isnull().sum(axis=1)

nullrows=list(nullrows[nullrows.values>=0.5*len(inp2)].index)

inp2.drop(labels=nullrows,axis=0,inplace=True)

print(len(nullrows))
#To Check the dataype of all the columns 

inp2.head(10)
# We will remove unwanted columns from this dataset



unwanted=['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',

       'FLAG_PHONE', 'FLAG_EMAIL','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','FLAG_EMAIL','CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',

       'REGION_RATING_CLIENT_W_CITY','DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',

       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',

       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',

       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']



inp2.drop(labels=unwanted,axis=1,inplace=True)
inp2.info()
#view Sample data frame 

inp2.head(10)
#To handle -ve values in the DAYS columns in the inp2 Dataframe

inp2 = inp2.apply(lambda x: x*-1 if x.name in ['DAYS_BIRTH', 'DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH'] else x)

inp2.head(10)
inpsample=inp2.head(10)
#inp2.drop([Current_Date],axis=1,inplace=True)

#inp2['date_only'] = inp2['Current_date'].dt.date

#inp2.head(10)

#inp2.drop(['Current_Date'], axis = 1,inplace=True) 
#Add CurrentDate value to Data frame inp2

inp2['Current_date'] = pd.to_datetime('today',utc=False)

inp2['Current_date'] = inp2['Current_date'].dt.date
inp2.head()
#Categorical columns having these 'XNA' values

    

#CODE_GENDER 

inp2[inp2['CODE_GENDER']=='XNA'].shape
# Organization column



inp2[inp2['ORGANIZATION_TYPE']=='XNA'].shape
# Describing the Gender column to check the number of females and males



inp2['CODE_GENDER'].value_counts()
# Updating the column 'CODE_GENDER' with "F" for the dataset



inp2.loc[inp2['CODE_GENDER']=='XNA','CODE_GENDER']='F'

inp2['CODE_GENDER'].value_counts()
# Describing the organization type column



inp2['ORGANIZATION_TYPE'].describe()
# Hence, dropping the rows of total 55374 have 'XNA' values in the organization type column



inp2=inp2.drop(inp2.loc[inp2['ORGANIZATION_TYPE']=='XNA'].index)

inp2[inp2['ORGANIZATION_TYPE']=='XNA'].shape


# Casting all variable into numeric in the dataset



numeric_columns=['TARGET','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','REGION_POPULATION_RELATIVE','DAYS_BIRTH',

                'DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','HOUR_APPR_PROCESS_START','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',

       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']



inp2[numeric_columns]=inp2[numeric_columns].apply(pd.to_numeric)



inp2.head()
inp2.info()
#These Bins are created to explore insights by cutting the amounts into specific class intervals 

#Creating bins for income amount



bins = [0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,425000,450000,475000,500000,10000000000]

slot = ['0-25000', '25000-50000','50000-75000','75000,100000','100000-125000', '125000-150000', '150000-175000','175000-200000',

       '200000-225000','225000-250000','250000-275000','275000-300000','300000-325000','325000-350000','350000-375000',

       '375000-400000','400000-425000','425000-450000','450000-475000','475000-500000','500000 and above']



inp2['AMT_INCOME_RANGE']=pd.cut(inp2['AMT_INCOME_TOTAL'],bins,labels=slot)
# Creating bins for Credit amount



bins = [0,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000,700000,750000,800000,850000,900000,1000000000]

slots = ['0-150000', '150000-200000','200000-250000', '250000-300000', '300000-350000', '350000-400000','400000-450000',

        '450000-500000','500000-550000','550000-600000','600000-650000','650000-700000','700000-750000','750000-800000',

        '800000-850000','850000-900000','900000 and above']



inp2['AMT_CREDIT_RANGE']=pd.cut(inp2['AMT_CREDIT'],bins=bins,labels=slots)
# Dividing the dataset into two dataset of  target=1(client with payment difficulties) and target=0(all other)



target0=inp2.loc[inp2["TARGET"]==0]

target1=inp2.loc[inp2["TARGET"]==1]
target0.head(10)
target1.head(10)


# Calculating Imbalance percentage

    

# Since the majority is target0 and minority is target1



round(len(target0)/len(target1),2)
# Count plotting in logarithmic scale



def uniplot(df,col,title,hue =None):

    

    sns.set_style('whitegrid')

    sns.set_context('talk')

    plt.rcParams["axes.labelsize"] = 20

    plt.rcParams['axes.titlesize'] = 22

    plt.rcParams['axes.titlepad'] = 30

    

    

    temp = pd.Series(data = hue)

    fig, ax = plt.subplots()

    width = len(df[col].unique()) + 7 + 4*len(temp.unique())

    fig.set_size_inches(width , 8)

    plt.xticks(rotation=45)

    plt.yscale('log')

    plt.title(title)

    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='magma') 

        

    plt.show()
# PLotting for income range



uniplot(target0,col='AMT_INCOME_RANGE',title='Distribution of income range',hue='CODE_GENDER')
# Plotting for Income type



uniplot(target0,col='NAME_INCOME_TYPE',title='Distribution of Income type',hue='CODE_GENDER')


# Plotting for Contract type



uniplot(target0,col='NAME_CONTRACT_TYPE',title='Distribution of contract type',hue='CODE_GENDER')


# Plotting for Organization type in logarithmic scale



sns.set_style('whitegrid')

sns.set_context('talk')

plt.figure(figsize=(15,30))

plt.rcParams["axes.labelsize"] = 20

plt.rcParams['axes.titlesize'] = 22

plt.rcParams['axes.titlepad'] = 30



plt.title("Distribution of Organization type for target - 0")



plt.xticks(rotation=90)

plt.xscale('log')



sns.countplot(data=target0,y='ORGANIZATION_TYPE',order=target0['ORGANIZATION_TYPE'].value_counts().index,palette='cool')



plt.show()

# PLotting for income range



uniplot(target1,col='AMT_INCOME_RANGE',title='Distribution of income range',hue='CODE_GENDER')
# Plotting for Income type



uniplot(target1,col='NAME_INCOME_TYPE',title='Distribution of Income type',hue='CODE_GENDER')

# Plotting for Contract type



uniplot(target1,col='NAME_CONTRACT_TYPE',title='Distribution of contract type',hue='CODE_GENDER')
# Plotting for Organization type



sns.set_style('whitegrid')

sns.set_context('talk')

plt.figure(figsize=(15,30))

plt.rcParams["axes.labelsize"] = 20

plt.rcParams['axes.titlesize'] = 22

plt.rcParams['axes.titlepad'] = 30



plt.title("Distribution of Organization type for target - 1")



plt.xticks(rotation=90)

plt.xscale('log')



sns.countplot(data=target1,y='ORGANIZATION_TYPE',order=target1['ORGANIZATION_TYPE'].value_counts().index,palette='cool')



plt.show()
# Finding some correlation for numerical columns for both target 0 and 1 



target0_corr=target0.iloc[0:,2:]

target1_corr=target1.iloc[0:,2:]



target0cr=target0_corr.corr(method='spearman')

target1cr=target1_corr.corr(method='spearman')

# Correlation for target 0



target0cr
# Correlation for target 1



target1cr
# Now, plotting the above correlation with heat map as it is the best choice to visulaize



# figure size



def targets_corr(data,title):

    plt.figure(figsize=(15, 10))

    plt.rcParams['axes.titlesize'] = 25

    plt.rcParams['axes.titlepad'] = 70



# heatmap with a color map of choice





    sns.heatmap(data, cmap="RdYlGn",annot=False)



    plt.title(title)

    plt.yticks(rotation=0)

    plt.show()



# For Target 0



targets_corr(data=target0cr,title='Correlation for target 0')
# For Target 1



targets_corr(data=target1cr,title='Correlation for target 1')




# Box plotting for univariate variables analysis in logarithmic scale



def univariate_numerical(data,col,title):

    sns.set_style('whitegrid')

    sns.set_context('talk')

    plt.rcParams["axes.labelsize"] = 20

    plt.rcParams['axes.titlesize'] = 22

    plt.rcParams['axes.titlepad'] = 30

    

    plt.title(title)

    plt.yscale('log')

    sns.boxplot(data =target0, x=col,orient='v')

    plt.show()


# Distribution of income amount



univariate_numerical(data=target0,col='AMT_INCOME_TOTAL',title='Distribution of income amount')
# Disrtibution of credit amount



univariate_numerical(data=target0,col='AMT_CREDIT',title='Distribution of credit amount')
#Plot to see outliers in AMT_INCOME_TOTAL 

plt.figure(figsize=[8,2])

sns.boxplot(inp2.AMT_CREDIT)

plt.show()



# make boxplot with Seaborn

bplot=sns.boxplot(inp2.AMT_CREDIT, 

                 width=0.5,

                 palette="colorblind")

 

# add stripplot to boxplot with Seaborn

bplot=sns.stripplot(inp2.AMT_CREDIT,  

                   jitter=True, 

                   marker='o', 

                   alpha=0.5,

                   color='black')

# Distribution of anuuity amount



univariate_numerical(data=target0,col='AMT_ANNUITY',title='Distribution of Annuity amount')
# Distribution of income amount



univariate_numerical(data=target1,col='AMT_INCOME_TOTAL',title='Distribution of income amount')
# Distribution of credit amount



univariate_numerical(data=target1,col='AMT_CREDIT',title='Distribution of credit amount')
# Distribution of Annuity amount



univariate_numerical(data=target1,col='AMT_ANNUITY',title='Distribution of Annuity amount')
# Box plotting for Credit amount



plt.figure(figsize=(16,12))

plt.xticks(rotation=45)

sns.boxplot(data =target0, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient='v')

plt.title('Credit amount vs Education Status')

plt.show()
# Box plotting for Income amount in logarithmic scale



plt.figure(figsize=(16,12))

plt.xticks(rotation=45)

plt.yscale('log')

sns.boxplot(data =target0, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v')

plt.title('Income amount vs Education Status')

plt.show()
# Box plotting for credit amount



plt.figure(figsize=(16,12))

plt.xticks(rotation=45)

sns.boxplot(data =target1, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient='v')

plt.title('Credit Amount vs Education Status')

plt.show()
# Box plotting for Income amount in logarithmic scale



plt.figure(figsize=(16,12))

plt.xticks(rotation=45)

plt.yscale('log')

sns.boxplot(data =target1, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v')

plt.title('Income amount vs Education Status')

plt.show()
#Reading the Data Dictionary for columns in previous application Data 

cols_data.rename(columns = {'Unnamed: 0':'Serial No'}, inplace = True) 

cols_data.set_index('Serial No')

#To check the shape of Previous application Dataframe 



inp1.shape
# Check for the columns in the inp1 Dataframe (Previous application)- henceforth called as inp1

inp1.columns
# Check the column stats

inp1.info()
# Describe to see the mean and min and max value in dataframe inp1

inp1.describe()
# Check for sample data from inp1



inp1.head(10)
# Cleaning the missing data



# listing the null values columns having more than 30%



emptycol1=inp1.isnull().sum()

emptycol1=emptycol1[emptycol1.values>(0.3*len(emptycol1))]

len(emptycol1)
# Cleaning the data 

# Exlpore for null values 

nullcolumns=inp1.isnull().sum()

nullcolumns
#Check the percentage of null values in the columns of inp1 dataframe 

round(inp1.isnull().sum()/len(inp1)*100,2)
##To find the columns having more than 50% null values 

emptynullcol=inp1.isnull().sum()

emptynullcol=emptynullcol[emptynullcol.values>(0.5*len(emptynullcol))]

emptynullcol
#Drop the Null values 



emptynullcol = list(emptynullcol[emptynullcol.values>=0.5].index)

inp1.drop(labels=emptynullcol,axis=1,inplace=True)

print(len(emptynullcol))
#Check the percentage of null values in the columns of inp1 dataframe after dropping few columns greater than 50% null

round(inp1.isnull().sum()/len(inp1)*100,2)
#Checck for XNA and XAP in Column NAME_CASH_LOAN_PURPOSE

inp1.NAME_CASH_LOAN_PURPOSE.value_counts()
# Removing the column values of 'XNA' and 'XAP'



inp1=inp1.drop(inp1[inp1['NAME_CASH_LOAN_PURPOSE']=='XNA'].index)

inp1=inp1.drop(inp1[inp1['NAME_CASH_LOAN_PURPOSE']=='XAP'].index)



inp1.NAME_CASH_LOAN_PURPOSE.value_counts()
# Check for shape after XNA and XAP handling 

inp1.shape
# Check for sample data in inp1

inp1.head(20)
#check Column info

inp1.info()
#Merging the Application dataset with previous appliaction dataset



Master=pd.merge(left=inp2,right=inp1,how='inner',on='SK_ID_CURR',suffixes='_x')
# Renaming the column names after merging



master1= Master.rename({'NAME_CONTRACT_TYPE_' : 'NAME_CONTRACT_TYPE','AMT_CREDIT_':'AMT_CREDIT','AMT_ANNUITY_':'AMT_ANNUITY',

                         'WEEKDAY_APPR_PROCESS_START_' : 'WEEKDAY_APPR_PROCESS_START',

                         'HOUR_APPR_PROCESS_START_':'HOUR_APPR_PROCESS_START','NAME_CONTRACT_TYPEx':'NAME_CONTRACT_TYPE_PREV',

                         'AMT_CREDITx':'AMT_CREDIT_PREV','AMT_ANNUITYx':'AMT_ANNUITY_PREV',

                         'WEEKDAY_APPR_PROCESS_STARTx':'WEEKDAY_APPR_PROCESS_START_PREV',

                         'HOUR_APPR_PROCESS_STARTx':'HOUR_APPR_PROCESS_START_PREV'}, axis=1)
# Check Sample data in master dataframe after merge of inp1 and inp1

Master.head()
#check for columns in master dataframe 

Master.columns
master1.head()
master1.columns
# Removing unwanted columns for analysis



master1.drop(['SK_ID_CURR','WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION', 

              'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',

              'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY','WEEKDAY_APPR_PROCESS_START_PREV',

              'HOUR_APPR_PROCESS_START_PREV', 'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY'],axis=1,inplace=True)
#view sample data after dropping unwanted columns 

master1.head()
# View columns 

master1.info()
master1.columns
# Distribution of contract status in logarithmic scale

sns.set_style('whitegrid')

sns.set_context('talk')



plt.figure(figsize=(15,30))

plt.rcParams["axes.labelsize"] = 20

plt.rcParams['axes.titlesize'] = 22

plt.rcParams['axes.titlepad'] = 30

plt.xticks(rotation=90)

plt.xscale('log')

plt.title('Distribution of contract status with purposes')

ax = sns.countplot(data =master1, y= 'NAME_CASH_LOAN_PURPOSE', 

                   order=master1['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue ='NAME_CONTRACT_STATUS',palette='magma')
# Distribution of contract status



sns.set_style('whitegrid')

sns.set_context('talk')



plt.figure(figsize=(15,30))

plt.rcParams["axes.labelsize"] = 20

plt.rcParams['axes.titlesize'] = 22

plt.rcParams['axes.titlepad'] = 30

plt.xticks(rotation=90)

plt.xscale('log')

plt.title('Distribution of purposes with target ')

ax = sns.countplot(data = master1, y= 'NAME_CASH_LOAN_PURPOSE', 

                   order=master1['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'TARGET',palette='magma')
# Box plotting for Credit amount in logarithmic scale



plt.figure(figsize=(16,12))

plt.xticks(rotation=90)

plt.yscale('log')

sns.barplot(data =master1, x='NAME_CASH_LOAN_PURPOSE',hue='NAME_INCOME_TYPE',y='AMT_CREDIT_PREV',orient='v')

plt.title('Prev Credit amount vs Loan Purpose')

plt.show()
# Box plotting for Credit amount prev vs Housing type in logarithmic scale



plt.figure(figsize=(16,12))

plt.xticks(rotation=90)

sns.barplot(data =master1, y='AMT_CREDIT_PREV',hue='TARGET',x='NAME_HOUSING_TYPE')

plt.title('Prev Credit amount vs Housing type')

plt.show()