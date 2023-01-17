# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/startup_funding.csv')
data.head()
#Find the datatypes
data.info()
data.describe(include='all')
#clean up the date columns
data['Date']=pd.to_datetime(data['Date'].str.replace('.','/').str.replace('//','/'))
data['month_year']=data['Date'].dt.strftime('%Y-%m')
#clean up the amount column
data['AmountInUSD']=data['AmountInUSD'].str.replace(',','').astype('float')
data = data.rename(columns={'AmountInUSD': 'amount'})
# Clean up InvestmentType column
data['InvestorsName']=data['InvestorsName'].str.lower().str.replace(' ','')
# Clean up InvestmentType column
data['InvestmentType']=data['InvestmentType'].str.lower().str.replace(' ','')
#function to get numerical columns
get_numerical_columns=lambda df: list(df._get_numeric_data().columns)
#function to get date columns
def get_date_columns(df):
    date_cols=[col for col in df.columns if df[col].dtype == 'datetime64[ns]']
    return date_cols
#Function to get categorical columns
def get_categorical_columns(df):
    num_cols=get_numerical_columns(df)
    cols=df.columns
    cat_cols=[col for col in cols if col not in num_cols]
    date_cols=get_date_columns(data)
    cat_cols=[col for col in cat_cols if col not in date_cols]
    cat_cols.remove('StartupName')
    cat_cols.remove('Remarks')
    return cat_cols
data.head()
#Find numerical columns
get_numerical_columns(data)
#find Date columns
date_cols=get_date_columns(data)
date_cols
#get categorical columns
cat_cols=get_categorical_columns(data)
cat_cols
#plot to check missing values
import missingno
missingno.matrix(data)
#To find missing values in all columns in terms of percentage

pd.isnull(data).sum()/data.shape[0]*100

# The missing values in Columns Subvertical, Amount and Remarks amounts to huge percentages of data and 
# hence the values cannot be imputed. The analysis will be done as is.
data['amount'].plot.box( figsize=(16,10))
#Industry Vertical
x=data['IndustryVertical'].value_counts()/data.shape[0]*100
x.head(20).plot.bar(figsize=(12,5),color='SteelBlue')

# The Industry Verticals  Consumer Internet has the highest number of investments
#Sub vertical
x=data['SubVertical'].value_counts()/data.shape[0]*100
x.head(20).plot.bar(figsize=(12,5),color='SteelBlue')


#The SubVerticals Online pharmacy and Delivery Platforms has highest no of invetsments
#City Location
x=data['CityLocation'].value_counts()/data.shape[0]*100
x.head(20).plot.bar(figsize=(12,5),color='SteelBlue')

#Bangalore,Mumbai, New Delhi and Gurgaon are the cities with highest no of investments
#Investors Names
x=data['InvestorsName'].value_counts()/data.shape[0]*100
x.head(20).plot.bar(figsize=(12,5),color='SteelBlue')

#Undisclosed Invertors are major no of investors
#Investments Type Names
x=data['InvestmentType'].value_counts()/data.shape[0]*100
x.head(20).plot.bar(figsize=(12,5),color='SteelBlue')

#SeedFunding and Crowd Funding are the major Investment Types 
# USER DEFINED FUNCTION to perform One Way ANOVA

def test_oneway_anova(df, KPI, cat_col):
    from scipy.stats import f_oneway
    
    #Create a keys list from unique values in categorical column
    categories=df[cat_col].unique()
    #Create an empty dictionay
    groups={}
    
    #for every role in categories find the dataset for each
    for role in categories:
        subgroup=df[df[cat_col]==role][KPI].values
        groups[role]=subgroup
    
    stat, prob=f_oneway(*groups.values()) #kwargs -Single star stands for
    return (stat,prob)
   
# Run for all categorical columns for MonthlyIncome and find the columns which influences it

KPI='amount'

for cat_col in cat_cols:
    stat,prob=test_oneway_anova(data,KPI ,cat_col )
    #print('\nTest Statistic=',stat)
    #print('p Cal-Value=',prob)

    if prob>0.05:
        print('\nAccept Null Hypothesis. %s does not Influences %s' %(cat_col ,KPI))
    else:
        print('\nReject Null Hypothesis. %s influences %s' %(cat_col ,KPI))
                                                             
                                            

# Function to perform chi-Square- test

def test_chi_square(df,col1,col2):
    from scipy.stats import chi2_contingency
    obs=df.groupby([col1,col2]).size()
    obs.name='Freq'
    obs=obs.reset_index()
    obs=obs.pivot_table(index=col1,columns=col2,values="Freq")
    stat, p, dof, exp =chi2_contingency(obs.fillna(0).values)
    return(p)
def cat_columns_dependency_test(df):
    
    #step1: Identify Categorical Columns
    cat_cols=get_categorical_columns(df)
    
    #Step2: Identify combination of categorical columns
    cat_cols_combi=[]
    for i in range(0,len(cat_cols)):
        for j in range(i+1,len(cat_cols)):
            cat_cols_combi.append([cat_cols[i],cat_cols[j]])
     
    #Step3: For each combination of columns, identify if they are independent or not using chi-square test 
    dependent_cols=[]
    for combi in cat_cols_combi:
        p=test_chi_square(df,combi[0],combi[1])
        if p<0.05:
            #Function Output: Return those combination of column names alone which are dependent on each other
            dependent_cols.append([combi[0],combi[1]])
    return dependent_cols

#columns which are dependent on each other
dependent_cols=cat_columns_dependency_test(data)

for cols in dependent_cols:
    print('Columns %s and %s are Dependent on each other'%(cols[0],cols[1]))

