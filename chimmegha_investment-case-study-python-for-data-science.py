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

#Importing required libraries for this analysis
import os;
import warnings
warnings.filterwarnings('ignore')
#Checking current working directory is the same as all thye files are kept
os.getcwd()
#Loading companies and rounds data into two dataframes
companies = pd.read_csv("../input/companies.txt", sep = '\t', encoding = "ISO-8859-1")
rounds2 = pd.read_csv("../input/rounds2.csv", encoding = "ISO-8859-1")
mapping = pd.read_csv("../input/mapping.csv")
#Checking if the dataframes are imported properly
companies.tail(10)
rounds2.tail(10)
print(companies.columns)
print(rounds2.columns)
#Removing Special Characters in both dataframes
companies.permalink = companies.permalink.str.encode('utf-8').str.decode('ascii', 'ignore')
companies.name = companies.name.str.encode('utf-8').str.decode('ascii', 'ignore')
rounds2.company_permalink = rounds2.company_permalink.str.encode('utf-8').str.decode('ascii', 'ignore')
rounds2.funding_round_permalink = rounds2.funding_round_permalink.str.encode('utf-8').str.decode('ascii', 'ignore')

companies.permalink = companies.permalink.str.lower()
companies.name = companies.name.str.lower()
rounds2.company_permalink = rounds2.company_permalink.str.lower()
rounds2.tail(10)
companies.tail(10)
#Performing exploratory data analysis 
rounds2.info()
companies.info()
print(rounds2.describe(), end='\n\n*********************************************\n\n')
print(companies.describe())
#Print the counts for unique companies in rounds2 and companies
print("Number of Unique Companies from rounds2: {0}".format(rounds2.company_permalink.nunique()))
print("Number of Unique Companies from companies: {0}".format(companies.permalink.nunique()))
#To check if companies in 'rounds2' dataframe are present in 'companies' dataframe
n = rounds2.assign(Incompanies = rounds2.company_permalink.isin(companies.permalink).astype(int))
len(n[n.Incompanies == 0])
#Merging the two dataframes rounds2 and companies 
 #Find total rows in master_frame
master_frame = pd.merge(left=rounds2,right=companies, how='outer', left_on='company_permalink', right_on='permalink')
print(master_frame.info(), end = '\n\n\n\n\n')
print(master_frame.describe())
master_frame.head(3)
#Dropping permalink from master_frame as company_permalink has same values
master_frame.drop(['permalink'], axis = 1, inplace = True)
print("List of Columns in Master_Frame: {0}".format(master_frame.columns))
#Derive unique funding round types in given data.
#master_frame[['funding_round_type','raised_amount_usd']]
round_types = master_frame['funding_round_type'].unique().tolist()
round_types
#Checking if filter for raised_amount_usd.isnull() returns any rows 
master_frame[master_frame['raised_amount_usd'].isnull()].head()
#Imputing null values
master_frame["raised_amount_usd"] = master_frame.groupby("funding_round_type").transform(lambda x: x.fillna(x.mean()))
#To check if null values are still present for raised_amount_usd column 
master_frame[master_frame['raised_amount_usd'].isnull()]
#Find out average raised amount for each funding_round_type
pd.set_option('display.float_format', lambda x: '%.2f' % x)
round_type_avg = master_frame.groupby('funding_round_type').mean().round(2)
round_type_avg
region = master_frame[master_frame.country_code.isnull()]['region'].unique().tolist()
region
#Keeping the rows where country_code value is not null
master_frame = master_frame[master_frame.country_code.notnull()]

#Since we are focusing on ventures only filtering dataframe for same
master_frame[master_frame.funding_round_type == "venture"][['raised_amount_usd', 'country_code']].head(5)
top9 = master_frame[master_frame.funding_round_type == "venture"].groupby('country_code').sum().reset_index().sort_values(by= 'raised_amount_usd', ascending=False)
top9 = top9.head(9).reset_index(drop = True)
top9 = top9.rename(index=str, columns={"country_code": "country", "raised_amount_usd": "total_funding_raised"})
#Print top9
top9
#In order to find top english speaking countries lets map them as "English/Non-English".
lng_dict = {
  "USA": "English",
  "CHN": "Non-English",
  "GBR": "English",
  "IND": "English",
  "CAN": "English",
  "DEU": "Non-English",
  "ISR": "Non-English",
  "FRA": "Non-English",
  "JPN": "Non-English",
  "NLD": "Non-English"
}
top9['Language'] = top9['country'].map(lng_dict)
#Converting Values in Millions
top9.total_funding_raised = top9.total_funding_raised/1000000
top9
mapping.head()
x = mapping.set_index('category_list',inplace=False).stack()
x = x[x!=0].to_frame().reset_index()
x.drop([0], axis = 1, inplace = True)
x = x.rename(index=str, columns={"category_list": "Category", "level_1": "main_sector"})


x.Category = x.Category.str.replace('0', 'na',regex=False)
x.head()
master_frame['primary_sector'] = master_frame['category_list'].str.split(pat= "|", expand=True)[0]
master_frame.head(3)
mapping.category_list = mapping.category_list.str.replace('0', 'na',regex=False)

master_frame = pd.merge(left=master_frame,right=x, how='left', left_on='primary_sector', right_on='Category')
#Creating Dataframe D1
D1 = master_frame[(master_frame.country_code == "USA") & (master_frame.funding_round_type == "venture") &
                                                     (master_frame.raised_amount_usd >= 5000000) & 
                                                     (master_frame.raised_amount_usd <= 15000000)].reset_index(drop = True)

D1['count_of_investments'] = D1['main_sector'].groupby(D1['main_sector']).transform('count')
D1['total_amt_invested'] = D1['raised_amount_usd'].groupby(D1['main_sector']).transform('sum')
D1.head(3)
#Total number of Investments in Country1
print(D1.company_permalink.count())

#Total amount of investmemts in Country1
print(D1.raised_amount_usd.sum(), end='\n\n********************************************\n\n')

#Top Sector name (no. of investment-wise) for Country1
print(D1.groupby('main_sector')['company_permalink'].count().reset_index().sort_values(by= 'company_permalink', ascending=False))

print(end = '\n\n*******************************************************************************************\n\n')

#For point 3 (top sector count-wise), which company received the highest investment?
print(D1[D1.main_sector == "Others"].groupby(['company_permalink','name'])[['raised_amount_usd']].
                    sum().reset_index().sort_values(by = 'raised_amount_usd', ascending = False).iloc[[0]])

print(end = '\n\n********************************************************************************************\n\n')

#For point 4 (second best sector count-wise), which company received the highest investment?
print(D1[D1.main_sector == "Social, Finance, Analytics, Advertising"].groupby(['company_permalink','name'])[['raised_amount_usd']].
                sum().reset_index().sort_values(by = 'raised_amount_usd', ascending = False).iloc[[0]])
#Creating Dataframe D2
D2 = master_frame[(master_frame.country_code == "GBR") & (master_frame.funding_round_type == "venture") &
                                                     (master_frame.raised_amount_usd >= 5000000) & 
                                                     (master_frame.raised_amount_usd <= 15000000)].reset_index(drop = True)

D2['count_of_investments'] = D2['main_sector'].groupby(D2['main_sector']).transform('count')
D2['total_amt_invested'] = D2['raised_amount_usd'].groupby(D2['main_sector']).transform('sum')
D2.head(3)
#Total number of Investments in Country2
print(D2.company_permalink.count())

#Total amount of investmemts in Country2
print(D2.raised_amount_usd.sum(), end='\n\n********************************************\n\n')

#Top Sector name (no. of investment-wise) for Country2
print(D2.groupby('main_sector')['company_permalink'].count().reset_index().sort_values(by= 'company_permalink', ascending=False))

print(end = '\n\n*******************************************************************************************\n\n')

#For point 3 (top sector count-wise), which company received the highest investment?
print(D2[D2.main_sector == "Others"].groupby(['company_permalink','name'])[['raised_amount_usd']].
                    sum().reset_index().sort_values(by = 'raised_amount_usd', ascending = False).iloc[[0]])

print(end = '\n\n********************************************************************************************\n\n')

#For point 4 (second best sector count-wise), which company received the highest investment?
print(D2[D2.main_sector == "Social, Finance, Analytics, Advertising"].groupby(['company_permalink','name'])[['raised_amount_usd']].
                sum().reset_index().sort_values(by = 'raised_amount_usd', ascending = False).iloc[[0]])
#Creating Dataframe D3
D3 = master_frame[(master_frame.country_code == "IND") & (master_frame.funding_round_type == "venture") &
                                                     (master_frame.raised_amount_usd >= 5000000) & 
                                                     (master_frame.raised_amount_usd <= 15000000)].reset_index(drop = True)

D3['count_of_investments'] = D3['main_sector'].groupby(D3['main_sector']).transform('count')
D3['total_amt_invested'] = D3['raised_amount_usd'].groupby(D3['main_sector']).transform('sum')
D3.head(3)
#Total number of Investments in Country3
print(D3.company_permalink.count())

#Total amount of investmemts in Country3
print(D3.raised_amount_usd.sum(), end='\n\n********************************************\n\n')

#Top Sector name (no. of investment-wise) for Country3
print(D3.groupby('main_sector')['company_permalink'].count().reset_index().sort_values(by= 'company_permalink', ascending=False))

print(end = '\n\n*******************************************************************************************\n\n')

#For point 3 (top sector count-wise), which company received the highest investment?
print(D3[D3.main_sector == "Others"].groupby(['company_permalink','name'])[['raised_amount_usd']].
                     sum().reset_index().sort_values(by = 'raised_amount_usd', ascending = False).iloc[[0]])

print(end = '\n\n********************************************************************************************\n\n')

#For point 4 (second best sector count-wise), which company received the highest investment?
print(D3[D3.main_sector == "Social, Finance, Analytics, Advertising"].groupby(['company_permalink','name'])[['raised_amount_usd']].
                sum().reset_index().sort_values(by = 'raised_amount_usd', ascending = False).iloc[[0]])
