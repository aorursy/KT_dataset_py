# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for data visulaisation

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#pd.read_csv method is to read the text file, 

#encoding ISO-8859-1 is the encoding style used in the text file

#Since its a text file, we have used sep as '\t'

companies = pd.read_csv('../input/companies.txt', encoding = "ISO-8859-1", sep = '\t')

#To get the first 5 observation of the dataset,it is a part of data understanding

companies.head()
#CHECKPOINT 1 ANALYSIS STARTS

# This will get your some insight about the data

# We have 66368 X 10 matrix

# We have all the rows containing object elememts, 

#so if we need to perform operation on some column we have to convert column to desired dtype 

companies.info()
#Checkpoint 1 Table 1.1 Question 3 solution, since we have only permalink column having all the rows with unique values

#This data understanding tells us below observation

#We have 27296 category of companies which receives funding

#We have 137 country and out of which we have 311 states who receives funding

companies.describe()
#This line is to read the round2 csv into dataframe.

rounds2 = pd.read_csv('../input/rounds2.csv', encoding='ISO-8859-1')
#Data understanding steps include:

#Get the insight using head() function

# Get the idea of spread of data using describe()

# Get the info about the dataframe using info()

rounds2.head()
#Checkpoint 1 Table 1.1 Question 1 solution

#This below query gives out the uniqueness of the data

print(len(rounds2.company_permalink.str.upper().unique()))
#Checkpoint 1 Table 1.1 Question 2 solution

#This will give us the unique companies in companies.txt file

print(len(companies['permalink'].str.upper().unique()))
#Checkpoint 1 Table 1.1 Question 4 solution

#Now to calculate companies which are present in rounds2 those are not present in companies

#we need to get the set difference 

unique_company_companies = pd.DataFrame(companies.permalink.str.upper().unique())

unique_company_rounds2 = pd.DataFrame(rounds2.company_permalink.str.upper().unique())

unique_company_companies.equals(unique_company_rounds2)
#Since the above analysis says that the number of unique companies are not same in two files,

#Lets see the names of the companies those are in rounds2 but not in companies

#This names confirms that rounds df has countries which are not there in companies df

set(rounds2.company_permalink.str.upper().unique()).difference(set(companies.permalink.str.upper().unique()))
#Before that lets analyse the two dataframes for null values and column importance

companies.isnull().any()
#Calculating the number of missing values in each row

companies.isnull().sum()
#Since we have significant numbers of null values in few columns, lets analyse the percentage

print(round(100 *(companies.isnull().sum()/len(companies)) , 2))
#Since we have significant percentage of data missing in founded_at, state_code,city,region we can drop these columns

#Dropping of column is justified as the columns having maximum missing values are of no business relevance too.

companies.drop(['founded_at', 'city', 'state_code', 'region'], axis=1, inplace=True)
#Now performing the same data cleansing for rounds df

print(round(100 * (rounds2.isnull().sum()/len(rounds2)) ,2))
#Since we have huge missing values for funding_round_code and it has no business relevance, hence dropping

rounds2.drop('funding_round_code', axis=1 , inplace=True)
#Since rounds2 df has company_permalink and companies df has permalink as unique column names, 

#we have to merge them on permalink

#Also making the column permalink in UPPER/LOWER case will help in clean merging

rounds2['company_permalink'] = list(map(lambda x : x.upper() , rounds2['company_permalink']))

companies['permalink'] = list(map(lambda x : x.upper(), companies['permalink']))
#Before merging lets rename the company_permalink column of rounds2 df to permalink

rounds2.rename(columns={'company_permalink':'permalink'}, inplace=True)
#Now we can merge the dataframes

master_frame = pd.merge(rounds2, companies, how = 'left', on ='permalink')

master_frame.head()
#CHECKPOINT 2 ANALYSIS STARTS

#This gives us the overview of the observation in master_frame

master_frame.info()
#As per Table2.1 we need to only consider 4 founding_round_type in calculation

master_frame = master_frame[(master_frame['funding_round_type'] == 'venture') 

                            | (master_frame['funding_round_type'] == 'seed')

                            | (master_frame['funding_round_type'] == 'angel')

                            | (master_frame['funding_round_type'] == 'private_equity')]

master_frame.head()
#Converting the raised_amount_usd in millions

master_frame['raised_amount_usd'] = master_frame['raised_amount_usd']/1000000
#Now we will group by master_frame by funding_round_type and calculate the average

print(round(master_frame.groupby(['funding_round_type']).raised_amount_usd.mean(), 2))

round(master_frame.groupby(['funding_round_type']).raised_amount_usd.mean(), 2).plot(kind='bar',x='funding_round_type',y='million_usd')
#As mentioned in the checkpoint we need to only consider the above received 'venture' investment type

master_frame = master_frame[(master_frame['funding_round_type'] == 'venture')]

master_frame.head()
#Analysing the master_frame

master_frame.isnull().sum(axis=0)
#Dropping the unncessary column

master_frame = master_frame.drop(['funding_round_permalink', 'funded_at', 'homepage_url',

                                  'funded_at','status'], axis = 1)
#Dropping rows based on null columns

master_frame = master_frame[~(master_frame['raised_amount_usd'].isnull() | master_frame['country_code'].isnull() |

                             master_frame['category_list'].isnull())]

master_frame.info()
#CHECKPOINT 3 ANALYSIS

#Creating the top9 df

top9 = master_frame.pivot_table(values='raised_amount_usd', index = 'country_code', aggfunc = 'sum').sort_values(['raised_amount_usd'], ascending = False).head(9)
#Getting the list of countries, to filter out the english speaking countries

print(top9)

top9.raised_amount_usd.plot(kind='bar' , x='coutry_code', y='raised_amount_usd')

#We can hence pick up USA(United States of America) ,GBR(Great Britain/UK), IND(India) as top 3 english speaking countries to invest in

#Use the link https://en.wikipedia.org/wiki/List_of_countries_by_English-speaking_population
#Sector Analysis

#Read the mapping CSV

mapping = pd.read_csv('../input/mapping.csv')

mapping.head()
#Reshaping the mapping dataframe to merge with the master_frame dataframe. Using melt() function to unpivot the table.

mapping = pd.melt(mapping, id_vars =['category_list'], value_vars =['Automotive & Sports',

                                                              'Cleantech / Semiconductors','Entertainment',

                                                             'Health','Manufacturing','News, Search and Messaging','Others',

                                                             'Social, Finance, Analytics, Advertising']) 
#Dropping the rows with value 0,

mapping = mapping[~(mapping.value == 0)]
#Also we do not need the value column

mapping.info()
#Treating the category_list of master_frame

master_frame['category_list'] = master_frame['category_list'].apply(lambda x: x.split('|')[0])
#Now we need to merge the master_frame and mapping

master_frame = master_frame.merge(mapping, how = 'left', on ='category_list')

master_frame.info()
#Renaming the column variable as main_sector

master_frame = master_frame.rename(columns={'variable':'main_sector'})
#List of primary sectors which have no main sectors in the master_frame

print(master_frame[master_frame.main_sector.isnull()].category_list.unique())
#Number of rows with NaN masin_sector value

len(master_frame[master_frame.main_sector.isnull()])
#Retaining the rows which have main_sector values

master_frame = master_frame[~(master_frame.main_sector.isnull())]

len(master_frame.index)
#Now creating the three dataframe D1, D2 and D3 for further sector analysis

D1 = master_frame[(master_frame['country_code'] == 'USA') & (master_frame.raised_amount_usd > 5.0) & (master_frame.raised_amount_usd < 15.0)]
#Now we need to calculate the total count of investment and total amount of investment and merge it with master_frame

D1_function = D1[['raised_amount_usd','main_sector']].groupby(['main_sector']).agg(['sum', 'count'])
#Merging D1 and D1_function

D1 = D1.merge(D1_function, on ='main_sector', how='left')

D1.head()
#Ploting the sector-wise graph for investment count and investment amount for USA

D1_function.plot(kind='bar')
#Similarly we can calculate D2

D2 = master_frame[(master_frame['country_code'] == 'GBR') & (master_frame.raised_amount_usd > 5.0) & (master_frame.raised_amount_usd < 15.0)]

D2_function = D2[['raised_amount_usd','main_sector']].groupby(['main_sector']).agg(['sum', 'count'])

D2 = D2.merge(D2_function, on ='main_sector', how = 'left')

D2.head()
#Ploting the sector-wise graph for investment count and investment amount for GBR

D2_function.plot(kind='bar')
#Similarly we can calculate D3

D3 = master_frame[(master_frame['country_code'] == 'IND') & (master_frame.raised_amount_usd > 5.0) & (master_frame.raised_amount_usd < 15.0)]

D3_function = D3[['raised_amount_usd','main_sector']].groupby(['main_sector']).agg(['sum', 'count'])

D3 = D3.merge(D3_function, on ='main_sector', how = 'left')

D3.head()
#Ploting the sector-wise graph for investment count and investment amount for IND

D3_function.plot(kind='bar')
#Now calculating the answers for table 5.1 in excel sheet

print(D1.raised_amount_usd.count())

print(D2.raised_amount_usd.count())

print(D3.raised_amount_usd.count())
#Total amount of investment (USD)

print(round(D1.raised_amount_usd.sum(), 2))

print(round(D2.raised_amount_usd.sum(), 2))

print(round(D3.raised_amount_usd.sum(), 2))
#Maximun number of investment per sector

D1_function.head(8)
D2_function.head(8)
D3_function.head(8)
#For the top sector USA , which company received the highest investment?

company = D1[D1['main_sector']=='Others']

company = company.pivot_table(values = 'raised_amount_usd', index = 'permalink', aggfunc = 'sum')

company = company.sort_values(by = 'raised_amount_usd', ascending = False).head()

print(company.head(1))



#For the second top sector USA , which company received the highest investment?

company = D1[D1['main_sector']=='Social, Finance, Analytics, Advertising']

company = company.pivot_table(values = 'raised_amount_usd', index = 'permalink', aggfunc = 'sum')

company = company.sort_values(by = 'raised_amount_usd', ascending = False).head()

print(company.head(1))
#For the top sector GBR , which company received the highest investment?

company = D2[D2['main_sector']=='Others']

company = company.pivot_table(values = 'raised_amount_usd', index = 'permalink', aggfunc = 'sum')

company = company.sort_values(by = 'raised_amount_usd', ascending = False).head()

print(company.head(1))



#For the second top sector USA , which company received the highest investment?

company = D2[D2['main_sector']=='Cleantech / Semiconductors']

company = company.pivot_table(values = 'raised_amount_usd', index = 'permalink', aggfunc = 'sum')

company = company.sort_values(by = 'raised_amount_usd', ascending = False).head()

print(company.head(1))
#For the top sector IND , which company received the highest investment?

company = D3[D3['main_sector']=='Others']

company = company.pivot_table(values = 'raised_amount_usd', index = 'permalink', aggfunc = 'sum')

company = company.sort_values(by = 'raised_amount_usd', ascending = False).head()

print(company.head(1))



#For the second top sector USA , which company received the highest investment?

company = D3[D3['main_sector']=='Social, Finance, Analytics, Advertising']

company = company.pivot_table(values = 'raised_amount_usd', index = 'permalink', aggfunc = 'sum')

company = company.sort_values(by = 'raised_amount_usd', ascending = False).head()

print(company.head(1))