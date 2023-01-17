# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # seaborn for data viz 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#read dataset1 into dataframe named 'crime'

import pandas as pd

crime= pd.read_csv("../input/mini-project-1/King_County_Sheriff_s_Office_-_Incident_Dataset_filtered_2019.csv")



#let's take a look at the dataset

crime.sort_values('state')

# Drop columns 'case_number', FCR', 'Address_1', 'Incident Block loation', 'KSCO Patrol Districts' and'KCSO Reporting Districts'

crime=crime.drop(columns=['case_number','FCR','created_at','updated_at','address_1','Incident Block Location','KCSO Patrol Districts','KCSO Reporting Districts'])



# Check if there is missing value in dataset

crime.isnull().sum()
# There are missing values in 'zip' column. Let's delete those entries.

crime=crime.dropna()



# Double-check the result

crime.isnull().sum()
#let's take a look at the dataset

crime.sort_values('state')

#Check datatypes in dataset 1

crime.dtypes
#Import dataset3 into dataframe named 'kingzip'

kingzip=pd.read_csv("../input/mini-project-1/King County Zip Code_pop2010.csv")



#Check datatypes in dataset 3

kingzip.dtypes

#'zip' in dataset 3 is currently integer. Let's convert that to string to match dataset 1.

kingzip['zip']=kingzip.zip.astype(str)



#Double-check result

kingzip.dtypes
#inner join

king_crime=pd.merge(crime, kingzip['zip'], on='zip',how='inner')



#View result by zip code in ascending orders 

king_crime.sort_values('zip')
#plot a bar graph according to frequency count in column 'hour_of_day'

king_crime['hour_of_day'].value_counts().plot(kind='barh')



#plot a bar graph according to frequency count in column 'day_of_week'

king_crime['day_of_week'].value_counts().plot(kind='barh')
#plot a bar graph according to frequency count in column 'incident_type'

king_crime['incident_type'].value_counts().plot(kind='barh')

# Use counter to count no. of crime incident entries per zip code in dataset1, then write into a dictionary named 'crimedict'

from collections import Counter

crimedict=Counter(king_crime['zip'])



#Create dataframe named 'df' from dictionary

import pandas as pd

df=pd.DataFrame.from_dict(crimedict, orient='index', columns=['crime count'])



#Renew column name 'index' to 'zip'

df=df.reset_index().rename(columns = {'index':'zip'})



#Preview dataframe in ascending zip code

df=df.sort_values('zip')

df
#Merge df with kingzip on 'zip'

kingzip=pd.merge(kingzip,df,on='zip',how='outer')



#Preview result

kingzip
# Check if there is missing value in dataset

# If there is a missing value, the crime count for that zip code will be zero

kingzip.isnull().sum()
#Fill in missing value with 0

kingzip['crime count'].fillna(0, inplace=True)



#Double-check result

kingzip.isnull().sum()
# Divide crime count by population of that zip code, then create new column

kingzip['crime count per capita']=kingzip['crime count']/kingzip['population']



# View results of the top 10 crime rate zip code 

kingzip.sort_values('crime count per capita').tail(10)

#Referecne back to dataset 1. Plot bar graph of incident types in 98134.

king_crime[king_crime.zip=='98134'].incident_type.value_counts().plot(kind='barh')



#Referecne back to dataset 1. Plot bar graph of incident types in 98288.

king_crime[king_crime.zip=='98288'].incident_type.value_counts().plot(kind='barh')
#Import dataset 2 as AverageAGI

import pandas as pd

AverageAGI = pd.read_csv("../input/mini-project-1/Washington State IRS tax -2017_AverageAGI.csv")



#View result

AverageAGI
#Convert datatype of 'zip' in dataset3 to string to match dataset 1

AverageAGI['zip']=AverageAGI.zip.astype(str)



#Double-check result

AverageAGI.dtypes
#Inner-join dataframe AverageAGI with dataframe kingzip according to zip code 

kingzip=pd.merge(AverageAGI,kingzip,on='zip',how='inner')



#View result

kingzip
# Which zip code has highest crime rate again and how's their average income looking ?

kingzip.sort_values('crime count per capita').tail(10)
# Try Seaborn jointplot (hist+scatterplot) to look at correlation between 'average adjust gross income per tax return' and 'crime count per capita'.

viz0=sns.jointplot(x='average adjust gross income per tax return', y='crime count per capita', data=kingzip, kind='reg')



#Calculate pearsonr coefficient

import scipy.stats as stats

viz0.annotate(stats.pearsonr)
#calculate outlier threshold

lower_bound=0.1

upper_bound=0.95

res=kingzip.quantile([lower_bound, upper_bound])

res
# For 'average agi per tax return', any value <60884.2 or >240385 are outliers. Let me find them.

outliers1 = kingzip[(kingzip['average adjust gross income per tax return'] < 60884.2)].index

outliers2 = kingzip[(kingzip['average adjust gross income per tax return'] >240385)].index



# For 'crime count per capita', any value <60884.2 or >240385 are outliers. Let me find them.

outliers3 = kingzip[(kingzip['crime count per capita'] >0.052894)].index

outliers4 = kingzip[(kingzip['crime count per capita'] <0.000543)].index



# Let's drop the outliers identified

kingzip.drop(outliers1, inplace=True)

kingzip.drop(outliers2, inplace=True)

kingzip.drop(outliers3, inplace=True)

kingzip.drop(outliers4, inplace=True)



# Let's preview the result

kingzip
# Perform Seaborn jointplot (hist+scatterplot) again to look at correlation between 'average adjust gross income per tax return' and 'crime count per capita'.

viz=sns.jointplot(x='average adjust gross income per tax return', y='crime count per capita', data=kingzip, kind='reg', height=8)



#Calculate pearsonr coefficient

import scipy.stats as stats

viz.annotate(stats.pearsonr)

#Let me try a density plot with seaborn to visualize the dataset density



viz2=sns.jointplot(x='average adjust gross income per tax return', y='crime count per capita', data=kingzip, kind='kde',height=8)



#Calculate pearsonr coefficient 

import scipy.stats as stats

viz2.annotate(stats.pearsonr)
