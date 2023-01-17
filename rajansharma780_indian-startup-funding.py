# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#load the csv file and make the data frame

startup_df = pd.read_csv('../input/startup_funding.csv')
#display the data frame

startup_df
#display the how many rows and columns

print("The dataframe has {} rows and {} columns".format(startup_df.shape[0],startup_df.shape[1]))
#display the data types of columns

startup_df.dtypes
#display the informtion of dataframe

startup_df.info()
#display how many null values are there in each column

startup_df.apply(lambda x : sum(x.isnull()))
#display graphically how many null values are there in each column 

sns.heatmap(startup_df.isnull(),cbar=0)

plt.show()
#fill the null values of CityLocation column with value NotSpecific

startup_df['CityLocation'] = startup_df['CityLocation'].fillna(value = 'NotSpecific')
#fill the null values of IndustryVertical column with value Other

startup_df['IndustryVertical'] = startup_df['IndustryVertical'].fillna(value = 'Other')
#display which CityLocation has more frequency or more startups

startup_df['CityLocation'].value_counts()
def convert_to_one_location(x):

    x = x.lower()

    if re.search('/',x):

        return(x.split('/')[0].strip())

    else:

        return(x.strip())

    

startup_df['CityLocation'] = startup_df['CityLocation'].apply(convert_to_one_location)
startup_df['CityLocation'].value_counts()
startup_df['InvestmentType'].value_counts()
def align_investment(x):

    x = str(x).lower()

    x = x.replace(' ','')

    return(x)

startup_df['InvestmentType'] = startup_df['InvestmentType'].apply(align_investment)
startup_df['InvestmentType'].value_counts()
# now copy this data frame and make new data frame and del unnecessary columns

new_startup_df = startup_df.copy()
#display the new data frame

new_startup_df
del new_startup_df['Remarks']

del new_startup_df['SNo']
new_startup_df
#we have to make sure that all date are in proper format(i.e.,DD/MM/YYYY)

def count_date_proper(x):

    return(len(x.split('/')))

new_startup_df[new_startup_df['Date'].apply(count_date_proper) != 3]
new_startup_df['Date'].replace('12/05.2015','12/05/2015',inplace=True)

new_startup_df['Date'].replace('13/04.2015','13/04/2015',inplace=True)

new_startup_df['Date'].replace('15/01.2015','15/01/2015',inplace=True)

new_startup_df['Date'].replace('22/01//2015','22/01/2015',inplace=True)
#now we have to make date to time series

new_startup_df['Date'] = pd.to_datetime(new_startup_df['Date'],format='%d/%m/%Y')
new_startup_df['Date'].dtypes
new_startup_df['AmountInUSD']
def convert_amount(x):

    if(re.search(',',x)):

        return(x.replace(',',''))

    else:

        return(x)

new_startup_df['AmountInUSD'] = new_startup_df[new_startup_df['AmountInUSD'].notnull()]['AmountInUSD'].apply(convert_amount).astype('int')
#now we have to fill na values in AmountInUSD column with mean values

new_startup_df['AmountInUSD'] = new_startup_df['AmountInUSD'].fillna(value = np.mean(new_startup_df['AmountInUSD']))
new_startup_df['AmountInUSD'] = new_startup_df['AmountInUSD'].astype('int')
new_startup_df['InvestmentType'].replace('nan',np.NaN,inplace=True)
new_startup_df['InvestmentType'].value_counts()
#fill the null value of investment type with backward filling method

new_startup_df['InvestmentType'].fillna(method='bfill',inplace=True)
#now we will calculate in each startup how many investors are there

def count_investors(x):

    if(re.search(',',x) and x!='empty'):

        return(len(x.split(',')))

    elif x != 'empty':

        return 1

    else:

        return -1

new_startup_df['NumberOfInvestors'] = new_startup_df['InvestorsName'].replace(np.NaN,'empty').apply(count_investors)
#display which startup has maximum investors

new_startup_df[new_startup_df['NumberOfInvestors'] == new_startup_df['NumberOfInvestors'].max()]
#fill na valus of InvestorsName with None

new_startup_df['InvestorsName'] = new_startup_df['InvestorsName'].fillna(value='None')
#find unique Investors

Investors = []

investor_list = new_startup_df['InvestorsName'].apply(lambda x : x.lower().strip().split(','))
for i in investor_list:

    for j in i:

        if(i!='none' or i!=''):

            Investors.append(j.strip())

unique_investors = list(set(Investors))
new_startup_df.iloc[:,[1,2,3,4,6]] = new_startup_df.iloc[:,[1,2,3,4,6]].applymap(lambda x:x.lower().replace(' ','')if pd.notnull(x) is True else x)
def check(x):

    if(pd.notnull(x)):

        return x.lower()

new_startup_df.iloc[:,3] = new_startup_df.iloc[:,3].apply(check)
#display graphically how many null values are there

sns.heatmap(new_startup_df.isnull(),cbar=0)

plt.show()
#make a list of unique startupname

uniquestartupname = list(new_startup_df['StartupName'].unique())
#make a list of all startupname

startupname = list(new_startup_df['StartupName'])
#remove ambigous records

for i in range(len(uniquestartupname)):

    for j in range(len(startupname)):

        if(re.search(uniquestartupname[i],startupname[j])):

            startupname[j] = uniquestartupname[i]
#StartupName column without ambigous records

new_startup_df['StartupName'] = startupname
#display how many unique startup are there

len(new_startup_df['StartupName'].unique())
#display 5 number summary of AmountInUSD column

new_startup_df['AmountInUSD'].describe().astype('int')
#display top 10 funding startups

new_startup_df.groupby(by='StartupName')['AmountInUSD'].sum().sort_values(ascending =False).head(10)
#display top 10 startups which are having more investors

new_startup_df.groupby(by='StartupName').sum().sort_values(by='NumberOfInvestors',ascending=False).head(10)