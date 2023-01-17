# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



'''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''

# Any results you write to the current directory are saved as output.
from scipy import stats

import seaborn as sns

import statsmodels.api as sm

import calendar

import datetime as dt

df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

df.head(20)



df.head(-20)
#show what columns that the dataset has:

df.columns 
##group on basis of 2 qualitative columns

byJournal = df.groupby(['source_x','has_full_text']).count()

byJournal.head(10)
len(df.date1)
def SplitDateTime(df, dateCol):

    df = df

    df['date'] = pd.to_datetime(df[dateCol]) 

    # print (df['date'].dtype)

    # Datetime64[ns]

    # For extracting year,month and day to new column,follow the code:

    df['year'] = df['date'].dt.year

    df['month'] = df['date'].dt.month

    df['day'] = df['date'].dt.day
#create new date cols for two different date format from the publish_time

df['date1'] = pd.to_datetime(df['publish_time'], errors='coerce', format='%Y-%m-%d')

df['date2'] = pd.to_datetime(df['publish_time'], errors='coerce', format='%Y %b %d')



df['pub_date'] = df['date1']



# replace all missing value of date1 with date2 and assign to new col pub_date

df.loc[df.date1.isna() == True, 'pub_date'] = df.date2

# drop date1 and date2

# df.drop(['date1'],axis=1, inplace=True)

# df.drop(['date2'],axis=1, inplace=True)

df.drop(['date1', 'date2'], axis = 1, inplace=True) 

df['pub_date']

#print (df['publish_time'].dtype)

SplitDateTime(df,'pub_date')

df.columns
cols = ["year", "month", "day"]



def ToInt(df,col):

    df[col] = df[col].astype('Int64')

# Convert cols to integer

for x in cols:

    ToInt(df,x)

    

df[['sha','source_x','pub_date','year','month','day']]
df1=df.dropna(subset=['year'])
print(df1.head(6))

print("-------------------------")

print (df.count, df1.count)