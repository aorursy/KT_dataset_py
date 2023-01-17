import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings('ignore')
#Importing dataset

df_churn_raw = pd.read_csv('../input/bigml_59c28831336c6604c800002a.csv')
#Display dimension of the data

df_churn_raw.shape
#Display first 5 rows of the dataset

df_churn_raw.head()
#Display last 5 rows of the dataset

df_churn_raw.tail()
#Display randomly selected 5 rows of the dataset

df_churn_raw.sample(5)
#Display list of column names of the dataset

df_churn_raw.columns
#Display datatype, non null entries count for each column

df_churn_raw.info()
# area code should not be an int. Lets change the data type to object

df_churn_raw['area code'] = df_churn_raw['area code'].astype('object')
#lets display the datatypes again

df_churn_raw.dtypes
#Get basic stats for numeric data

df_churn_raw.describe()
#Object features are not present in the describe stat. lets include them

df_churn_raw.describe(include=['object','bool'])
#Check for null values in each column

df_churn_raw.isnull().any()
#Sum null values

df_churn_raw.isnull().sum()
#Lets see unique values in 'state' column

df_churn_raw.state.unique()
#Display number of unique values in a column

df_churn_raw['area code'].nunique()
#Display the count of unique values for churn column

df_churn_raw['churn'].value_counts()
# Relative frequency of the values in Churn column. 85.5% have not churned. 14.5% churned

df_churn_raw['churn'].value_counts(normalize =True)
df_churn_raw.head()
#Selection rows in position 0 to 10 and columns 4 and 6

df_churn_raw.iloc[0:10,4:6]
#Selecting last 4 rows

df_churn_raw.iloc[-5:-1,3:6]
#Selecting rows with label from 0 to 10 and column name 'total day charge'

df_churn_raw.loc[0:10,'total day charge']
#Selecting rows labeled from 0 to 10 and 2 columns named 'total day charge' and 'total day minutes'

df_churn_raw.loc[0:10,['total day charge','total day minutes']]
#Selecting first 20 rows and columns specified

df_churn_raw.loc[:20,['total day charge','total day minutes']]
#Selecting 2 columns and displaying as a dataframe. Limiting the entries displayed to 10.

df_churn_raw[['total day charge','total day minutes']].head(10)
#Lets select customers who made day calls below average and day charge was above average. Valuable customers for us.



fil = (df_churn_raw['total day calls']<df_churn_raw['total day calls'].mean()) & (df_churn_raw['total day charge']> df_churn_raw['total day charge'].mean())

df_churn_raw.loc[fil].head()
#Copy the valuable customers data to another dataset

df_temp = df_churn_raw[fil]
df_temp.shape
df_temp.head()
df_temp['total minutes'] = df_temp['total day minutes'] + df_temp['total eve minutes'] + df_temp['total night minutes']

df_temp['total charge'] = df_temp['total day charge'] + df_temp['total eve charge'] + df_temp['total night charge']

df_temp['total calls'] = df_temp['total day calls'] + df_temp['total eve calls'] + df_temp['total night calls']
#If I want to change the minute to hour, map gives the easies way to do so. 

#map works on series and hence it can be used to manipulate one column at a time

df_temp['total minutes'].map(lambda x: x/60).head(10)
#Let us write a small funtion to change minutes top hours

def change_mins_hrs(x):

    return x/60
#Series manipulation can be done using function as well:

df_temp['total minutes'].map(change_mins_hrs).head(10)
df_temp.head()
#I do not like the 'yes' and 'no' values in International plan and voice mail plan. Let us change them to True and False.

#Defining a dictionary that maps yes with True and no with False.

dic = {'yes':'True','no':'False'}
#Using map to replace yes with True and no with False in 'International plan' column

df_temp['international plan'] = df_temp['international plan'].map(dic)
#Using map to replace yes with True and no with False in 'Voice plan' column

df_temp['voice mail plan'] = df_temp['voice mail plan'].map(dic)
df_temp.head()
#If we want to manipulate data in 2 or more columns, applymap is our guy. Applying function on 3 columns at the same time:

df_temp[['total day minutes','total eve minutes','total night minutes']].applymap(change_mins_hrs).head(10)
#Now using 'apply'. 'apply' has same qualities as 'applymap' and few additional powers.

#When we need to apply an operation column-wise or row-wise on a dataframe, 'apply' comes handy.

#Mean of 3 columns(column-wise):

df_temp[['total day minutes','total eve minutes','total night minutes']].apply(np.mean,axis=0)
#Mean of 3 columns(row-wise) for each observation:

df_temp[['total day minutes','total eve minutes','total night minutes']].apply(np.mean,axis=1).head(10)
df_churn_raw.head()
#By area code, we want to see sum of total charge for day, evening and night:

df_churn_raw.groupby('area code').aggregate({'total day charge':'sum','total eve charge':'sum',

                                            'total night charge':'sum'})
#How many subscribe to international plan by area code?

pd.crosstab(df_churn_raw['area code'],df_churn_raw['international plan'],margins=True)
#How many subscribe to international plan by area code (relative frequency)?

pd.crosstab(df_churn_raw['area code'],df_churn_raw['international plan'],margins=True,normalize = True)
#What is the average total day charge by each area code for international plan subscribers and non-subscribers?

pd.crosstab(df_churn_raw['area code'],df_churn_raw['international plan'],

            values=df_churn_raw['total day charge'],aggfunc=np.mean)
#What is the average total night charge by each area code for international plan subscribers and non-subscribers?

pd.crosstab(df_churn_raw['area code'],df_churn_raw['international plan'],

            values=df_churn_raw['total night charge'],aggfunc=np.mean)
#Number of customer service calls and churn relationship in each area?

pd.crosstab(df_churn_raw['area code'],df_churn_raw['churn'],values=df_churn_raw['customer service calls'],aggfunc=np.mean)
#How many customer service calls were placed by customers who did and did not churned?

table = pd.pivot_table(df_churn_raw,index=['churn','international plan'],

                    values = 'customer service calls',aggfunc = np.mean)
table

#Is there something to do with customer service?