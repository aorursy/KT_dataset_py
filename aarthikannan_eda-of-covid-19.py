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
# Reading Train and Test Datasets

traindata = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

testdata = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')
#To analyse the top five rows

traindata.head()
#To analyse the top five rows

testdata.head()
#To find the count shape of the data

traindata.shape
#To find the bottom five rows

traindata.tail()
#To find the number of missing values

traindata.isnull().sum()
traindata[~traindata['Province_State'].isnull()]['Country_Region'].value_counts()
#Deleting the Id, since there is no use of it

df=traindata.drop(['Id'], axis=1)

df
#Finding the importing insights



print ("How many Province on train set ==> " +str(len(df["Province_State"].unique())))

print ("How many country on train set ==> " +str(len(df["Country_Region"].unique())))

print ("Date period for train set ==> " +df["Date"].unique()[0]+" to "+df["Date"].unique()[-1])
#Finding the different different types countries in the data

df['Country_Region'].value_counts()

#Finding the highly affected areas



df1=df.loc[(df['ConfirmedCases']>=1000) & (df['Date']>="2020-03-01")].sort_values('Date')
#printing the highly affected areas



df1
#Finding the importing insights

import datetime

print ("How many Province on train set ==> " +str(len(df1["Province_State"].unique())))

print ("How many country on train set ==> " +str(len(df1["Country_Region"].unique())))

print ("Date period for train set ==> " +df1["Date"].unique()[0]+" to "+df1["Date"].unique()[-1])
#Find whether is there any null values for States in India country



traindata[traindata['Country_Region'] == 'India']['Province_State'].isnull().sum()
#Find whether is there any null values for States in Dubai country



traindata[traindata['Country_Region'] == 'Dubai']['Province_State'].isnull().sum()
#Finding the name of the Province in Dubai



traindata[traindata['Country_Region'] == 'Dubai']['Province_State'].unique()
#Find whether is there any null values for States in Bahrain country



traindata[traindata['Country_Region'] == 'Bahrain']['Province_State'].isnull().sum()
#Find whether is there any null values for States in US country



traindata[traindata['Country_Region'] == 'US']['Province_State'].isnull().sum()
#Finding the name of the Province in US



traindata[traindata['Country_Region'] == 'US']['Province_State'].unique()
cases=traindata.groupby(by='Country_Region')[['ConfirmedCases','Fatalities']].sum()

cases
#Finding the maximum number of Confirmed cases



cases["ConfirmedCases"].max()
#Finding which country hold this highest number



cases.loc[cases["ConfirmedCases"] == 3776300]
#Finding the maximum number of Deaths cases



cases["Fatalities"].max()
#Finding which country hold this highest number



cases.loc[cases["Fatalities"] == 129278.0]
#Finding the Top 10 Countries having highest deaths cases



cases.nlargest(10, 'Fatalities')['Fatalities']
#Finding the Top 10 Countries having highest deaths cases



cases.nlargest(10, 'ConfirmedCases')['ConfirmedCases']