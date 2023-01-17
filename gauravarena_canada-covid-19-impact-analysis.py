# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#reading and viewing the GDP by NAICS classified industries data set

gdp=pd.read_csv("../input/GDP by industry.csv")

gdp.head()
#This displays general information about the dataset with informations like the column names their data types 

#and the count of non-null values for every column.

gdp.info()
#displays the columns present in the dataset

gdp.columns
#reading and viewing the CPI by product and product groups data set

cpi=pd.read_csv('../input/CPI_monthly.csv')

cpi.head()
#This displays general information about the dataset with informations like the column names their data types 

#and the count of non-null values for every column.

cpi.info()
#reading and viewing the workhours by NAICS classified industries dataset

workhours=pd.read_csv('../input/Workhours_by_industry.csv')

workhours.head()
#This displays general information about the dataset with informations like the column names their data types 

#and the count of non-null values for every column.

workhours.info()
#taking transpose of the gdp dataframe for easier analysis

gdp_t=gdp.transpose()

gdp_t.columns=gdp_t.iloc[0]

gdp_t.drop('North American Industry Classification System (NAICS)',inplace=True,axis=0)

gdp_t.head()
#taking transpose of the cpi dataframe for easier analysis

cpi_t=cpi.transpose()

cpi_t.columns=cpi_t.iloc[0]

cpi_t.drop(['Products and product groups3 4'],axis=0,inplace=True)

cpi_t.dropna(inplace=True,axis=1)

cpi_t.head()
#taking transpose of the workhours dataframe for easier analysis

workhours_t=workhours.transpose()

workhours_t.columns=workhours_t.iloc[0]

workhours_t.drop('North American Industry Classification System (NAICS)5',inplace=True,axis=0)

workhours_t.head()
#checking for null values

gdp_t.isnull().sum()
#checking for null values

cpi_t.isna().sum()
#checking for null values

workhours_t.isnull().sum()
#changing the datatypes from object to float 

for x in gdp_t.iloc[:,:]:

    gdp_t[x]=gdp_t[x].apply(lambda y: float(y.replace(',','')))

    

gdp_t.info()
#changing the datatypes from object to float 

for x in workhours_t.iloc[:,:]:

    workhours_t[x]=workhours_t[x].apply(lambda y: float(y.replace(',','')))

    

workhours_t.info()
#deriving a new feature which will represent the percentage of increase/decrease in GDP 

x=[]

for i in gdp_t.iloc[:,0:]:

    x.append(((gdp_t.loc['Mar-20',i]-gdp_t.loc['Nov-19',i])/gdp_t.loc['Nov-19',i])*100)

    

print(x)
#appending the newly created feature to the gdp_t dataframe

x=pd.Series(x,name='Increased/Decreased Percentage',index=gdp_t.columns)

gdp_t=gdp_t.append(x,ignore_index=False)



gdp_t.tail()
#deriving a new feature which will represent the percentage of increase/decrease in CPI 

x=[]

for i in cpi_t.iloc[:,0:]:

    x.append(((cpi_t.loc['Apr-20',i]-cpi_t.loc['Dec-19',i])/cpi_t.loc['Dec-19',i])*100)

    

print(x)
#appending the newly created feature to the dataframe

x=pd.Series(x,name='Increased/Decreased Percentage',index=cpi_t.columns)

cpi_t=cpi_t.append(x,ignore_index=False)



cpi_t.tail()
#deriving a new feature which will represent the percentage of increase/decrease in CPI 

x=[]

for i in workhours_t.iloc[:,0:]:

    x.append(((workhours_t.loc['May-20',i]-workhours_t.loc['Jan-20',i])/workhours_t.loc['Jan-20',i])*100)

    



print(x)  
#appending the newly created feature in the dataframe

x=pd.Series(x,name='Increased/Decreased Percentage',index=workhours_t.columns )

workhours_t=workhours_t.append(x,ignore_index=False)

workhours_t.tail()
plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

plt.bar(gdp_t.columns,height=gdp_t.loc['Nov-19'])

plt.xticks(rotation=90)

plt.xlabel('Industry')

plt.ylabel('GDP')

plt.title(' GDP by industry in Nov-2019')



plt.subplot(1,2,2)

plt.bar(gdp_t.columns,height=gdp_t.loc['Mar-20'])

plt.xticks(rotation=90)

plt.xlabel('Industry')

plt.ylabel('GDP')

plt.title(' GDP by industry in Mar-2020')

plt.show()
#plotting the decrease in GDP by industry

plt.figure(figsize=(15,10))

plt.bar(x=gdp_t.columns,height=-gdp_t.loc['Increased/Decreased Percentage'])

plt.xticks(rotation=90)

plt.xlabel('Industry')

plt.ylabel('% Decrease in GDP')

plt.title(' Percentage decrease in GDP by industry ')

plt.show()
#plotting some of the industry and their GDP trend

plt.figure(figsize=(15,10))

plt.plot(gdp_t.iloc[:-1,30:])

plt.xlabel('Months')

plt.ylabel('GDP')

plt.title('GDP trends for Nov-19 to Mar-20 industry wise')

plt.legend(gdp_t.iloc[:-1,28:].columns,loc='lower right')

plt.show()
plt.figure(figsize=(15,10))



plt.subplot(2,2,1)

plt.plot(gdp_t['Arts, entertainment and recreation  [71]'].loc[['Nov-19','Dec-19','Jan-20','Feb-20','Mar-20']])

plt.title('GDP trend for Arts, entertainment and recreation')

plt.xlabel('Months')

plt.ylabel('GDP')



plt.subplot(2,2,2)

plt.plot(gdp_t['Accommodation and food services  [72]'].loc[['Nov-19','Dec-19','Jan-20','Feb-20','Mar-20']])

plt.title('GDP trend for Accommodation and food services')

plt.xlabel('Months')

plt.ylabel('GDP')



plt.subplot(2,2,3)

plt.plot(gdp_t['Non-durable manufacturing industries  [T011]4'].loc[['Nov-19','Dec-19','Jan-20','Feb-20','Mar-20']])

plt.title('GDP trend for Non-durable manufacturing industries')

plt.xlabel('Months')

plt.ylabel('GDP')



plt.subplot(2,2,4)

plt.plot(gdp_t['Real estate and rental and leasing  [53]'].loc[['Nov-19','Dec-19','Jan-20','Feb-20','Mar-20']])

plt.title('GDP trend for Real estate and rental and leasing')

plt.xlabel('Months')

plt.ylabel('GDP')



plt.tight_layout()

plt.show()
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

plt.bar(cpi_t.columns,height=cpi_t.loc['Dec-19'])

plt.xticks(rotation=90)

plt.xlabel('Product/Product Group')

plt.ylabel('CPI')

plt.title(' CPI by product for Dec-2019 ')





plt.subplot(1,2,2)

plt.bar(cpi_t.columns,height=cpi_t.loc['Apr-20'])

plt.xticks(rotation=90)

plt.xlabel('Product/Product Group')

plt.ylabel('CPI')

plt.title(' CPI by product for Apr-2020 ')

plt.show()
#plotting the change in CPY by product/product group

plt.figure(figsize=(15,10))

plt.bar(x=cpi_t.columns,height=cpi_t.loc['Increased/Decreased Percentage'])

plt.xticks(rotation=90)

plt.xlabel('Industry')

plt.ylabel('% Change in CPI')

plt.title(' Percentage Change in CPI by product/product groups ')

plt.show()
plt.figure(figsize=(15,10))

plt.plot(cpi_t.iloc[:-1,2:])

plt.xlabel('Months')

plt.ylabel('CPI')

plt.legend(cpi_t.iloc[:,2:].columns,loc='lower left')

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

plt.plot(cpi_t['Gasoline'].loc[['Dec-19','Jan-20','Feb-20','Mar-20','Apr-20']])

plt.xlabel('Months')

plt.ylabel('CPI')

plt.title('CPI trend for product : Gasoline')



plt.subplot(2,2,2)

plt.plot(cpi_t['Energy 7'].loc[['Dec-19','Jan-20','Feb-20','Mar-20','Apr-20']])

plt.xlabel('Months')

plt.ylabel('CPI')

plt.title('CPI trend for product : Energy')



plt.subplot(2,2,3)

plt.plot(cpi_t['Food 5'].loc[['Dec-19','Jan-20','Feb-20','Mar-20','Apr-20']])

plt.xlabel('Months')

plt.ylabel('CPI')

plt.title('CPI trend for product : Food')



plt.subplot(2,2,4)

plt.plot(cpi_t['Recreation, education and reading'].loc[['Dec-19','Jan-20','Feb-20','Mar-20','Apr-20']])

plt.xlabel('Months')

plt.ylabel('CPI')

plt.title('CPI trend for product : Recreation, education and reading')



plt.tight_layout()
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

plt.bar(workhours_t.columns,height=workhours_t.loc['Jan-20'])

plt.xticks(rotation=90)

plt.xlabel('Industry')

plt.ylabel('Workhours')

plt.title(' Workhours by industry for Jan-2020 ')





plt.subplot(1,2,2)

plt.bar(workhours_t.columns,height=workhours_t.loc['May-20'])

plt.xticks(rotation=90)

plt.xlabel('Industry')

plt.ylabel('Workhours')

plt.title(' Workhours by industry for May-2020 ')

plt.show()

plt.tight_layout()
plt.figure(figsize=(15,10))

plt.bar(x=workhours_t.columns,height=-workhours_t.loc['Increased/Decreased Percentage'])

plt.xticks(rotation=90)

plt.xlabel('Industry')

plt.ylabel('% Decrease in Workhours')

plt.title('Percentage decrease in workhours industry wise')

plt.show()
plt.figure(figsize=(15,10))

plt.plot(workhours_t.iloc[:-1,1:19])

plt.xlabel('Months')

plt.ylabel('Hours')

plt.title('Workhours trend for Jan-20 to May-20 industry wise')

plt.legend(workhours_t.iloc[:,1:].columns,loc='lower right')

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

plt.plot(workhours_t['Professional, scientific and technical services'].loc[['Jan-20','Feb-20','Mar-20','Apr-20','May-20']])

plt.xlabel('Months')

plt.ylabel('Hours')

plt.title('Workhours trend for Professional, scientific and technical services industry')



plt.subplot(2,2,2)

plt.plot(workhours_t['Services-producing sector 11'].loc[['Jan-20','Feb-20','Mar-20','Apr-20','May-20']])

plt.xlabel('Months')

plt.ylabel('Hours')

plt.title('Workhours trend for Services-producing sector')



plt.subplot(2,2,3)

plt.plot(workhours_t['Accommodation and food services'].loc[['Jan-20','Feb-20','Mar-20','Apr-20','May-20']])

plt.xlabel('Months')

plt.ylabel('Hours')

plt.title('Workhours trend for Accommodation and food services')



plt.subplot(2,2,4)

plt.plot(workhours_t['Public administration'].loc[['Jan-20','Feb-20','Mar-20','Apr-20','May-20']])

plt.xlabel('Months')

plt.ylabel('Hours')

plt.title('Workhours trend for Public administration')



plt.tight_layout()