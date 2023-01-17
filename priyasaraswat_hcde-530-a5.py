# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
#reading the .csv files for respective years

data2019 = pd.read_csv("/kaggle/input/h1b_2019.csv")

data2018 = pd.read_csv("/kaggle/input/h1b_2018.csv")

data2017 = pd.read_csv("/kaggle/input/h1b_2017.csv")

data2016 = pd.read_csv("/kaggle/input/h1b_2016.csv")

data2015 = pd.read_csv("/kaggle/input/h1b_2015.csv")
#combining all files to form one data frame

df = pd.concat([data2019, data2018, data2017,data2016, data2015])
#Dropping columns that I won't be using

df = df.drop(['Employer', 'Tax ID', 'State', 'City', 'ZIP', '2'], axis=1)
#Renaming column NAICS to Industry code for comprehension

df = df.rename(columns = {'NAICS':'Industry Code'})
#Renaming column NAICS to Industry code for

df = df.dropna()
#Finding the data types

df.dtypes
#Converting the data types to required format

df['Fiscal Year'] = df['Fiscal Year'].astype('str')

df['Industry Code'] = df['Industry Code'].astype('object')

df.dtypes
#Displaying the first five data rows

df.head()



#Displaying the last five data rows

df.tail()



#Displaying a random sample of 10 rows

df.sample(10)
#Setting 'Fiscal Year', 'Industry Code' as index to get descriptive statistic value for integer columns

df_descriptive = df.set_index(['Fiscal Year', 'Industry Code'])
# Printing descriptive statistical values

df_descriptive.describe()
#Finding all approvals for new applications

New_approval = np.array(df['Initial Approvals'].groupby(df['Fiscal Year']).sum())



#Finding all denials for new applications

New_denial = np.array(df['Initial Denials'].groupby(df['Fiscal Year']).sum())



#Extracting years in array format from the dataframe

Years = np.sort(df['Fiscal Year'].unique())
#Plotting total approvals each year for new applications from 2015 to 2019 in green color 

plt.plot(Years, New_approval, color='g')



#Plotting total denials each year for new applications from 2015 to 2019 in orange color 

plt.plot(Years, New_denial, color='orange')



#Defining display label for x-axis

plt.xlabel('Years')

#Defining display label for y-axis

plt.ylabel('Total applications')

#Defining title for the plot

plt.title('Approval and denial of new applications from 2015 - 2019')



plt.show()
#Finding all approvals for continuing applications

Cont_approval = np.array(df['Continuing Approvals'].groupby(df['Fiscal Year']).sum())



#Finding all denials for continuing applications

Cont_denial = np.array(df['Continuing Denials'].groupby(df['Fiscal Year']).sum())
#Plotting total approvals each year for continuing applications from 2015 to 2019 in red color 

plt.plot(Years, Cont_approval, color= 'red')



#Plotting total approvals each year for continuing applications from 2015 to 2019 in blue color 

plt.plot(Years, Cont_denial, color='blue')



plt.xlabel('Years')

plt.ylabel('Total applications')

plt.title('Approval and denial of continuing applications from 2015 - 2019')



plt.show()
# Created a groupby variable that groups initial and continuing approvals by year

groupby_init = df['Initial Approvals'].groupby(df['Fiscal Year'])

groupby_conti = df['Continuing Approvals'].groupby(df['Fiscal Year'])



# Added the two values to get total approvals each year

Total_approval = groupby_init.sum() + groupby_conti.sum()
#Converted Total_approval to a dataframe

Total_approval = pd.DataFrame(Total_approval)



#Extracted year values from the data frame

df_years = df['Fiscal Year'].unique()



#Sorted the years in ascending order

df_years = np.sort(df_years)



#Added years as a new column to new dataframe

Total_approval['Years'] = df_years



#Assigned a column name to default column

Total_approval = Total_approval.rename(columns = {0:'Total approvals'})



#Setting the index to years

Total_approval = Total_approval.set_index('Years')



#And here is the new dataframe

Total_approval
#Plotting year over year approvals from 2015 to 2019

Total_approval.plot(kind = 'bar')
#Grouping the denial numbers and summing them together

groupby_init = df['Initial Denials'].groupby(df['Fiscal Year'])

groupby_conti = df['Continuing Denials'].groupby(df['Fiscal Year'])

Total_denials = groupby_init.sum() + groupby_conti.sum()
#Creating a small dataframe to show total denials each year 

Total_denials = pd.DataFrame(Total_denials)

Total_denials['Years'] = df_years

Total_denials = Total_denials.rename(columns = {0:'Total denials'})

Total_denials = Total_denials.set_index('Years')

Total_denials
#Plotting the total denials each year

Total_denials.plot(kind = 'bar')
# finding the count of unique values of industry codes in dataframe

df_count = df['Industry Code'].value_counts()



# converting the count variable to dataframe

df_industry = pd.DataFrame(df_count)

df_industry = df_industry.rename(columns = {'Industry Code':'Total number of companies'})
# Declaring a list of industries based on industry codes 

industry = ['Professional, Scientific, and Technical Services', 

            'Health Care and Social Assistance', 

            'Manufacturing', 

            'Finance and Insurance','Educational Services',

            'Information', 

            'Wholesale Trade', 

            'Manufacturing', 

            'Industry is unknown', 

            'Retail Trade', 'Construction',

            'Administrative and Support and Waste Management and Remediation Services',

            'Manufacturing','Other Services (except Public Administration)', 

            'Real Estate Rental and Leasing','Accommodation and Food Services',

            'Transportation and Warehousing', 

            'Retail Trade', 

            'Public Administration', 

            'Arts, Entertainment, and Recreation', 

            'Mining', 

            'Utilities',

           'Management of Companies and Enterprises', 

            'Agriculture, Forestry, Fishing and Hunting', 

            'Transportation and Warehousing'] 



#Adding the name of industries to the new dataframe

df_industry['Industry'] = industry 



#Displaying the new dataframe

df_industry
df_industry = df_industry.sort_values(by='Total number of companies',ascending=True)

df_industry.plot.barh(x='Industry', y='Total number of companies', title = 'Companies associated with specific industry filing H1b application from 2015 - 2019')
