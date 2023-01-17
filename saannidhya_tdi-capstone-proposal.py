# Importing libraries

import numpy as np

import pandas as pd 

import plotly.offline as pyo

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Reading in the dataset

data_science_jobs_df = pd.read_csv("../input/data_science_jobs_df.csv")

df = pd.DataFrame(data_science_jobs_df)

print(df.head())
# Checking the first 5 rows

data_science_jobs_df.head()

# The column names are not very intuitive. Let's change them.

cols = {'Unnamed: 0' : 'ID', '0': 'Title', '1' : 'Company', '2' : 'Company_Address', '3' : 'Salary', '4': 'Summary'}

df = df.rename(columns = cols)

df.head()
# Checking the n.o of obs in the dataset

len(df)

# df.loc[0:100,:]
# Removing observations with no salary information

df_sal = df.loc[~df.loc[:, 'Salary'].isnull(),:]

# Checking length of the new dataset

len(df_sal)

# nobs removed

len(df) - len(df_sal)
# Checking the n.o of duplicates in the dataset based on Title, Company, Company_Address and Salary

df_sal[df_sal.duplicated(['Title', 'Company', 'Company_Address', 'Salary'])]

# len(df_sal[df_sal.duplicated(['Title', 'Company', 'Company_Address', 'Salary'])])

# 3736

# An e.g of duplicate observation -

df_sal.loc[(df_sal['Company'] == 'Qloo') & (df_sal['Title'] == 'Senior Data Scientist') , :]
# Removing duplicates

df_sal_no_dup = df_sal.drop_duplicates(subset = ['Title', 'Company', 'Company_Address', 'Salary'])

len(df_sal_no_dup)
# Extracting city out of company address

pd.options.mode.chained_assignment = None  # default='warn'

df_sal_no_dup.loc[:, 'City'] = df_sal_no_dup['Company_Address'].apply(lambda x: str(x).split(',')[0])
# Extracting year, month or day out of salary

df_sal_no_dup.loc[:, 'Sal_type'] = df_sal_no_dup['Salary'].apply(lambda x: str(x).split(' ')[-1])



# Extracting max of the salary range

df_sal_no_dup.loc[:, 'Sal_max'] = df_sal_no_dup['Salary'].apply(lambda x: str(x).split(' ')[-3])

# Removing dollar sign and converting column into string

df_sal_no_dup.loc[:, 'Sal_max'] = df_sal_no_dup['Salary'].apply(lambda x: str(x).split('$')[1])

df_sal_no_dup.loc[:, 'Sal_max'] = df_sal_no_dup['Sal_max'].apply(lambda x: str(x).split(' ')[0])



# Converting all monthly, hourly, and weekly salaries into yearly

## Yearly

mon_bool = df_sal_no_dup['Sal_type'] == 'year'

df_sal_no_dup.loc[(mon_bool), 'Annual_Max_Salary'] = df_sal_no_dup['Sal_max'].apply(lambda x: float(str(x).replace(',','')))

## Monthly

mon_bool = df_sal_no_dup['Sal_type'] == 'month'

df_sal_no_dup.loc[(mon_bool), 'Annual_Max_Salary'] = df_sal_no_dup['Sal_max'].apply(lambda x: float(str(x).replace(',','')) * 12)

## Hourly (assuming 40 work hours and 52 weeks)

hour_bool = df_sal_no_dup['Sal_type'] == 'hour'

df_sal_no_dup.loc[(hour_bool), 'Annual_Max_Salary'] = df_sal_no_dup['Sal_max'].apply(lambda x: float(str(x).replace(',','')) * 40 * 52)

## Weekly

week_bool = df_sal_no_dup['Sal_type'] == 'week'

df_sal_no_dup.loc[(week_bool), 'Annual_Max_Salary'] = df_sal_no_dup['Sal_max'].apply(lambda x: float(str(x).replace(',','')) * 52)

## Removing class Salary type and resetting index

df_sal_no_dup = df_sal_no_dup.loc[~(df_sal_no_dup['Sal_type'] == 'class'), :].reset_index(drop=True)
# Removing observations with no cities

df_sal_no_dup = df_sal_no_dup[~(df_sal_no_dup['City'] == 'nan')]

# nobs = 961 
# Creating a box plot with n.o of data science job openings in different locations

# plot1 = df_sal_no_dup[['ID','City']].groupby(['City']).agg('count').plot(kind = 'bar', legend = False, title = "Number of Data Science Postings by City")

plot1 = df_sal_no_dup[['ID','City']].groupby(['City']).agg('count').reset_index().rename(columns = {'ID' : 'Count'})

sns.set_style("dark")

sns.barplot(x = plot1['City'], y = plot1['Count'])

plt.xticks(rotation = 90)

plt.title("Number of Data Science Postings by City")

plt.tight_layout()

plt.show()
# Calculating and plotting average salaries offered in different locations (average of the max)

df_sal_no_dup['Annual_Max_Salary'] = df_sal_no_dup['Annual_Max_Salary'].astype(int)

# plot2 = df_sal_no_dup[['City','Annual_Max_Salary']].groupby('City').agg('mean').plot(kind = 'bar', color = 'gray', title = 'On Average, Maximum Salaries offered by different Cities')

plot2 = df_sal_no_dup[['City','Annual_Max_Salary']].groupby('City').agg('mean').reset_index()

sns.set_style("whitegrid")

sns.set()

sns.barplot(x = plot2['City'], y = plot2['Annual_Max_Salary'], color='brown')

plt.xticks(rotation = 90)

plt.ylabel('Salary')

plt.title("On Average, Maximum Salaries offered by different Cities")

plt.tight_layout()

plt.show()
import chart_studio.plotly as py

## Companies with highest n.o of postings

plot3 = df_sal_no_dup[['ID','Company']].groupby(['Company']).agg('count').reset_index().rename(columns = {'ID': 'Count'})

plot3 = plot3.loc[plot3['Count'] >= 10, :]

data_bar = [go.Bar(x = plot3['Company'] , y = plot3['Count'], name = 'company_count_barplot', marker = dict(color = '#109618'), width = 0.5)]

layout = go.Layout(title = 'Companies with 10 or more Data Science job postings', xaxis_tickangle = -90)

fig = go.Figure(data = data_bar, layout = layout)

fig.layout.template = 'plotly_white'

pyo.iplot(fig)
## Out of the 9 companies with highest n.o of job postings, which company is offering the highest salary on average

plot4 = df_sal_no_dup.loc[df_sal_no_dup['Company'].isin(plot3['Company'].unique()), :]

plot4 = plot4[['Company','Annual_Max_Salary']].groupby('Company').agg('mean').reset_index().rename(columns = {'Annual_Max_Salary' : 'Average Salary'})

data_bar2 = [go.Bar(x = plot4['Company'], y = plot4['Average Salary'], name = 'company_mean_sal_barplot', marker = dict(color = '#FF7F0E'), width = 0.5)]

layout = go.Layout(title = 'Average Salary offered by Companies with 10 or more Data Science job postings', xaxis_tickangle = -90)

fig = go.Figure(data = data_bar2, layout = layout)

fig.layout.template = 'plotly_white'

pyo.iplot(fig)
# Checking n.o of unique cities

df_sal_no_dup['City'].unique()



# df_sal_no_dup.loc[df_sal_no_dup['City'] == 'Davidson', :]



# Grouping the cities

# if New York, Brooklyn, Jersey City, Fort Lee then NYC

# if Fort Mill, Huntersville, Davidson, Charlotte then Charlotte

# if San Rafael, Oakland, Walnut Creek, San Francisco then San Francisco

# Boston

# if Burbank, Torrance, Woodland Hills, Cypress then Los Angeles

# if Fort Meade, Hyattsville, Arlington, Greenbelt then Washington



city_mapping = {'New York': 'New York City', 'Brooklyn' : 'New York City', 'Jersey City' : 'New York City', 'Fort Lee' : 'New York City',

                'Fort Mill':'Charlotte','Huntersville':'Charlotte','Davidson':'Charlotte',

                'San Rafael':'San Francisco','Oakland':'San Francisco','Walnut Creek':'San Francisco',

                'Burbank':'Los Angeles','Torrance':'Los Angeles','Woodland Hills':'Los Angeles', 'Cypress':'Los Angeles',

                'Fort Meade':'Washington','Hyattsville':'Washington','Arlington':'Washington','Greenbelt':'Washington'

        }

# print(city_mapping)
# Re-mapping to major cities

df_sal_no_dup['Major_City'] = df_sal_no_dup['City']

df_sal_no_dup = df_sal_no_dup.replace({'Major_City': city_mapping})

# df_sal_no_dup.loc[df_sal_no_dup['Major_City'].isnull(),:]