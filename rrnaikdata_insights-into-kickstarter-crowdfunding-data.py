import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter

import os

from datetime import datetime

import seaborn as sns
df = pd.read_csv("../input/ks-projects-201801.csv")

df.head()
df.shape
df.describe()
df.dtypes
# drop the nan rows 

df.dropna

df.shape
# drop the columns 'goal','pledged','usd pledged','launched','deadline','ID' 

df_cleaned = df.drop(['goal','pledged','usd pledged','launched','deadline','ID'], axis = 1)

#convert the time to date time format

df_cleaned['launched_date'] = pd.to_datetime(df['launched'], format='%Y-%m-%d %H:%M:%S')

df_cleaned['deadline_date'] = pd.to_datetime(df['deadline'], format='%Y-%m-%d %H:%M:%S')
# calculate the duration of the project in days. This is the difference between the deadline date and the launched date

df_cleaned['duration'] = df_cleaned['deadline_date'] - df_cleaned['launched_date']

df_cleaned['duration'] = df_cleaned['duration'].dt.days
df_cleaned[df_cleaned['state'] == 'successful']['state'].value_counts()
# successful projects / total projects * 100

success_percentage = df_cleaned[df_cleaned['state'] == 'successful']['state'].value_counts() / len(df_cleaned["state"])

print('Sucess Percent = {0:2.2%}' .format(success_percentage[0]))
# percentages of all different states

state_percentage = round(df_cleaned['state'].value_counts() / len(df_cleaned["state"]) * 100,1)

print(state_percentage)
df_test = df_cleaned[(df_cleaned['state'] == 'successful') | (df_cleaned['state'] == 'failed')]

df_test_live = df_cleaned[df_cleaned['state'] == 'live']



print (df_test.shape)

print (df_test_live.shape)

#length of the name of the projects

df_test['length_name'] = df_test.loc[:,'name'].str.len()
# Launch year, month, date

df_test['launch_year'] = pd.DatetimeIndex(df_test.loc[:,'launched_date']).year

df_test['launch_month'] = pd.DatetimeIndex(df_test.loc[:,'launched_date']).month

df_test['launch_date'] = pd.DatetimeIndex(df_test.loc[:,'launched_date']).day

df_test= df_test.sort_values('launched_date',ascending=True)

df_test = df_test.drop(['launched_date','deadline_date'], axis = 1)
df_test.columns
# Generate plots of projects per year, month, date, country, currency

# Generate plots to see the sucess percentages

# Observe any trends



plot_columns = ['launch_year','launch_month','launch_date','country','currency'] 

x_label = ['Launch Year','Launch Month','Launch Date', 'Country','Currency'] 

y_label = ['Number of Projects', 'Sucess Percentage(%)']

fig, axarr = plt.subplots(len(plot_columns), len(y_label), figsize=(20,5*len(plot_columns)))

for i in range(len(plot_columns)):

    a = None

    a = round(100 * df_test[df_test.state == "successful"][plot_columns[i]].value_counts()/df_test[plot_columns[i]].value_counts(), 1)

    # taking the column data/variable for example launch year and count the sucessful number of projects for each year and divide by the total projects in that year. 

    #print (a)

    sns.countplot(df_test[plot_columns[i]],ax=axarr[i][0])

    axarr[i][0].set_xlabel(x_label[i])

    axarr[i][0].set_ylabel(y_label[0])

    sns.barplot(x = a.index,y = a.values,ax=axarr[i][1])

    axarr[i][1].set_xlabel(x_label[i])

    axarr[i][1].set_ylabel(y_label[1])
for i in range(len(plot_columns)):

    a = None

    a = round(100 * df_test[df_test.state == "successful"][plot_columns[i]].value_counts()/df_test[plot_columns[i]].value_counts(), 1)

    # taking the column data/variable for example launch year and count the sucessful number of projects for each year and divide by the total projects in that year. 

    print ( a, '\n')
# percentage of the main categories in the data

round(100* df_test['main_category'].value_counts()/df_test['main_category'].value_counts().sum(),2)
fig, ax = plt.subplots(1,1, figsize=(20, 5))

sns.countplot(df_test['main_category'])

ax.set_xlabel ('Main Category')

ax.set_ylabel ('Number of Projects')
fig, ax = plt.subplots(1,1, figsize=(30, 5))

sns.barplot(df_test['duration'].value_counts().index,df_test['duration'].value_counts().values)

ax.set_xlabel ('Duration')

ax.set_ylabel ('Number of Projects')
fig, ax = plt.subplots(1,1, figsize=(20, 5))

df_test['length_name'].value_counts().plot(kind='bar', figsize=(20,5))

ax.set_xlabel ('Length of the Project Name')

ax.set_ylabel ('Number of Projects')
df_US = df_test[(df_test.country == "US") & df_test.main_category.str.contains('Film & Video')]

df_US['state_val'] = (df_US['state'] == 'successful') * 1

df_US = df_US.drop(['country','currency','main_category','state'], axis = 1)

print(df_US.shape)

df_US.category.value_counts().index[0:10]
list_cat = list(df_US.category.value_counts().index[0:10]) # Top 10 of the categories in the Film & Video 

df_US_sub = None

for j in range(len(list_cat)):

    #print (j)

    if j == 0:

        #print (list_cat[j])

        df_US_sub = df_US[df_US.category.str.contains(list_cat[j])]

        #print (df_US_sub.shape)

    else:

        #print (list_cat[j])

        new_portion = df_US[df_US.category.str.contains(list_cat[j])]

        df_US_sub = pd.concat([df_US_sub,new_portion])

        #print (df_US_sub.shape)

        #print (new_portion.shape)

df_US_sub.shape

# Observing the number of projects in each of the categories

df_US_sub.category.value_counts()
fig, ax = plt.subplots(1,1, figsize=(15, 5))

sns.countplot(df_US_sub['category'])

ax.set_xlabel ('Category')

ax.set_ylabel ('Number of Projects')
# sucess percentage of the Film&Video

print ("Film & Video sucess percentage is : ", round(100 * df_US_sub[df_US_sub.state_val == 1].size/df_US_sub.size, 1))
fig, ax = plt.subplots(1,1, figsize=(15, 5))

a = None

a = round(100 * df_US_sub[df_US_sub.state_val == 1].category.value_counts()/df_US_sub.category.value_counts(), 1)

sns.barplot(x = a.index,y = a.values)

ax.set_xlabel('Category')

ax.set_ylabel('Success Percentage')
print ('Category.......Sucess Percentage')

print (a)
plot_columns = ['launch_year','launch_month','launch_date'] 

x_label = ['Launch Year','Launch Month','Launch Date'] 

y_label = ['Count', 'Sucess Percentage']

fig, axarr = plt.subplots(len(plot_columns), len(y_label), figsize=(20,5*len(plot_columns)))

for i in range(len(plot_columns)):

    a = None

    a = round(100 * df_US_sub[df_US_sub.state_val == 1][plot_columns[i]].value_counts()/df_US_sub[plot_columns[i]].value_counts(), 1)

    # taking the column data/variable for example launch year and count the sucessful number of projects for each year and divide by the total projects in that year. 

    #print (i)

    sns.countplot(df_US_sub[plot_columns[i]],ax=axarr[i][0])

    axarr[i][0].set_xlabel(x_label[i])

    axarr[i][0].set_ylabel(y_label[0])

    sns.barplot(x = a.index,y = a.values,ax=axarr[i][1])

    axarr[i][1].set_xlabel(x_label[i])

    axarr[i][1].set_ylabel(y_label[1])
fig, ax = plt.subplots(1,1, figsize=(25, 5))

sns.barplot(df_US_sub['duration'].value_counts().index,df_US_sub['duration'].value_counts().values)

ax.set_xlabel('Duration')

ax.set_ylabel('Number of Projects')
fig, ax = plt.subplots(1,1, figsize=(20, 5))

df_US_sub['length_name'].value_counts().plot(kind='bar', figsize=(20,5))

ax.set_xlabel('Length of the Project Name')

ax.set_ylabel('Number of Projects')
df_US_sub['backer_per_usd'] = (df_US_sub.backers/df_US_sub.usd_goal_real)

print (df_US_sub.backer_per_usd.round(2).value_counts().head())

df_US_sub['pledge_to_goal'] = (df_US_sub.usd_pledged_real/df_US_sub.usd_goal_real)

print (df_US_sub.pledge_to_goal.round(2).value_counts().head())
backers = df_US_sub['backers'].unique()

launch_year = df_US_sub['launch_year'].unique()

launch_month = df_US_sub['launch_month'].unique()

launch_date = df_US_sub['launch_date'].unique()

state = df_US_sub['state_val'].unique()
df_US_sub.columns
corr = df_US_sub.corr(method = 'pearson')

# plot the heatmap

fig, ax = plt.subplots(1,1, figsize=(10, 10))

fig.suptitle('Correlation heat map', fontsize = 15)

sns.set(font_scale=1)  

sns.heatmap(corr, 

            cmap = 'coolwarm',

            xticklabels=corr.columns,

            yticklabels=corr.columns,

            annot = True,

            fmt = '.2f',

            linewidths = 0.25,

            cbar_kws={"orientation": "vertical"})
df_group = df_US_sub.groupby(['launch_year'])
df_group.head()
df_group.state_val.value_counts().plot(kind='barh', figsize = (15,10))
a = df_test.groupby(['launch_year']).state.value_counts()
year_data=[]

year_sucess_percentage = []

for launch_year, state in df_test.groupby(['launch_year']).state.value_counts().groupby(level=0):

    #print(state[0])

    #print (state[1])        

    year_data.append(launch_year)

    year_sucess_percentage.append(round((state[0]/(state[0]+state[1]))*100))
df_year_percent = pd.DataFrame.from_dict({'Year':year_data, 'Sucess_percentage':year_sucess_percentage}).set_index('Year')

df_year_percent['Failed_percentage'] = 100 - df_year_percent['Sucess_percentage']

print ("Success percentages of projects over the years:")

print (df_year_percent)

df_year_percent.plot(kind='barh',stacked = True, legend = 'True', figsize=(15,10),title='Project percentages for each year')
cat_data=[]

cat_sucess_percentage = []

for category, state in df_test.groupby(['main_category']).state.value_counts().groupby(level=0):

    #print (state[0])

    #print (state[1]) 

    #print (category)

    cat_data.append(category)

    cat_sucess_percentage.append(round((state[0]/(state[0]+state[1]))*100))
df_cat_percent = pd.DataFrame.from_dict({'Category':cat_data, 'Sucess_percentage':cat_sucess_percentage}).set_index('Category')

df_cat_percent['Failed_percentage'] = 100 - df_cat_percent['Sucess_percentage']

print ("Success percentages of projects for the main categories:")

print (df_cat_percent)

df_cat_percent.plot(kind='barh',stacked = True, legend = 'best', figsize=(15,10),title='Sucess percentages for main categories')
country_data=[]

country_sucess_percentage = []

for country, state in df_test.groupby(['country']).state.value_counts().groupby(level=0):

    #print (state[0])

    #print (state[1]) 

    #print (category)

    country_data.append(country)

    country_sucess_percentage.append(round((state[0]/(state[0]+state[1]))*100))
df_country_percent = pd.DataFrame.from_dict({'Country':country_data, 'Sucess_percentage':country_sucess_percentage}).set_index('Country')

print ("Success percentages of projects for different countries:")

print (df_country_percent)

df_country_percent.plot(kind='barh',legend = None, figsize=(10,10),title='Sucess percentages for countries')
