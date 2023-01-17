

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



cities_disc_2020_data_dict = pd.read_csv("/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Disclosing/Cities_Disclosing_to_CDP_Data_Dictionary.csv")
cities_disc_2020_data_dict.shape
cities_disc_2020_data_dict.info()
cities_disc_2020_data_dict.head(13)
cities_disc_2020_data = pd.read_csv("/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Disclosing/2020_Cities_Disclosing_to_CDP.csv")
cities_disc_2020_data.shape
cities_disc_2020_data.info()
cities_disc_2020_data.head(10)
cities_resp_2020_data_dict = pd.read_csv("/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Responses/Full_Cities_Response_Data_Dictionary.csv")

cities_resp_2020_data_dict.shape
cities_resp_2020_data_dict.info()
cities_resp_2020_data_dict.head(18)


# read in all our data

cities_resp_2020_data = pd.read_csv("/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Responses/2020_Full_Cities_Dataset.csv")

                            

cities_resp_2020_data_dict = pd.read_csv("/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Responses/Full_Cities_Response_Data_Dictionary.csv")



# set seed for reproducibility

np.random.seed(0) 
cities_resp_2020_data.shape
cities_resp_2020_data.info()
cities_resp_2020_data.head()
cities_resp_2020_data.columns
def display_columns_uniqvals(df):

    for i, col in enumerate(df.columns.tolist()):

         #print('\n ({} {}) Sz:{} \n Uniq: {} '.format(i,col, df[col].unique().size, df[col].unique() ))

        print('\n ({} {}) Sz:{} '.format(i,col, df[col].unique().size ))

    print('\n')
display_columns_uniqvals(cities_resp_2020_data)

cities_resp_2020_data['CDP Region'].unique()
cities_resp_2020_data['Parent Section'].unique()

cities_resp_2020_data['Section'].unique()
plt.style.use('ggplot')



# Create a data frame of CDP Region counts 

parent_section_counts = cities_resp_2020_data['Parent Section'].value_counts()

print(parent_section_counts)



print(parent_section_counts.index.values)

print(parent_section_counts.values)



# Get the figure and the axes (or subplots)

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=False,figsize=(15, 5))





#Axes.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)

#ax0.bar([1,0], height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)

parent_section_counts.plot(kind='bar', ax= ax0)

ax0.set(title = '', xlabel='Parent Section' , ylabel = 'Number of Rows')





# Create a data frame of CDP Region counts 

cdp_region_counts = cities_resp_2020_data['CDP Region'].value_counts()

print(cdp_region_counts)



cdp_region_counts.plot(kind='bar', ax= ax1);

ax1.set(title = '', xlabel='CDP Region' , ylabel = 'Number of Rows')







# Title the figure

fig.suptitle('Frequency Counts', fontsize=20, fontweight='bold');





# Create a data frame of CDP Region counts 

section_counts = cities_resp_2020_data['Section'].value_counts()

print(section_counts)







print(section_counts.index.values)

print(section_counts.values)



# Get the figure and the axes (or subplots)

fig, (ax0) = plt.subplots(nrows=1, ncols=1, sharey=True,figsize=(15, 5))





#Axes.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)

#ax0.bar([1,0], height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)

section_counts.plot(kind='bar', ax= ax0)

ax0.set(title = '', xlabel='Section' , ylabel = 'Number of Rows')









# Title the figure

fig.suptitle('Frequency Counts', fontsize=14, fontweight='bold');

cities_resp_2020_df1 = cities_resp_2020_data.groupby(['Parent Section','CDP Region'])['Organization'].aggregate('count').unstack()

cities_resp_2020_df1







cities_resp_2020_df2 = cities_resp_2020_df1.fillna(0)

cities_resp_2020_df2



#Side-by-side bar plot



# Get the figure and the axes (or subplots)

fig, ax = plt.subplots(figsize=(10, 5))



# Using dataframe's plot

cities_resp_2020_df2.plot(kind='bar', ax=ax, fontsize=10, colormap=plt.cm.coolwarm_r, grid=False)



ax.set(xlabel='Parent Section' , ylabel = 'Number of Rows',title='By each CDP Region in 2020')



fig.suptitle('Frequency of Parent Section', fontsize=14, fontweight='bold');
#### Cities Responses Data Set 2019

# read in all our data

cities_resp_2019_data = pd.read_csv("/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Responses/2019_Full_Cities_Dataset.csv")

                            

cities_resp_2019_data_dict = pd.read_csv("/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Responses/Full_Cities_Response_Data_Dictionary.csv")



# set seed for reproducibility

np.random.seed(0) 

cities_resp_2019_data.shape
cities_resp_2019_data.info()

cities_resp_2019_data.head()

cities_resp_2019_data.columns
display_columns_uniqvals(cities_resp_2019_data)

cities_resp_2019_data['CDP Region'].unique()

cities_resp_2019_data['Parent Section'].unique()

cities_resp_2019_df1 = cities_resp_2019_data.groupby(['Parent Section','CDP Region'])['Organization'].aggregate('count').unstack()

cities_resp_2019_df1







cities_resp_2019_df2 = cities_resp_2019_df1.fillna(0)

cities_resp_2019_df2



#Side-by-side bar plot



# Get the figure and the axes (or subplots)

fig, ax = plt.subplots(figsize=(10, 5))



# Using dataframe's plot

cities_resp_2019_df2.plot(kind='bar', ax=ax, fontsize=10, colormap=plt.cm.coolwarm_r, grid=False)



ax.set(xlabel='Parent Section' , ylabel = 'Number of Rows',title='By CDP Region in 2019')



fig.suptitle('Frequency of Parent Section', fontsize=14, fontweight='bold');
# read in all our data

cities_resp_2018_data = pd.read_csv("/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Responses/2018_Full_Cities_Dataset.csv")

                            

cities_resp_2018_data_dict = pd.read_csv("/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Responses/Full_Cities_Response_Data_Dictionary.csv")



# set seed for reproducibility

np.random.seed(0) 
cities_resp_2018_data.shape
cities_resp_2018_data.info()

cities_resp_2018_data.head()
cities_resp_2018_data.columns
display_columns_uniqvals(cities_resp_2018_data)
cities_resp_2018_data['CDP Region'].unique()

cities_resp_2018_data['CDP Region'].unique().size
cities_resp_2018_data['Parent Section'].unique()
cities_resp_2018_data['Parent Section'].unique().size
cities_resp_2018_df1 = cities_resp_2018_data.groupby(['Parent Section','CDP Region'])['Organization'].aggregate('count').unstack()

cities_resp_2018_df1







cities_resp_2018_df2 = cities_resp_2018_df1.fillna(0)

cities_resp_2018_df2



#Side-by-side bar plot



# Get the figure and the axes (or subplots)

fig, ax = plt.subplots(figsize=(15, 5))



# Using dataframe's plot

cities_resp_2018_df2.plot(kind='bar', ax=ax, fontsize=10, colormap=plt.cm.coolwarm_r, grid=False)



ax.set(xlabel='Parent Section' , ylabel = 'Number of Rows',title='By each CDP Region in 2018')



fig.suptitle('Frequency of Parent Section', fontsize=14, fontweight='bold');
# Get the figure and the axes (or subplots)

fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True,figsize=(15, 5))





cities_resp_2020_df2.plot(kind='bar', ax=ax0, fontsize=10, colormap=plt.cm.coolwarm_r, grid=False,legend=None)

ax0.set(xlabel='Parent Section' , ylabel = '#Rows',title='Year 2020')





cities_resp_2019_df2.plot(kind='bar', ax=ax1, fontsize=10, colormap=plt.cm.coolwarm_r, grid=False)

ax1.set(xlabel='Parent Section' , ylabel = '#Rows',title='Year 2019')



#cities_resp_2018_df2.plot(kind='bar', ax=ax2, fontsize=10, colormap=plt.cm.coolwarm_r, grid=False,legend=None)

#ax2.set(xlabel='Parent Section' , ylabel = 'Number of Rows',title=' Year 2018')





fig.suptitle('Frequency of Parent Section by CDP Region (2020 & 2019)', fontsize=14, fontweight='bold');
# Get the figure and the axes (or subplots)

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True,figsize=(15, 8))





cities_resp_2020_df2.plot(kind='bar', ax=ax0, fontsize=10, colormap=plt.cm.coolwarm_r, grid=False,legend=None)

ax0.set(xlabel='Parent Section' , ylabel = 'Number of Rows',title='Year 2020')





cities_resp_2019_df2.plot(kind='bar', ax=ax1, fontsize=10, colormap=plt.cm.coolwarm_r, grid=False)

ax1.set(xlabel='Parent Section' , ylabel = 'Number of Rows',title='Year 2019')



cities_resp_2018_df2.plot(kind='bar', ax=ax2, fontsize=10, colormap=plt.cm.coolwarm_r, grid=False,legend=None)

ax2.set(xlabel='Parent Section' , ylabel = 'Number of Rows',title=' Year 2018')





fig.suptitle('Year wise Frequency of Parent Section by CDP Region', fontsize=14, fontweight='bold');
# get the number of missing data points per column

missing_cities_resp_2020_count = cities_resp_2020_data.isnull().sum()



# look at the # of missing points in the first ten columns

missing_cities_resp_2020_count[0:17]
# how many total missing values do we have?

total_cells = np.product(cities_resp_2020_data.shape)

total_missing = missing_cities_resp_2020_count.sum()



# percent of data that is missing

percent_missing = (total_missing/total_cells) * 100

print(percent_missing)
# look at the # of missing points in the first ten columns

missing_cities_resp_2020_count[0:10]
# look at the # of missing points in the next remaining columns

missing_cities_resp_2020_count[10:18]