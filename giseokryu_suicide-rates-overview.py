# for some basic operations

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



# re(=regular expression operations) for changing string to float

import re
# reading the data for Suicide Rates



df = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')



# checking the head of the data



df.head(10)
# show the brief information about df DataFrame



df.info()
# check is there any null object in it



df.isnull().sum()
# delete 'HDI for year' column to remove null object



df = df.drop('HDI for year', axis=1)
# change the name of the columns for easy usage



df=df.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age','population':'Population','suicides_no':'NumOfSuicides', 'suicides/100k pop':'SuicidesPer100k',

                          'country-year':'CountryYear',' gdp_for_year ($) ':'GdpForYear','gdp_per_capita ($)':'GdpPerCapital','generation':'Generation'})
# sort the age

df['Age'] = df['Age'].replace({'5-14 years': '1', '15-24 years': '2', '25-34 years': '3', '35-54 years': '4', '55-74 years': '5', '75+ years': '6'})



# change data type str to int for GdpForYear

df['GdpForYear'] = df['GdpForYear'].apply(lambda x : re.sub("[^\d\.]", "", x))

df['GdpForYear'] = df['GdpForYear'].astype('double')
# check after finishing data cleaning

df.head()
def visualization(Title = None, xlabel = None, ylabel= None, xTitle= None):

    fig = plt.figure()

    plt.style.use(['fivethirtyeight'])

    axes = fig.add_axes([0.3, 0.1, 1, 0.8])

    if Title == 'Year & Suicides':

        axes.plot(xlabel, ylabel, color = 'navy', linewidth=3, ls='--')

    elif Title == 'Gender & Suicides':

        axes.bar(xlabel, ylabel, color = ['crimson', 'navy'], linewidth=3)

    else:

        axes.bar(xlabel, ylabel, color = 'navy', linewidth=3)

    axes.set_xlabel(xTitle, fontsize=12)

    axes.set_ylabel('SucidiesPer100k', fontsize=12)

    axes.set_title(Title, fontsize=15)

    if Title == 'Age & Suicides':

        plt.xticks(fontsize=10.5)

        axes.set_xticklabels(('5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years'))

    if Title == 'Gender & Suicides':

        plt.xticks(fontsize=15)

    return axes

# make new DataFrame call 'df_year', which contains only 'Year', 'NumOfSuicides', 'Population', 'SucidiesPer100k'



df_year = df.drop(['Country', 'Gender', 'Age', 'CountryYear', 'GdpForYear', 'GdpPerCapital', 'Generation'], axis=1)



# group data based on Year



year_group = df_year.groupby(['Year'], as_index=False).sum()





# reset the SucidiesPer100k

year_group['SucidiesPer100k'] = (year_group['NumOfSuicides'] / year_group['Population']) * 100000



year_group.head()
year_visualization = visualization(Title= 'Year & Suicides',xlabel=year_group['Year'], ylabel=year_group['SucidiesPer100k'], xTitle='Year')
# make new DataFrame call 'df_year', which contains only 'Age', 'NumOfSuicides', 'Population', 'SucidiesPer100k'



df_age = df.drop(['Country', 'Gender', 'Year', 'CountryYear', 'GdpForYear', 'GdpPerCapital', 'Generation'], axis=1)



# group data based on Age



age_group = df_age.groupby(['Age'], as_index=False).sum()





# reset the SucidiesPer100k

age_group['SuicidesPer100k'] = (age_group['NumOfSuicides'] / age_group['Population']) * 100000



age_group.head()
age_visualization = visualization(Title= 'Age & Suicides',xlabel=age_group['Age'], ylabel=age_group['SuicidesPer100k'], xTitle='Age')
# make new DataFrame call 'df_year', which contains only 'GdpForYear', 'NumOfSuicides', 'Population', 'SucidiesPer100k'



df_GDP = df.drop(['Country', 'Gender', 'Age', 'CountryYear', 'Year', 'GdpPerCapital', 'Generation'], axis=1)



df_GDP.info()
print("The number of unique value in GdpForYear: ",len(df_GDP['GdpForYear'].unique()))

# check minimum & maximum value in GdpForYear

print("Minimum value in GdpForYear: ", df_GDP['GdpForYear'].min())

print("Maximum value in GdpForYear: ", df_GDP['GdpForYear'].max())
# since the number of GdpForYear is to large, I will make a new column call 'GdpForYearPer$1M', which divides data with 1,000,000(=$1M)

df_GDP['GdpForYearPer$1M'] = df_GDP['GdpForYear'].apply(lambda x : x /1000000)



# sort the data in GdpForYearPer$1M

df_GDP = df_GDP.sort_values(by=['GdpForYearPer$1M'])



# set the range of GdpForYearPer$1M



range_GDP = {'$0 ~ $100M': np.arange(0,100), '$100M ~ $1B': np.arange(100.0000001,1000), '$1B ~ $10B': np.arange(1000.000001, 10000), '$10B ~ $100B': np.arange(10000.000001, 100000),

            '$100B ~ $1T': np.arange(100000.000001, 1000000), '$1T ~': np.arange(1000000.000001, 20000000)}



# generate "cuts" (bins) and associated labels from `range_GDP`.    



cut_data = [(np.min(v), k) for k, v in range_GDP.items()]

bins, labels = zip(*cut_data)



# bins required to have one more value than labels

bins = list(bins) + [np.inf]

df_GDP['GdpForYearPer$1M'] = pd.cut(df_GDP['GdpForYearPer$1M'], bins=bins, labels=labels)



# drop GdpForYear column

df_GDP = df_GDP.drop('GdpForYear', axis=1)



# check the data

df_GDP.head()
# group data based on GdpForYear



GDP_group = df_GDP.groupby(['GdpForYearPer$1M'], as_index=False).sum()



# reset the SucidiesPer100k

GDP_group['SuicidesPer100k'] = (GDP_group['NumOfSuicides'] / GDP_group['Population']) * 100000



GDP_group.head()
GDP_visualization = visualization(Title= 'GDP & Suicides',xlabel=GDP_group['GdpForYearPer$1M'], ylabel=GDP_group['SuicidesPer100k'], xTitle='GdpForYearPer$1M')
# make new DataFrame call 'df_Gene', which contains only 'Generation', 'NumOfSuicides', 'Population', 'SucidiesPer100k'



df_Gene = df.drop(['Country', 'Gender', 'Age', 'CountryYear', 'Year', 'GdpPerCapital', 'GdpForYear'], axis=1)



# group data based on Generation



Gene_group = df_Gene.groupby(['Generation'], as_index=False).sum()



# reset the SucidiesPer100k

Gene_group['SuicidesPer100k'] = (Gene_group['NumOfSuicides'] / Gene_group['Population']) * 100000





Gene_group.head()
# put Generation column to index for reordering



Gene_group = Gene_group.set_index('Generation')



# make order list for Generation

reorderlist = ['G.I. Generation', 'Boomers', 'Generation X', 'Millenials', 'Generation Z']



# reorder the Generation

Gene_group = Gene_group.reindex(reorderlist)



# put Generation index into column

Gene_group = Gene_group.reset_index()



# check the DataFrame

Gene_group.head()
Gene_visualization = visualization(Title= 'Generation & Suicides',xlabel=Gene_group['Generation'], ylabel=Gene_group['SuicidesPer100k'], xTitle='Generation')
# make new DataFrame call 'df_gender', which contains only 'Gender', 'NumOfSuicides', 'Population', 'SucidiesPer100k'



df_gender = df.drop(['Country', 'Age', 'Year', 'CountryYear', 'GdpForYear', 'GdpPerCapital', 'Generation'], axis=1)



# group data based on Gender



gender_group = df_gender.groupby(['Gender'], as_index=False).sum()





# reset the SucidiesPer100k

gender_group['SuicidesPer100k'] = (gender_group['NumOfSuicides'] / gender_group['Population']) * 100000



gender_group.head()
gender_visualization = visualization(Title= 'Gender & Suicides',xlabel=gender_group['Gender'], ylabel=gender_group['SuicidesPer100k'], xTitle='Gender')