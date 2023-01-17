# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Data visualisation
import seaborn as sns; sns.set(style="whitegrid", color_codes=True)
import matplotlib.pyplot as plt
from IPython.display import display
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load the data from http://data.houstontx.gov/dataset/city-of-houston-harvey-damage-assessment-open-data
df = pd.read_csv('../input/Harvey_BG.csv')
#for convenience
df.columns=df.columns.str.lower()
df.head(2)
#Quick summary
df.info()
#Quick summary for numerical data
summary = pd.DataFrame({'Total' : 
            [df.pop_total.sum(), 
             df.housing_un.sum(), 
             df.count_affe.sum(), 
             round(df.count_affe.sum() / df.housing_un.sum() * 100, 1)]}, 
             index = ['Population','Households','Affected','Affected, %'])
summary
#Statistics for geographic data
def summary(s):
    s1 = pd.Series()
    s1['count'] = s.size
    s1['unique'] = s.nunique()
    s1['top'] = s.mode()[0]
    s1['freq'] = s.value_counts().values[0]
    return s1

df[['state','county','blkgrp','tract','geography']].apply(summary).T
#What are the county names?
df.county.unique()
#Find the county names by checking the geography column with description
display(df[df.county == 157].iloc[0,4])
display(df[df.county == 201].iloc[0,4])
display(df[df.county == 339].iloc[0,4])
#Change numbers to actual names for counties
df.county.replace(to_replace = [157,201,339],
                    value = ['Fort Bend','Harris','Montgomery'], inplace = True)
#What is the distribution of households among counties and block groups?
sns.countplot(x = 'county',data = df, hue='blkgrp').set(title = 'Count per location')
###Summary about the city of Houston geography###
#How many block groups, tracts and geographic locations are in each county?
df_location = df.county.value_counts().to_frame()

#find block groups
d = df.groupby(['county'])['blkgrp'].value_counts()
df_location['blkgrp'] = [d.loc[name].size for name in df_location.index]

#find tracts
d = df.groupby(['county'])['tract'].value_counts()
df_location['tract'] = [d.loc[name].size for name in df_location.index]

#rename and reorder new dataframe with geography info
df_location.rename(columns = {'blkgrp': 'block groups','county': 'location'}, inplace = True)
df_location.index.name = 'county'
df_location=df_location[['block groups', 'tract', 'location']]
display(df_location)

#How many locatinos are in each tract?
df.tract.value_counts().describe().drop(['count','25%','50%','75%']).\
         round(1).to_frame().rename(columns = {'tract': 'locations per tract'})
#Statistics about geographic locations in terms of housing units
sns.distplot(df.housing_un, rug=True, hist=True, kde=True, 
             axlabel='Households per location')\
             .set_title("KDE for number of households per location")
df[['housing_un']].rename(columns={'housing_un': 'households per location'})\
                  .describe().drop('count').round(0).T
###Inconsistency in the data###

#locations with zero housing units but non-zero population
units_zero = df[['housing_un']][(df.housing_un == 0) & (df.pop_total > 0)].size
print('There are {} locations with zero households but non-zero population.'.format(units_zero))

#locations with the number of affected units larger than actual units
units_more_affe = df[['housing_un','count_affe']][df.housing_un < df.count_affe].shape[0]
print('There are {} locations with number of affected units larger than households.'.format(units_more_affe))
#add a column: number of unaffected households
count_unaffe = df.housing_un - df.count_affe
count_unaffe[df.housing_un < df.count_affe] = np.zeros(units_more_affe)
df['count_unaffe'] = count_unaffe

#group by county and blockgroup
grp_county_blkgrp = df.groupby(['county','blkgrp'])

#define additional variables
total_affe = df.count_affe.sum()
total_units = df.housing_un.sum()

#plot percentage of unaffected and affected households
grp_county_blkgrp['count_unaffe','count_affe']\
    .sum().div(total_units).mul(100)\
    .rename(columns={'count_unaffe':'Unaffected','count_affe':'Affected'})\
    .plot(kind='bar', stacked=True,fontsize=12)\
    .set(title = 'Households, %', ylabel='Percent')
#geographic summary for affected households location
cut=5.2
g1 = grp_county_blkgrp['housing_un'].sum().div(total_units).mul(100)
df_location_all = g1[g1 > cut].append(pd.Series(g1[g1 <= cut].sum(), index=['Other']))

g2 = grp_county_blkgrp['count_affe'].sum().div(total_affe).mul(100)
df_location_affe = g2[g2 > cut].append(pd.Series(g2[g2 <= cut].sum(), index=['Other']))

fig, ax = plt.subplots(1,2,figsize=(13, 6))
df_location_all.plot(kind='pie',autopct='%1.1f%%',label ='',
                     colormap = 'cool',fontsize=14,ax=ax[0])\
               .set_title('Overall',fontsize=14)

df_location_affe.plot(kind='pie',autopct='%1.1f%%',label ='',
                      colormap = 'cool',fontsize=14,ax=ax[1])\
                .set_title('Affected',fontsize=14)
#add a column with percentage of affected households per a given location
pct_affe = df.count_affe / df.housing_un
pct_affe[df.housing_un == 0] = np.zeros(units_zero)
pct_affe[df.housing_un < df.count_affe] = np.ones(units_more_affe)
df['pct_affe'] = pct_affe
###occupancy analysis###

#summary of household occupancy in the city of Houston
#here number of occupied/vacant affected households is estimated 
#using percentage of affected units at a given location
df_occupancy = pd.DataFrame({'Overall': [df.occupied.sum(),df.vacant.sum()],
                   'Affected': [round(sum(df.occupied * df.pct_affe),0),
                                round(sum(df.vacant * df.pct_affe),0)]},
                    index=['Occupied', 'Vacant'])

df_occupancy.plot(kind='pie',autopct='%1.1f%%', subplots=True,
                  label = ['',''], colormap = 'Wistia', fontsize=14, 
                  legend=False, figsize=(11, 5))
df_occupancy
###age analysis###

#summary of household occupancy in the city of Houston
other_age = df.housing_un - df.seniorciti - df.children_u
df_age = pd.DataFrame({'Overall': [df.seniorciti.sum(),df.children_u.sum(),other_age.sum()],
                   'Affected': [round(sum(df.seniorciti * df.pct_affe),0),
                                round(sum(df.children_u * df.pct_affe),0),
                                round(sum(other_age * df.pct_affe),0)]},
                    index=['Senior', 'Children','Other'])

df_age.plot(kind='pie',autopct='%1.1f%%', subplots=True,
                  label = ['',''], colormap = 'Wistia', fontsize=14, legend=False, figsize=(11, 5))
df_age
###income analysis###

#add a column with salary ranges
salary_bins = list(np.array(range(0,121000,20000))) \
              + [df.mhi_acs.max()]
salary_labels = ['<20k', '[20k,40k]', '[40k,60k]','[60k,80k]', 
                 '[80k,100k]', '[100k,120k]', '>120k']

salary_categories = pd.cut(df['mhi_acs'], bins = salary_bins, 
                           labels = salary_labels, include_lowest = True)
salary_categories.cat.set_categories(salary_labels, ordered = True,
                                     inplace = True)

df['income'] = salary_categories
#summary on affected househods versus income
group_income = df.groupby('income')
df_income = group_income[['housing_un']].agg(np.sum)
                
df_income['households, %'] = group_income['housing_un'].agg(np.sum)\
                             .div(total_units).mul(100).round(2)
df_income['affected, %'] = group_income['count_affe'].agg(np.sum)\
                             .div(total_affe).mul(100).round(2)
df_income['affected relative, %'] = group_income['count_affe']\
                             .agg(np.sum).div(group_income['housing_un'].agg(np.sum))\
                             .mul(100).round(2)
df_income
df_income.reset_index(inplace = True)
#plot household income distribution in Houston
fig, ax = plt.subplots(1,3,figsize=(20, 5))
sns.barplot(x='income', y='households, %', data=df_income,ax=ax[0])\
   .set(ylabel='Percent, %',ylim=(0,40),
        title = 'Overall household income distribution')
sns.barplot(x='income', y='affected, %', data=df_income,ax=ax[1])\
   .set(ylabel='Percent, %',ylim=(0,40),
        title = 'Income distribution of affected households')
sns.barplot(x='income', y='affected relative, %', data=df_income,ax=ax[2])\
   .set(ylabel='Percent, %',ylim=(0,40),
        title = 'Distribution of affected households relative to income category')  
###race analysis###

#select the race columns and rename using available data dictionary
columns_race = [x for x in df.columns if 'nh' in x];
columns_race.append('hispanic');
columns_race_rename = ['White','Black', 
                       'American Indian and Alaska Native', 
                       'Asian','Native Hawaiian and Other Pacific', 
                       'Other','Two and More','Hispanic']

#summary on affected househods versus race
race = np.array([getattr(df,col).sum() for col in columns_race])
df_race = pd.DataFrame(data = race,index = columns_race_rename, 
                       columns = ['population'])
df_race['population, %'] = round(df_race.population / df_race.population.sum() * 100,2)


#estimate affected households versus race
df_race['affected, %'] = np.array([(getattr(df,col) / df.pop_total * df.housing_un * df.pct_affe).sum() 
                             / total_affe * 100 for col in columns_race]).round(2)
df_race['affected relative, %'] = np.array([(getattr(df,col) / df.pop_total * df.housing_un * df.pct_affe).sum() 
                             / (getattr(df,col) / df.pop_total * df.housing_un).sum() * 100 
                            for col in columns_race]).round(2)

df_race.sort_values(by = ['population'], inplace = True, ascending = False)
df_race.index.name = 'race'

df_race
df_race.reset_index(inplace = True)
#plot race distribution in Houston
fig, ax = plt.subplots(3,1,figsize=(7,17))

sns.barplot(y='race', x='population, %', data=df_race, palette="cubehelix",ax=ax[0])\
   .set(xlabel = 'Population, %',ylabel='',xlim=(0,41),
        title = 'Race distribution')
sns.barplot(y='race', x='affected, %', data=df_race, palette="cubehelix",ax=ax[1])\
   .set(xlabel = 'Population, %',ylabel='',xlim=(0,41),
        title = 'Race distribution of affected people')
sns.barplot(y='race', x='affected relative, %', data=df_race, palette="cubehelix",ax=ax[2])\
   .set(xlabel = 'Population, %',ylabel='',xlim=(0,41),
        title = 'Distribution of affected people relative to race')
df_race.set_index('race',inplace=True)
df_income.set_index('income',inplace=True)
###income and race bivariate analysis###

#select colunms in original dataframe
columns_race_select = ['hispanic','nh_white','nh_black','nh_asian','race_other']
columns_race_group = [x for x in columns_race if x not in columns_race_select]
#add new column with the values for grouped other races
df['race_other'] = df.loc[:,columns_race_group].sum(axis=1)

for col in columns_race_select:
    df[col + '_affe'] = getattr(df,col) / df.pop_total * df.housing_un * df.pct_affe
    df_income[col + '_affe'] = df.groupby('income')[col + '_affe']\
                                 .agg(np.sum).div(total_affe).mul(100)
    df_income[col + '_affe_rel'] = round(df_income[col + '_affe'] /
                                         df_income[col + '_affe'].sum() * 100, 2)

columns_affe = [x for x in df_income.columns if x.endswith('_affe')]

df_income_race_affe = df_income[columns_affe]

#rename the colums
df_income_race_affe.columns = [x.replace('_affe','').replace('nh_','').replace('race_','').capitalize()
                               for x in df_income_race_affe.columns]

#plot the data
fig, ax = plt.subplots(1,2,figsize=(15,5))
df_income_race_affe.plot(kind='bar',fontsize=12,ax=ax[0])\
                   .set(ylabel='Percent,%',title='Household income and race distribution')
sns.heatmap(df_income_race_affe,annot=True,fmt=".1f",cmap="Reds",ax=ax[1])\
   .set(xlabel='race')  
#relative affected for a given race 
columns_affe_rel = [x for x in df_income.columns if x.endswith('_rel')]
df_income_race_affe_rel = df_income[columns_affe_rel]
df_income_race_affe_rel.columns = [x.replace('_affe_rel','').replace('nh_','').replace('race_','').capitalize()
                               for x in df_income_race_affe_rel.columns]

df_income_race_affe_rel.plot(kind='bar',fontsize=12)\
                       .set(ylabel='Percent, %',title='Income and race distribution relative to race')
df_income_race_affe_rel.style.highlight_max(color = 'red')
###Summary of the affected households in the city of Houston###

fig, ax = plt.subplots(2,3,figsize = (18,10))
df_location_affe.plot(kind='pie',autopct='%1.1f%%',label ='',
                      colormap = 'cool',fontsize=14,ax=ax[0,0])\
                .set_title('Geography',fontsize=14)
df_occupancy.Affected.plot(kind='pie',autopct='%1.1f%%', 
                  label = '', colormap = 'Wistia', fontsize=14, ax=ax[0,1])\
                     .set_title('Occupancy',fontsize=14)
    
df_age.Affected.plot(kind='pie',autopct='%1.1f%%', 
                  label = '', colormap = 'Wistia', fontsize=14, ax=ax[0,2])\
               .set_title('Age',fontsize=14)
    
df_income['affected, %'].plot(kind='pie',autopct='%1.1f%%',label ='',
                              colormap = 'Wistia',fontsize=14,ax=ax[1,0])\
                        .set_title('Income',fontsize=14)
    
g = df_race['affected, %']; cut=5
df_race_affe = g[g > cut].append(pd.Series(g[g <= cut].sum(), index=['Other']))
df_race_affe.plot(kind='pie',autopct='%1.1f%%',label ='',
                      colormap = 'summer',fontsize=14,ax=ax[1,1])\
              .set_title('Race',fontsize=14)
    
sns.heatmap(df_income_race_affe,annot=True,fmt=".1f",cmap="Reds",ax=ax[1,2])\
   .set(xlabel='Race',ylabel='Income')

for i in range(3):
    ax[0,i].set_aspect('equal')
    ax[1,i].set_aspect('equal')
