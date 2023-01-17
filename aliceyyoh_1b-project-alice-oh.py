# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # matplotlib

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read the mortality rate file that I uploaded in Kaggle Data
dataset = pd.read_csv("../input/project1/Mortality_Rate.csv")
df = pd.DataFrame(dataset)

# print to check
df.head()
# Get the data only with mortality rate "Child mortality rate (aged 1-4 years)" under Indicator column
df = df[df['Indicator'] == 'Child mortality rate (aged 1-4 years)']

# Rename the columns (delete '-06' & 'Geograpphic areas' to 'Countries') to make it simpler to write codes 
df.columns = df.columns.str.replace('-06','')
df.rename(columns={"Geographic Area": "Countries"}, inplace=True)

# Make new columns of the averages for every 5 years
df['avr_01_05'] = df[['2001', '2005']].mean(axis=1)
df['avr_06_10'] = df[['2006', '2010']].mean(axis=1)
df['avr_11_15'] = df[['2011', '2015']].mean(axis=1)
df['avr_16_18'] = df[['2016', '2018']].mean(axis=1)

df
# get total population (not dividing into male and female)
tdf = df[df['Sex'] == 'Total']

# drop 'Sex', 'Indicator' column
tdf = tdf.drop(columns=['Sex', 'Indicator'])

# result of mortality rates higher than 40% // tdf: right before moving the first column to index
tdf = tdf[tdf['2000'] > 40]
tdf.head()
# make Countries column index

rslt_tdf = tdf.set_index('Countries')
rslt_tdf.head()
# importing heatmap

import seaborn as sb
# plotting a heatmap based on the table 
fig, ax = plt.subplots(figsize=(20,15))
sb.heatmap(rslt_tdf, cmap="coolwarm",linewidths=.5, ax=ax)

# putting titles and labels 
plt.title('Child Mortality Rate for 1-4 Years',fontsize = 18, fontweight='bold')
plt.ylabel('Countries',fontsize = 18, fontweight='bold')
plt.xlabel('Years ',fontsize = 18, fontweight='bold')

plt.show()
# read the literacy rate file that I uploaded in Kaggle Data
dataset2 = pd.read_csv("../input/project1/Literacy_Rate.csv")
ldf = pd.DataFrame(dataset2)

# print to check
ldf.head()
# Rename the column 'Geograpphic areas' to 'Countries' to make it simpler to write codes 
ldf.rename(columns={"Geographic Area": "Countries"}, inplace=True)

# Make new columns of the averages for every 5 years
ldf['avr_01_05'] = ldf[['2001', '2005']].mean(axis=1)
ldf['avr_06_10'] = ldf[['2006', '2010']].mean(axis=1)
ldf['avr_11_15'] = ldf[['2011', '2015']].mean(axis=1)
ldf['avr_16_18'] = ldf[['2016', '2018']].mean(axis=1)

# get total population (not dividing into male and female)
tldf = ldf[ldf['Sex'] == 'Total']

# drop 'Sex', 'Indicator' column
tldf = tldf.drop(columns=['Sex', 'Indicator'])

# result of literacy rates lower than 90% in 2001-2005
# there are not many countries under 90% so I decided to check all countries
tldf = tldf[tldf['avr_01_05'] < 100]


# make Countries column index
rslt_tldf = tldf.set_index('Countries')

# print
rslt_tldf
# plotting a heatmap based on the table 
fig, ax = plt.subplots(figsize=(20,15))
sb.heatmap(rslt_tldf, cmap="BuGn",linewidths=.5, ax=ax)

# putting titles and labels 
plt.title('Literacy rate',fontsize = 18, fontweight='bold')
plt.ylabel('Countries',fontsize = 18, fontweight='bold')
plt.xlabel('Years ',fontsize = 18, fontweight='bold')

plt.show()
# need to unpivot the table for both files to have all values in the same column.

# mortality file first - unpivoting, getting data of 01-05 average & 16-18 average
tdf = pd.melt(tdf, id_vars =['Countries'], var_name='Year', value_name='Mortality Rate')
tdf05 = tdf[tdf['Year'] == 'avr_01_05']
tdf18 = tdf[tdf['Year'] == 'avr_16_18']

# literacy file unpivoting, getting data of 01-05 average & 16-18 average
tldf = pd.melt(tldf, id_vars =['Countries'], var_name='Year', value_name='Literacy Rate')
tldf05 = tldf[tldf['Year'] == 'avr_01_05']
tldf18 = tldf[tldf['Year'] == 'avr_16_18']


# get rid of the Year column - use drop method
tdf05 = tdf05.drop(columns=['Year'])
tdf18 = tdf18.drop(columns=['Year'])
tldf05 = tldf05.drop(columns=['Year'])
tldf18 = tldf18.drop(columns=['Year'])

#test
tldf18

# I wanna know how to join the two datasets corresponding with 2 columns at the same time (year and countries).
# Because I couldn't find a way to do this, I splited dataframes into 01-05 and 16-18.
# create a new dataframe to find both rates in one dataframe
# use merge method
# need to setup index here
tdf05 = tdf05.set_index('Countries')
tdf18 = tdf18.set_index('Countries')
tldf05 = tldf05.set_index('Countries')
tldf18 = tldf18.set_index('Countries')

df3 = pd.merge(tdf05, tldf05, left_index=True, right_index=True)
df4 = pd.merge(tdf18, tldf18, left_index=True, right_index=True)
# df3 = pd.concat([tldf18, tdf18], axis=1, join='inner') -> not working.

# test
df3
# df4
# style
plt.style.use('ggplot')

# To bring each variables on the X and Y axis
# Bring the two data entries to X and Y
# X is mortality rate and Y is literacy rate

X = df3.iloc[:, 0].values.reshape(-1,1)
Y = df3.iloc[:, 1].values.reshape(-1,1)

# first plot
plt.figure(figsize=(8,8))
plt.scatter(X, Y, s=30)
plt.title('Correlation between child mortality rate and child literacy rate - rates in average from 2001 to 2005',fontsize = 20, fontweight='bold')
plt.xlabel('Mortality rate', fontsize = 15, fontweight='bold')
plt.ylabel('Literacy rate', fontsize = 15, fontweight='bold')
plt.show()

# second plot
X2 = df4.iloc[:, 0].values.reshape(-1,1)
Y2 = df4.iloc[:, 1].values.reshape(-1,1)

plt.figure(figsize=(8,8))
plt.scatter(X2, Y2, s=30)
plt.title('Correlation between child mortality rate and child literacy rate - rates in average from 2016 to 2018', fontsize = 20, fontweight='bold')
plt.xlabel('Mortality rate', fontsize = 15, fontweight='bold')
plt.ylabel('Literacy rate', fontsize = 15, fontweight='bold')
plt.show()
# mortality rate: df
# literacy rate: ldf

# drop columns except the 2018 data / drop Indicator column - mortality
df.drop(df.iloc[:, 3:21], inplace = True, axis = 1)
df.drop(df.iloc[:, 4:8], inplace = True, axis = 1)
df = df.drop(columns=['Indicator'])
# df

# literacy data - do the same
ldf.drop(ldf.iloc[:, 3:21], inplace = True, axis = 1)
ldf.drop(ldf.iloc[:, 4:8], inplace = True, axis = 1)
ldf = ldf.drop(columns=['Indicator'])
ldf

# first column to index
ndf = df.set_index('Countries')
nldf = ldf.set_index('Countries')

# drop based on rows (NaN, Total)
ndf = ndf[ndf.Sex != 'Total'] # mortality
nldf = nldf[nldf.Sex != 'Total'] # literacy


# ndf
# pivot table have two columns of male and female
ndf = pd.pivot_table(ndf, values='2018', index=['Countries'], columns=['Sex']) # mortality

nldf = pd.pivot_table(nldf, values='2018', index=['Countries'], columns=['Sex']) # literacy

nldf

# mortality
# mort_rate = df.groupby('Country')['Male','Female'].mean().sort_values(by='Deaths', ascending=False)
import seaborn as sns

# mortality rate

# to prevent too many bar graphs showing on the chart, I am refining the number of countries -> mortality rate over 40%
ndf = ndf[ndf['Female'] > 20]

# sort in advance
ndf = ndf.sort_values('Female')

Comparison_mort = ndf.plot(kind = 'barh', figsize=(20,10), legend = True, linewidth=20)

# plot the bar chart
# plt.title('title',fontsize = 18, fontweight='bold')
# plt.ylabel('Country',fontsize = 18, fontweight='bold')
# plt.xlabel('M Rate',fontsize = 18, fontweight='bold')
# plt.xticks(rotation=75)
# plt.show()


# average death rate - dereived from cancer only table
#Comparison_mort = ndf.groupby('Countries')['Male', 'Female'].mean().sort_values(by=['Male'], ascending=False)
#mort_bar = Comparison_mort.plot(kind = 'bar', figsize=(20,40), legend = True, linewidth=20)

# plot the bar chart
plt.title('Child mortality rate comparison by sex and country',fontsize = 25, fontweight='bold')
plt.ylabel('Country',fontsize = 18, fontweight='bold')
plt.xlabel('Child mortality rate', fontsize = 18, fontweight='bold')
plt.xticks(rotation=75)
plt.show()
# literacy rate

# to prevent too many bar graphs showing on the chart, I am refining the number of countries -> literacy rate over 94%
nldf = nldf[nldf['Female'] < 94]

# sort in advance
nldf = nldf.sort_values('Female')

Comparison_mort = nldf.plot(kind = 'barh', figsize=(20,10), legend = True, linewidth=20)


# plot the bar chart
plt.title('Child literacy rate comparison by sex and country',fontsize = 25, fontweight='bold')
plt.ylabel('Country',fontsize = 18, fontweight='bold')
plt.xlabel('Child literacy rate', fontsize = 18, fontweight='bold')
plt.xticks(rotation=75)
plt.show()