# Import the required Libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import glob

from functools import reduce

from itertools import cycle, islice

pd.options.display.float_format='{:.4f}'.format

plt.rcParams['figure.figsize'] = [11.5,8]

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', -1)
# File path. 



path= '../input/eda-gdp-analysis-india/'
# Reading the relevant file on which Analysis needs to be done



file = path + 'SGDP.csv'

dfx = pd.read_csv(file)

dfx.head(4)
# shape of data



dfx.shape
# Data description



dfx.describe()
# Data Information



dfx.info()
# Calculating the Missing Values % contribution in DF



df_null=dfx.isna().mean().round(4) * 100

df_null
# Dropping columns where all rows are NaN



dfx1 = dfx.dropna(axis = 1, how = 'all')
# Dropping the data for Duration 2016-17 as it will not be used in Analysis



dfx2 = dfx1[dfx1.Duration != '2016-17']
# Dropping the UT as it is not needed for Analysis



dfx3 = dfx2.T

dfx4 = dfx3.drop(labels = ['Andaman & Nicobar Islands','Chandigarh','Delhi','Puducherry'])

#dfx3
# Mean of the row (% Growth over previous year) for duration 2013-14, 2014-15 and 2015-16



dfx4_mean = dfx4.iloc[2:,6:10].mean(axis = 1).round(2).sort_values()

dfx4_mean
# Bar Plot for Average growth rates of the various states for duration 2013-14, 2014-15 and 2015-16

plt.rcParams['figure.figsize'] = [11.5,8]

dfx4_mean.plot(kind='barh',stacked=True, colormap = 'Set1')

plt.title("Avg.% Growth of States for Duration 2013-14, 2014-15 and 2015-16", fontweight = 'bold')

plt.xlabel("Avg. % Growth", fontweight = 'bold')

plt.ylabel("States", fontweight = 'bold')
# Average growth rate of my home state against the National average Growth rate



dfx4_myhome = dfx4_mean[['Madhya Pradesh', 'All_India GDP']]
dfx4_myhome.plot(kind='bar',stacked=True, colormap = 'Dark2')

plt.title("Avg. % Growth of Home State vs National Avg. for Duration 2013-14, 2014-15 and 2015-16", fontweight = 'bold')

plt.ylabel("Average % Growth", fontweight = 'bold')

plt.xlabel("Home State Vs National Average", fontweight = 'bold')
#Selecting the GSDP for year 2015-16



dfx5_total_gdp = dfx4.iloc[2:,4:5]
# Dropping the GSDP of All_India as it will not be included in the plot



dfx6_total_gdp = dfx5_total_gdp.drop(labels = ['All_India GDP'])
#Plot for GSDP of all states including States with NaN



dfx6_total_gdp.sort_values(by=4).plot(kind='bar',stacked=True, colormap = 'Set1')

plt.title("Total GDP of States for duration 2015-16" , fontweight = 'bold')

plt.ylabel("Total GDP (in cr)",fontweight = 'bold')

plt.xlabel("States",fontweight = 'bold')
# Dropping the States whose GSDP in NaN for year 2015-16



dfx7_total_gdp = dfx6_total_gdp.dropna().sort_values(by = 4)
#Plot for GSDP of all states excluding States with NaN



dfx7_total_gdp.plot(kind='bar',stacked=True, colormap = 'autumn')

plt.title("Total GDP of States for duration 2015-16" , fontweight = 'bold')

plt.ylabel("Total GDP (in cr)",fontweight = 'bold')

plt.xlabel("States",fontweight = 'bold')
dfx7_total_gdp.shape
# GSDP of Top 5 States

dfx7_total_gdp.tail(5).plot(kind='bar',stacked=True, colormap = 'Dark2')

plt.title("Total GDP of top 5 States for 2015-16", fontweight = 'bold')

plt.ylabel("Total GDP (in cr)",fontweight = 'bold')

plt.xlabel("States",fontweight = 'bold')





# GSDP of Bottom 5 States

dfx7_total_gdp.head(5).plot(kind='bar',stacked=True, colormap = 'Set1')

plt.title("Total GDP of bottom 5 States for 2015-16", fontweight = 'bold')

plt.ylabel("Total GDP (in cr)",fontweight = 'bold')

plt.xlabel("States",fontweight = 'bold')
# Reading all the csv files using glob functionality from a directory for further analysis



dir = path + 'N*.csv'



files = glob.glob(dir)



data = pd.DataFrame()



for f in files:

    dfs = pd.read_csv(f, encoding = 'unicode_escape')

    dfs['State'] = f.replace(path, '').replace('NAD-', '').replace('-GSVA_cur_2016-17.csv','').replace('-GSVA_cur_2015-16.csv','').replace('-GSVA_cur_2014-15.csv','').replace('_',' ')

    data = data.append(dfs)

data = data.iloc[:, ::-1]

sort=True
# Selecting the required columns for the Analysis



df = data[['State', 'Item', '2014-15']] 

df1 = df.reset_index(drop = True)
# Cleansing the columns name



df1['Item'] = df1['Item'].map(lambda x: x.rstrip('*')).copy()

df1 = df1.set_index('State')
# Pivoting the df for enhanced analysis of data



df2 = pd.pivot_table(df1, values = '2014-15', index=['Item'], columns = 'State').reset_index()

df3 = df2.set_index('Item',drop=True)

#df3
# Dropping the UT as it will not be used in further analysis



df4=df3.drop(['Andaman Nicobar Islands','Chandigarh','Delhi','Puducherry'],axis=1)
df5_percapita = df4.loc['Per Capita GSDP (Rs.)'].sort_values()
#Plot for GDP per capita in Rs. for all states



df5_percapita.plot(kind='barh',stacked=True, colormap = 'gist_rainbow')

plt.title("GDP per Capita for All States for duration 2014-15", fontweight = 'bold')

plt.xlabel("GDP per Capita (in Rs.)",fontweight = 'bold')

plt.ylabel("States", fontsize = 12, fontweight = 'bold')
#Plot for GDP per Capita of top 5 States for 2014-15



df5_percapita.tail(5).plot(kind='bar',stacked=True, colormap = 'winter')

plt.title("GDP per Capita of top 5 States for 2014-15", fontweight = 'bold')

plt.ylabel("GDP per Capita (in Rs.)", fontweight = 'bold')

plt.xlabel("States", fontsize = 12, fontweight = 'bold')
#Plot for GDP per Capita of bottom 5 States for 2014-15



df5_percapita.head(5).plot(kind='bar',stacked=True, colormap = 'Set1')

plt.title("GDP per Capita of bottom 5 States for 2014-15", fontweight = 'bold')

plt.ylabel("GDP per Capita (in Rs.)", fontweight = 'bold')

plt.xlabel("States", fontweight = 'bold')
Goa_percapita = (df5_percapita['Goa']/df5_percapita.sum()*100).round(2)

Goa_percapita1 = (df5_percapita['Goa']/df5_percapita.mean()).round(2)

Goa_per_Bihar =  df5_percapita['Goa']/df5_percapita['Bihar']

Sikkim_percapita = (df5_percapita['Sikkim']/df5_percapita.sum()*100).round(2)

Bihar_percapita = (df5_percapita['Bihar']/df5_percapita.sum()*100).round(2)

UP_percapita = (df5_percapita['Uttar Pradesh']/df5_percapita.sum()*100).round(2)
# Ratio of the highest per capita GDP to the lowest per capita GDP



h_percapita = df5_percapita.iloc[-1]

l_percapita = df5_percapita.iloc[0]

percapita_ratio = (h_percapita/l_percapita).round(3)



percapita_ratio
# Selecting Primary Secondary and Tertiary sector for percentage contribution in total GDP



df_gdp_con = df4.loc[['Primary', 'Secondary', 'Tertiary','Gross State Domestic Product']]

df_gdp_percon = (df_gdp_con.div(df_gdp_con.loc['Gross State Domestic Product'])*100).round(2)

df_gdp_percon =df_gdp_percon.T.iloc[:,:3]
# Plot for % contribution of sectors in total GDP



df_gdp_percon.plot(kind='bar',stacked=True, colormap = 'prism')

plt.title("% Contribution of Primary, Secondary, Tertiary sector in total GDP for 2014-15",fontweight = 'bold')

plt.ylabel("% Contribution", fontweight = 'bold')

plt.xlabel("States", fontweight = 'bold')
# Sorting the df for better visualization



df_sort = df4.T.sort_values(by = 'Per Capita GSDP (Rs.)', ascending = False)
# Define the quantile values and bins for categorisation



df_sort.quantile([0.2,0.5,0.85,1], axis = 0)

bins = [0, 67385, 101332, 153064.85, 271793]

labels = ["C4", "C3", "C2", "C1"]

df_sort['Category'] = pd.cut(df_sort['Per Capita GSDP (Rs.)'], bins = bins, labels = labels)

df_index = df_sort.set_index('Category')

df_sum = df_index.groupby(['Category']).sum()

df_rename =  df_sum.rename(columns = {"Population ('00)" : "Population (00)"})
# Selecting the sub sectors which will be used for further analysis



df7_sector = df_rename[['Agriculture, forestry and fishing','Mining and quarrying','Manufacturing','Electricity, gas, water supply & other utility services',

                 'Construction','Trade, repair, hotels and restaurants','Transport, storage, communication & services related to broadcasting','Financial services',

                'Real estate, ownership of dwelling & professional services','Public administration','Other services','Gross State Domestic Product']]
# Calculating and rounding the percentage contribution of each subsector in total GSDP



df8_per = (df7_sector.T.div(df7_sector.T.loc['Gross State Domestic Product'])*100)

df8_round = df8_per.round(2)

df9_per = df8_round.drop('Gross State Domestic Product')

df9_per
# Plot for % Contribution of subsectors in Total GDP for C1 states for 2014-15



df9_per['C1'].sort_values().plot(kind='bar',stacked=True, colormap = 'Accent')

plt.title("% Contribution of subsectors in Total GDP for C1 states for 2014-15", fontweight = 'bold')

plt.xlabel("Sub-sectors", fontweight = 'bold')

plt.ylabel("% Contribution", fontweight = 'bold')
# Plot for % Contribution of subsectors in Total GDP for C2 states for 2014-15



df9_per['C2'].sort_values().plot(kind='bar',stacked=True, colormap = 'Accent')

plt.title("% Contribution of subsectors in Total GDP for C2 states for 2014-15", fontweight = 'bold')

plt.ylabel("% Contribution", fontweight = 'bold')

plt.xlabel("Sub-sectors", fontweight = 'bold')
# Plot for % Contribution of subsectors in Total GDP for C3 states for 2014-15



df9_per['C3'].sort_values().plot(kind='bar',stacked=True, colormap = 'Accent')

plt.title("% Contribution of subsectors in Total GDP for C3 states for 2014-15", fontweight = 'bold')

plt.ylabel("% Contribution", fontweight = 'bold')

plt.xlabel("Sub-sectors", fontweight = 'bold')
# Plot for % Contribution of subsectors in Total GDP for C4 states for 2014-15



df9_per['C4'].sort_values().plot(kind='bar',stacked=True, colormap = 'Accent')

plt.title("% Contribution of subsectors in Total GDP for C4 states for 2014-15", fontweight = 'bold')

plt.ylabel("% Contribution", fontweight = 'bold')

plt.xlabel("Sub-sectors", fontweight = 'bold')
# 80% Contribution by top subsectors in Total GSDP for C1/C2/C3/C4 States 2014-15



fig, axes = plt.subplots(2,2, figsize=(15,12))

fig.tight_layout()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=1.8)





df9 = df9_per.sort_values(by = ['C1', 'C2', 'C3', 'C4'], ascending = False)

topsubsector = df9[df9.C1.cumsum() <= 80]

top_c1 = topsubsector[['C1']]

top_c1.plot(kind='bar',stacked=True, colormap = 'Dark2',ax=axes[0][0])





df9 = df9_per.sort_values(by = ['C2', 'C3', 'C4','C1'], ascending = False)

topsubsector = df9[df9.C2.cumsum() <= 80]

top_c2 = topsubsector[['C2']]

top_c2.plot(kind='bar',stacked=True, colormap = 'Dark2',ax=axes[0][1])





df9 = df9_per.sort_values(by = ['C3', 'C4','C1','C2'], ascending = False)

topsubsector = df9[df9.C3.cumsum() <= 80]

top_c3 = topsubsector[['C3']]

top_c3.plot(kind='bar',stacked=True, colormap = 'prism',ax=axes[1][0])





df9 = df9_per.sort_values(by = ['C4','C1','C2', 'C3'], ascending = False)

topsubsector = df9[df9.C4.cumsum() <= 80]

top_c4 = topsubsector[['C4']]

top_c4.plot(kind='bar',stacked=True, colormap = 'prism',ax=axes[1][1])

# Reading the relevant file on which Analysis needs to be done



file1 = path + 'Dropout rate dataset.csv'

df_dropout = pd.read_csv(file1)
# Renaming the columns which are incorrect



df_rename = df_dropout.rename(columns = {'Primary - 2014-2015' : 'Primary - 2013-2014','Primary - 2014-2015.1' : 'Primary - 2014-2015'})
# Selecting the columns which will be used for further analysis



dfa = df_rename[['Level of Education - State','Primary - 2014-2015','Upper Primary - 2014-2015','Secondary - 2014-2015']] 
# Dropping the union territory because it will not be used in further analysis



dfa1 = dfa.drop([0,5,7,8,9,18,26,35,36])

dfa2 = dfa1.reset_index(drop=True)
# Calculating the Missing Values % contribution in DF



dfa2.isna().mean().round(2) * 100
# Selecting the required column for further analysis



dfa3 = df4.T.reset_index()

dfa4 = dfa3[['State', 'Per Capita GSDP (Rs.)']]
# Concatenating the Education dropout df and Per Capita of States df



dfa5 = pd.concat([dfa2, dfa4], axis = 1)

dfa6 = dfa5.drop(['State'], axis = 1) 

dfa7 = dfa6.set_index('Level of Education - State', drop = True)
# Scatter Plot for GDP per capita with dropout rates in education



f = plt.figure()    

f, axes = plt.subplots(nrows = 2, ncols = 2, sharex=True, sharey = False)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)



sc = axes[0][0].scatter(dfa7['Primary - 2014-2015'],dfa7['Per Capita GSDP (Rs.)'], s=100, c='DarkRed',marker="o")

axes[0][0].set_ylabel('Per Capita GSDP (Rs.)')

axes[0][0].set_xlabel('Primary Education')



sc = axes[0][1].scatter(dfa7['Upper Primary - 2014-2015'],dfa7['Per Capita GSDP (Rs.)'], s=100, c='DarkBlue',marker="*")

axes[0][1].set_ylabel('Per Capita GSDP (Rs.)')

axes[0][1].set_xlabel('Upper Primary Education')



sc = axes[1][0].scatter(dfa7['Secondary - 2014-2015'],dfa7['Per Capita GSDP (Rs.)'], s=100, c='DarkGreen',marker="s")

axes[1][0].set_ylabel('Per Capita GSDP (Rs.)')

axes[1][0].set_xlabel('Secondary Education')
dfa7.plot(kind='scatter',x='Primary - 2014-2015',y='Per Capita GSDP (Rs.)', s=150, c='DarkRed',marker="o")
dfa7.plot(kind='scatter',x='Upper Primary - 2014-2015',y='Per Capita GSDP (Rs.)', s=150, c='DarkRed',marker="*")
dfa7.plot(kind='scatter',x='Secondary - 2014-2015',y='Per Capita GSDP (Rs.)', s=150, c='DarkRed',marker="s")