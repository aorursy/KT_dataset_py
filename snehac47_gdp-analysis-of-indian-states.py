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
#importing data from CSV file into pandas dataframe



import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

 

df_rawgsdp = pd.read_csv('/kaggle/input/ab40c054-5031-4376-b52e-9813e776f65e.csv.csv')

df_rawgsdp
##Remove the rows: '(% Growth over the previous year)' and 'GSDP - CURRENT PRICES (` in Crore)' for the year 2016-17

df_rawgsdp = df_rawgsdp[df_rawgsdp.Duration != '2016-17']



#Setting index to item duration so that i can chose rows based on item description value

df_rawgsdp = df_rawgsdp.set_index('Items  Description')

df_rawgsdp
#Dividing the dataframe into two part for GSDP Value and %Growth Value

df_gsdpcurrent = df_rawgsdp.filter(like='GSDP', axis=0)

df_gsdpgrowth = df_rawgsdp.filter(like='Growth', axis=0)



#using transpose for unpivoting and to have states in column

df_gsdpcurrent = df_gsdpcurrent.set_index('Duration').T

df_gsdpgrowth = df_gsdpgrowth.set_index('Duration').T



df_gsdpcurrent.index.name = 'States'

df_gsdpgrowth.index.name = 'States'



df_gsdpcurrent = df_gsdpcurrent.add_prefix('GSDP_')

df_gsdpgrowth = df_gsdpgrowth.add_prefix('Percentage Growth ')



#checking data

del df_gsdpcurrent.columns.name

df_gsdpcurrent.head(10)
#checking data

del df_gsdpgrowth.columns.name

df_gsdpgrowth.head(10)
#dropping row for year 2012-13 because analysis only has to be done for 2013-14, 2014-15 and 2015-16

df_gsdpgrowth = df_gsdpgrowth.drop('Percentage Growth 2012-13', axis=1)



#dropping row for West Bengal Value since it contains no data

df_gsdpgrowth = df_gsdpgrowth.dropna(axis=0, thresh=1)

df_gsdpgrowth
#checking dtypes of columns



df_gsdpgrowth['Average Growth Percentage'] = df_gsdpgrowth.mean(axis=1)

df_gsdpgrowth=df_gsdpgrowth.sort_values(by='Average Growth Percentage', ascending = False)

df_gsdpgrowth=df_gsdpgrowth.round({'Average Growth Percentage': 2})

df_gsdpgrowth
df_gsdpgrowth_avg=df_gsdpgrowth.filter(like='Average', axis=1)

del df_gsdpgrowth_avg.columns.name

df_gsdpgrowth_avg
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(15,10))

plot_gsdp_meangrowth = sns.barplot(x=df_gsdpgrowth['Average Growth Percentage'], y=df_gsdpgrowth.index, data=df_gsdpgrowth)

plt.xlabel("Average Growth Percentage")

plt.ylabel("States")

plt.title("Average Growth Rates of States over 2013 to 2016")

plt.show()
#top 5 states, consistetly growing

df_gsdpgrowth[['Average Growth Percentage']].head()
#bottom 5 states, struggling to grow

df_gsdpgrowth[['Average Growth Percentage']].tail()
#creating a new dataframe with relevant values

df_totalgdp15_16 = df_gsdpcurrent.filter(items=['GSDP_2015-16'], axis=1)



#sorting based on GDP values

df_totalgdp15_16 = df_totalgdp15_16.sort_values(by='GSDP_2015-16', ascending = False)



#dropping rows with null values and all india GDP value from dataframe

df_totalgdp15_16 = df_totalgdp15_16.dropna()

df_totalgdp15_16 = df_totalgdp15_16.drop('All_India GDP', axis=0)

del df_totalgdp15_16.columns.name

df_totalgdp15_16
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(15,10))

plot_totalgsdp = sns.barplot(x=df_totalgdp15_16['GSDP_2015-16'], y=df_totalgdp15_16.index, data=df_totalgdp15_16)

plt.xlabel("Total GDP of States")

plt.ylabel("States")

plt.title("GSDP for all States in 2015-2016")

plt.show()
#top 5 states

df_totalgdp15_16.head()
#Bottom 5 states

df_totalgdp15_16.tail()
import pandas as pd

import os

#dirs=os.listdir('/kaggle/input')

dirs = os.listdir('../input/')

df_1=[ ]

for items in dirs:

    if items.find('GSVA')>0 and items.find('csv')>0:

        x="../input/"

        i=x+items

        df_temp2=pd.read_csv((i), encoding='ISO-8859-1')

        df_temp2=df_temp2.loc[::,['S.No.','Item','2014-15']]

        df_temp2['State']=items.split('-')[1]

        df_1.append(df_temp2)

mastergdp=pd.concat(df_1,axis=0, sort=False)

mastergdp.State = mastergdp.State.str.replace('_', ' ')

mastergdp.head(10)
#creating a new dataframe with relevant values

df_gdp_percapita=mastergdp.loc[32,['2014-15','State']]



df_gdp_percapita = df_gdp_percapita.set_index('State')

df_gdp_percapita.rename(columns = {'2014-15':'Per Capita GDP in 2014 15'}, inplace = True)



#sorting based on GDP values

df_gdp_percapita = df_gdp_percapita.sort_values(by='Per Capita GDP in 2014 15', ascending = False)

df_gdp_percapita

df_gdp_percapita_x=df_gdp_percapita.copy()
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(15,10))

plot_percapitagdp = sns.barplot(x=df_gdp_percapita['Per Capita GDP in 2014 15'], y=df_gdp_percapita.index, data=df_gdp_percapita)

plt.xlabel("Per Capital GDP of States")

plt.ylabel("States")

plt.title("Per Capita GDP for States in 2014-2015")

plt.show()
# Identifying the top 5 states

df_gdp_percapita.head()
# Identifying the bottom 5 states

df_gdp_percapita.tail()
Ratio_highest_lowest = round(max(df_gdp_percapita['Per Capita GDP in 2014 15'])/min(df_gdp_percapita['Per Capita GDP in 2014 15']),2)

print ("The Ratio of highest GDP per capita to lowest GDP per capita is",Ratio_highest_lowest)
#creating a new dataframe with relevant values

df_gdpcontribution=mastergdp.loc[mastergdp['Item'].isin(['Primary','Secondary','Tertiary','Gross State Domestic Product']), ['Item','2014-15','State']]

df_gdpcontribution.reset_index(drop=True)

df_gdpcontribution.head()
# Cleaning and preparing data for analyis

df_gdpcontribution.rename(columns = {'2014-15':'Total GSDP 2014-15'}, inplace = True)

df_gdpcontribution =df_gdpcontribution.pivot(index='State', columns='Item', values='Total GSDP 2014-15')

df_gdpcontribution = df_gdpcontribution.sort_values(by='Gross State Domestic Product', ascending = False)

columnsTitles = ['Primary','Secondary','Tertiary','Gross State Domestic Product']

df_gdpcontribution = df_gdpcontribution.reindex(columns=columnsTitles)

del df_gdpcontribution.columns.name

df_gdpcontribution.head()
df_gdpcontribution_1=df_gdpcontribution.iloc[:,0:3].apply(lambda s: s*100 / df_gdpcontribution.iloc[:, 3])

df_gdpcontribution_1=df_gdpcontribution_1.add_suffix('_Percentage_Contribution')

df_gdpcontribution_1.head()
colors = ["#808080", "#00FA9A","#20B2AA"]

df_gdpcontribution_1.loc[:,['Primary_Percentage_Contribution','Secondary_Percentage_Contribution','Tertiary_Percentage_Contribution']].plot.barh(stacked=True, color=colors, figsize=(15,12))

plt.show()
#creating a new dataframe with relevant values

df_gdp_percapita_1=mastergdp.loc[mastergdp['Item'].isin(['Per Capita GSDP (Rs.)']), ['Item','2014-15','State']]

df_gdp_percapita_1=df_gdp_percapita_1.set_index('State')

df_gdp_percapita_1.head()
# Cleaning Data

df_gdp_percapita_1 = df_gdp_percapita_1.drop('Item', axis=1)

df_gdp_percapita_1.rename(columns = {'2014-15':'GDP Per Capita in 2014-2015'}, inplace = True)

df_gdp_percapita_1 = df_gdp_percapita_1.sort_values(by='GDP Per Capita in 2014-2015', ascending = False)

df_gdp_percapita_1
#Dividing it into 4 quantiles based on q value

df_gdp_percapita_1['Quantile_rank']=pd.qcut(df_gdp_percapita_1['GDP Per Capita in 2014-2015'],q=[0,0.20,0.5,0.85,1], labels=['C4','C3','C2','C1'])

df_gdp_percapita_2=df_gdp_percapita_1.drop('GDP Per Capita in 2014-2015', axis=1)

df_gdp_percapita_2
#Merging Quantile Values with main dataset

df_merged = pd.merge(df_gdp_percapita_2, mastergdp, on='State')

df_merged.head()
#Removing Total Values

df_merged = df_merged[df_merged['S.No.'] != 'Total']



df_merged.loc[:, ['S.No.']] = df_merged.loc[:, ['S.No.']].astype(float)



#Removing sub-sub sectors

df_merged.set_index('S.No.', inplace=True)

df_merged_1=df_merged.filter(like='.0', axis=0)

df_merged_1.drop([12.0,13.0,14.0,16.0,17.0], axis=0, inplace=True)



df_merged_1=df_merged_1.reset_index(drop=True)

df_merged_1.rename(columns = {'2014-15':'GDP per Sector'}, inplace = True)

df_merged_1.head()
#Dividing merged dataframe into 4 different dataframes for C1, C2, C3, C4

df_merged_c1=df_merged_1.loc[df_merged_1['Quantile_rank'] == 'C1']

df_merged_c2=df_merged_1.loc[df_merged_1['Quantile_rank'] == 'C2']

df_merged_c3=df_merged_1.loc[df_merged_1['Quantile_rank'] == 'C3']

df_merged_c4=df_merged_1.loc[df_merged_1['Quantile_rank'] == 'C4']

df_merged_c1.head(10)
#Using groupby to aggregate all values belonging to the same sector



def agg_by_sector(df, sector):

    df=df.groupby(['Item'])

    df_1=pd.DataFrame(df['GDP per Sector'].sum().sort_values(ascending = True))

    df_1.rename(columns = {'GDP per Sector':'GDP per Sector for %s' %sector}, inplace = True)

    return (df_1)

    print(df_1)    
df_c1=agg_by_sector(df_merged_c1, 'C1')

df_c1
#calculating percentage contribution of each sector



def percentage_contribution(df,sector):

    df['Percentage Contribution for %s'%sector] = df.iloc[0:-1, :].apply(lambda s: s*100 / df.iloc[-1,0])

    df=df.drop('Gross State Domestic Product', axis=0)

    df = df.sort_values(by='Percentage Contribution for %s'%sector, ascending = False)

    return df

    print(df)
df_c1=percentage_contribution(df_c1,'C1')

df_c1
#calculating cumulative sum 

def cumulative_sum(df,sector):

    df['Cumulative Sum of GDP Contribution for %s'%sector] = df["GDP per Sector for %s"%sector].cumsum()

    df['Cumulative Percentage of GDP Contribution for %s'%sector] = df["Percentage Contribution for %s"%sector].cumsum()

    return df

    print (df)
df_c1=cumulative_sum(df_c1,'C1')

df_c1
#selecting categories with ~80% contribution to GDP

C1_categories = df_c1.loc[(df_c1['Cumulative Percentage of GDP Contribution for C1']  < 82)]

C1_categories.iloc[:, [-1]]
#plotting a pie chart

def plot_pie(df,sector):

    plot_x = df.plot.pie(y='Percentage Contribution for %s'%sector, figsize=(7, 7))

    plot_x.legend_ = None

    plt.show()

    

plot_pie(df_c1,'C1')
df_c2=agg_by_sector(df_merged_c2, 'C2')

df_c2=percentage_contribution(df_c2,'C2')

df_c2=cumulative_sum(df_c2,'C2')

df_c2
#selecting categories with ~80% contribution to GDP

C2_categories = df_c2.loc[(df_c2['Cumulative Percentage of GDP Contribution for C2']  < 82)]

C2_categories.iloc[:, [-1]]
plot_pie(df_c2,'C2')
df_c3=agg_by_sector(df_merged_c3, 'C3')

df_c3=percentage_contribution(df_c3,'C3')

df_c3=cumulative_sum(df_c3,'C3')

df_c3
#selecting categories with ~80% contribution to GDP

C3_categories = df_c3.loc[(df_c3['Cumulative Percentage of GDP Contribution for C3']  < 82)]

C3_categories.iloc[:, [-1]]
plot_pie(df_c3,'C3')
df_c4=agg_by_sector(df_merged_c4, 'C4')

df_c4=percentage_contribution(df_c4,'C4')

df_c4=cumulative_sum(df_c4,'C4')

df_c4
#selecting categories with ~80% contribution to GDP

C4_categories = df_c4.loc[(df_c4['Cumulative Percentage of GDP Contribution for C4']  < 82)]

C4_categories.iloc[:, [-1]]
plot_pie(df_c4,'C4')
#merging contribution for all categories

df_c1.reset_index(drop=False, inplace=True)

df_c2.reset_index(drop=False, inplace=True)

df_c3.reset_index(drop=False, inplace=True)

df_c4.reset_index(drop=False, inplace=True)
df_c4

df_merged_all = pd.merge(df_c1, df_c2, on='Item')

df_merged_all = pd.merge(df_merged_all, df_c3, on='Item')

df_merged_all = pd.merge(df_merged_all, df_c4, on='Item')

df_merged_all.set_index('Item', inplace=True)

df_merged_all_1=df_merged_all.filter(like='Percentage Contribution', axis=1)

df_merged_all_1
colors = ["#808080", "#00FA9A","#20B2AA", "#20B2AB"]

df_merged_all_1.loc[:,['Percentage Contribution for C1','Percentage Contribution for C2','Percentage Contribution for C3','Percentage Contribution for C4']].plot.barh(stacked=True, color=colors, figsize=(15,12))

plt.show()
#importing csv to dataframe

df_education_raw = pd.read_csv('/kaggle/input/rs_session243_au570_1.1.csv')

df_education_raw.head()
df_gdp_percapita_x = df_gdp_percapita_x.sort_values(by='State', ascending = True)

df_gdp_percapita_x = df_gdp_percapita_x.reset_index(drop=False)

df_gdp_percapita_x
df_dropout_rates = df_education_raw.copy()

df_dropout_rates= df_dropout_rates.set_index('Level of Education - State', drop=True)

df_dropout_rates
#filtering column only for 2014-2015

df_dropout_rates=df_dropout_rates.filter(like='2014-2015', axis=1)



#Dropping 2nd column as per TA instruction

df_dropout_rates=df_dropout_rates.drop(['Primary - 2014-2015.1','Senior Secondary - 2014-2015'], axis=1)

df_dropout_rates=df_dropout_rates.drop('All India', axis=0)





UT = pd.Series(['A & N Islands','Chandigarh','Dadra & Nagar Haveli','Daman & Diu','Delhi','Jammu and Kashmir','Lakshadweep','Puducherry'])

df_dropout_rates.drop(UT, axis=0, inplace=True)



#data clean up and preparation for analysis

df_dropout_rates= df_dropout_rates.reset_index(drop=False)

df_dropout_rates.rename(columns = {'Level of Education - State':'State','Primary - 2014-2015':'Primary_DropOut_Rate_14_15','Upper Primary - 2014-2015':'Upper_Primary_DropOut_Rate_14_15','Secondary - 2014-2015':'Secondary_DropOut_Rate_14_15'}, inplace = True)

df_dropout_rates.State = df_dropout_rates.State.str.replace('Chhatisgarh', 'Chhattisgarh')

df_dropout_rates.State = df_dropout_rates.State.str.replace('Uttrakhand', 'Uttarakhand')

df_dropout_rates
df_merged_dropout = pd.merge(df_dropout_rates, df_gdp_percapita_x, on='State')

df_merged_dropout = df_merged_dropout .set_index('State', drop=True)

df_merged_dropout = df_merged_dropout.sort_values(by='Per Capita GDP in 2014 15', ascending = True)

df_merged_dropout
# plotting scatter maps

f = plt.figure(figsize=(10,10))



plt.subplot(2, 2, 1)

plt.title('Primary Drop out Rates')

plt.scatter(df_merged_dropout['Primary_DropOut_Rate_14_15'], df_merged_dropout['Per Capita GDP in 2014 15'])



# subplot 2

plt.subplot(2, 2, 2)

plt.title('Upper Primary Drop out Rates')

plt.scatter(df_merged_dropout['Upper_Primary_DropOut_Rate_14_15'], df_merged_dropout['Per Capita GDP in 2014 15'])



# subplot 3

plt.subplot(2, 2, 3)

plt.title('Secondary Drop out Rates')

plt.scatter(df_merged_dropout['Secondary_DropOut_Rate_14_15'], df_merged_dropout['Per Capita GDP in 2014 15'])





plt.show()
#calculating correlation

df_merged_dropout_correlation = df_merged_dropout.corr()

round(df_merged_dropout_correlation, 3)
# figure size

plt.figure(figsize=(10,8))



# heatmap

sns.heatmap(df_merged_dropout_correlation, cmap="YlGnBu", annot=True)

plt.show()