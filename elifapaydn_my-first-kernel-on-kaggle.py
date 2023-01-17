import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import scipy.stats as ss

from matplotlib.gridspec import GridSpec



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df2015=pd.read_csv("/kaggle/input/world-happiness/2015.csv").rename(columns={'Happiness Rank': 'Happiness Rank 2015','Happiness Score': 'Happiness Score 2015'})

df2016=pd.read_csv("/kaggle/input/world-happiness/2016.csv").rename(columns={'Happiness Rank': 'Happiness Rank 2016','Happiness Score': 'Happiness Score 2016'})

df2017=pd.read_csv("/kaggle/input/world-happiness/2017.csv").rename(columns={'Whisker.high':'Upper Confidence Interval','Whisker.low':'Lower Confidence Interval', 'Happiness.Rank': 'Happiness Rank 2017','Happiness.Score': 'Happiness Score 2017'})

df2018=pd.read_csv("/kaggle/input/world-happiness/2018.csv").rename(columns={'Country or region':'Country','Overall rank': 'Happiness Rank 2018','Score': 'Happiness Score 2018'})

df2019=pd.read_csv("/kaggle/input/world-happiness/2019.csv").rename(columns={'Country or region':'Country','Overall rank': 'Happiness Rank 2019','Score': 'Happiness Score 2019'})

df2017.head()
df2015.groupby('Region').size()
df2015.Country=df2015.Country.replace({'Somaliland region': 'Somaliland Region'})



df2017.Country=df2017.Country.replace({'Hong Kong S.A.R., China': 'Hong Kong', 'Taiwan Province of China': 'Taiwan'})

df2017=pd.merge(df2017, df2016[['Country', 'Region']], how='left', on='Country')

df2017.loc[[112, 138, 154], 'Region']='Sub-Saharan Africa'



df2018.Country=df2018.Country.replace({'Hong Kong S.A.R., China': 'Hong Kong', 'Taiwan Province of China': 'Taiwan', 'Trinidad & Tobago': 'Trinidad and Tobago'})

df2018=pd.merge(df2018, df2016[['Country', 'Region']], how='left', on='Country')

df2018.loc[[122, 140, 154], 'Region']='Sub-Saharan Africa'

df2018.loc[37, 'Region']='Latin America and Caribbean'

df2018.loc[57, 'Region']='Western Europe'



df2019.Country=df2019.Country.replace({'Hong Kong S.A.R., China': 'Hong Kong', 'Taiwan Province of China': 'Taiwan', 'Trinidad & Tobago': 'Trinidad and Tobago'})

df2019=pd.merge(df2019, df2016[['Country', 'Region']], how='left', on='Country')

df2019.loc[[119, 122, 134, 143, 154], 'Region']='Sub-Saharan Africa'

df2019.loc[38, 'Region']='Latin America and Caribbean'

df2019.loc[63, 'Region']='Western Europe'

df2019.loc[83, 'Region']='Central and Eastern Europe'
byregion_2015=df2015.groupby('Region').mean()['Happiness Score 2015'].reset_index().sort_values(by='Region', ascending=True)

byregion_2016=df2016.groupby('Region').mean()['Happiness Score 2016'].reset_index().sort_values(by='Region', ascending=True)

byregion_2017=df2017.groupby('Region').mean()['Happiness Score 2017'].reset_index().sort_values(by='Region', ascending=True)

byregion_2018=df2018.groupby('Region').mean()['Happiness Score 2018'].reset_index().sort_values(by='Region', ascending=True)

byregion_2019=df2019.groupby('Region').mean()['Happiness Score 2019'].reset_index().sort_values(by='Region', ascending=True)



byregion_overall=pd.concat([byregion_2015, byregion_2016['Happiness Score 2016'], byregion_2017['Happiness Score 2017'], byregion_2018['Happiness Score 2018'], byregion_2019['Happiness Score 2019']], axis=1)

byregion_overall.head()
byregion_overall=byregion_overall.T

byregion_overall.columns=byregion_overall.iloc[0]

byregion_overall=byregion_overall.reset_index()

byregion_overall.drop(0, inplace=True)



byregion_overall['index']=[2015,2016,2017,2018,2019]

byregion_overall['index']=pd.to_datetime(byregion_overall['index'], format='%Y')

byregion_overall
bycountry_2015=df2015[['Country', 'Region', 'Happiness Score 2015', 'Standard Error']].sort_values(by=['Region', 'Country'], ascending=True)

bycountry_2016=df2016[['Country', 'Region', 'Happiness Score 2016', 'Lower Confidence Interval', 'Upper Confidence Interval']].sort_values(by=['Region', 'Country'], ascending=True)

bycountry_2017=df2017[['Country', 'Region', 'Happiness Score 2017', 'Lower Confidence Interval', 'Upper Confidence Interval']].sort_values(by=['Region', 'Country'], ascending=True)

bycountry_2018=df2018[['Country', 'Region', 'Happiness Score 2018']].sort_values(by=['Region', 'Country'], ascending=True)

bycountry_2019=df2019[['Country', 'Region', 'Happiness Score 2019']].sort_values(by=['Region', 'Country'], ascending=True)
bycountry_2017.head()
from functools import reduce

dfs=[bycountry_2015, bycountry_2016, bycountry_2017, bycountry_2018, bycountry_2019]

bycountry_overall=reduce( lambda left, right: pd.merge(left, right, how='outer', on=['Country', 'Region']), dfs )



happcol=['Happiness Score 2015', 'Happiness Score 2016','Happiness Score 2017']

bycountry_overall['Overall Mean Score']=bycountry_overall[happcol].mean(axis=1)

bycountry_overall=bycountry_overall.rename(columns={'Standard Error':'Standard_Error2015', 'Lower Confidence Interval_x': 'LCI2016','Upper Confidence Interval_x': 'UCI2016','Lower Confidence Interval_y': 'LCI2017','Upper Confidence Interval_y': 'UCI2017'})

bycountry_overall.head()
bycountry_middleeast=bycountry_overall.query(' Region=="Middle East and Northern Africa" ').T

bycountry_middleeast.columns=bycountry_middleeast.loc['Country']

bycountry_middleeast.drop(['Country', 'Region'], inplace=True)

bycountry_middleeast=bycountry_middleeast.reset_index()

bycountry_middleeast
bycountry_middleeast.drop('Oman', axis=1, inplace=True)
#PLOTTING



plt.style.use('seaborn-colorblind')

fig=plt.figure(figsize=[15,15])



gs = GridSpec(6,2, figure=fig, hspace=0.25, wspace=0.2)



ax1 = fig.add_subplot(gs[0:2,0])

bycountry_overall.boxplot(column='Overall Mean Score', by='Region', ax=ax1)



labels=ax1.get_xticklabels()

ax1.set_xticklabels(labels, rotation=35, ha='right')



ax1.set_title('Overall Mean Happiness Scores of Countries in Each Region')

ax1.set_ylabel('Mean Happiness Score')

plt.suptitle("World Happiness Report")

ax1.xaxis.grid(False)

ax1.yaxis.grid(False)

ax1.spines['top'].set_visible(False)

ax1.spines['right'].set_visible(False)



#2nd plot



columns=byregion_overall.columns[1:].to_list()

cm = plt.get_cmap('gist_rainbow')

NUM_COLORS=10

ax2 = fig.add_subplot(gs[0:2,1])

ax2.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

line=byregion_overall.plot('index',columns, marker='o', alpha=0.6, ax=ax2)



ax2.set_title('Happiness Scores Over 2015-2019')

ax2.set_ylabel('Happiness Score')

ax2.set_xlabel('Years')



#highlight Middle East and Northern Africa

position=byregion_overall.columns.get_loc('Middle East and Northern Africa')-1 #minus one because first col is index which we dont use

line.lines[position].set_alpha(1)

line.lines[position].set_linewidth(2)



# Put a legend below current axis

ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),

          fancybox=True, shadow=True, ncol=3, fontsize='small')



ax2.spines['top'].set_visible(False)

ax2.spines['right'].set_visible(False)



#3rd plot: Countries in middle east

N=19

width = 0.25       # the width of the bars

countries=bycountry_middleeast.columns.to_list()[1:]

ax3 = fig.add_subplot(gs[3:,:])

bars1=ax3.bar(np.arange(N), bycountry_middleeast.iloc[0, 1:], label='2015', width=width,

        yerr=bycountry_middleeast.iloc[1,1:],capsize=7,alpha=0.6,

        error_kw={'capsize': 5, 'elinewidth': 2, 'alpha':0.7}) 



bars2=ax3.bar(np.arange(N)+width,bycountry_middleeast.iloc[2, 1:], label='2016',width=width,

       yerr=np.c_[bycountry_middleeast.iloc[2, 1:]-bycountry_middleeast.iloc[3,1:],bycountry_middleeast.iloc[4, 1:]-bycountry_middleeast.iloc[2,1:] ].T,

        alpha=0.6,error_kw={'capsize': 5, 'elinewidth': 2, 'alpha':0.7})



bars3=ax3.bar(np.arange(N)+width*2, bycountry_middleeast.iloc[5, 1:], label='2017',width=width,

       yerr=np.c_[bycountry_middleeast.iloc[5, 1:]-bycountry_middleeast.iloc[6,1:],bycountry_middleeast.iloc[7, 1:]-bycountry_middleeast.iloc[5,1:] ].T,

        alpha=0.6,error_kw={'capsize': 5, 'elinewidth': 2, 'alpha':0.7})

ax3.legend(fancybox=True, shadow=True,fontsize='small')



ax3.set_title('Happiness Scores of Countries in Middle East & North Africa')



ax3.set_xticks(np.arange(N) + width)

ax3.set_xticklabels(countries, rotation=35, ha='right')



ax3.set_ylim([2.5,8])



ax3.set_ylabel('Happiness Score')



[ax3.spines[loc].set_visible(False) for loc in ['top', 'right']] 



#highlight Syria

pos=bycountry_middleeast.columns.get_loc('Syria')-1

bars1[pos].set_alpha(1)

bars2[pos].set_alpha(1)

bars3[pos].set_alpha(1)
#fig.savefig('assignment4.png', dpi=fig.dpi)