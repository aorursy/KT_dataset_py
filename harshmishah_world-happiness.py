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
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd
yr_2015 = pd.read_csv("/kaggle/input/world-happiness-report/2015.csv")

yr_15 = yr_2015[['Happiness Rank', 'Country', 'Happiness Score','Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)','Freedom', 'Generosity', 'Trust (Government Corruption)']].copy()

yr_15
yr_2016 = pd.read_csv("/kaggle/input/world-happiness-report/2016.csv")

yr_16 = yr_2016[['Happiness Rank', 'Country', 'Happiness Score','Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)','Freedom', 'Generosity', 'Trust (Government Corruption)']].copy()

yr_16
yr_2017 = pd.read_csv("/kaggle/input/world-happiness-report/2017.csv")

yr_2017.rename(columns={'Happiness.Rank':'Happiness Rank','Happiness.Score':'Happiness Score','Health..Life.Expectancy.':'Health (Life Expectancy)','Economy..GDP.per.Capita.':'Economy (GDP per Capita)','Trust..Government.Corruption.':'Trust (Government Corruption)','Dystopia.Residual':'Dystopia Residual','Whisker.high':'Upper Confidence Interval','Whisker.low':'Lower Confidence Interval'},inplace=True)

yr_17 = yr_2017[['Happiness Rank', 'Country', 'Happiness Score','Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)','Freedom', 'Generosity', 'Trust (Government Corruption)']].copy()

yr_17
yr_2018 = pd.read_csv("/kaggle/input/world-happiness-report/2018.csv")

yr_2018.rename(columns={'Overall rank':'Happiness Rank','Country or region':'Country','Score':'Happiness Score','GDP per capita':'Economy (GDP per Capita)','Social support':'Family','Healthy life expectancy':'Health (Life Expectancy)','Freedom to make life choices':'Freedom','Perceptions of corruption':'Trust (Government Corruption)'},inplace=True)

yr_18 = yr_2018[['Happiness Rank', 'Country', 'Happiness Score','Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)','Freedom', 'Generosity', 'Trust (Government Corruption)']].copy()

yr_18
yr_2019 = pd.read_csv("/kaggle/input/world-happiness-report/2019.csv")

yr_2019.rename(columns={'Overall rank':'Happiness Rank','Country or region':'Country','Score':'Happiness Score','GDP per capita':'Economy (GDP per Capita)','Social support':'Family','Healthy life expectancy':'Health (Life Expectancy)','Freedom to make life choices':'Freedom','Perceptions of corruption':'Trust (Government Corruption)'},inplace=True)

yr_19 = yr_2019[['Happiness Rank', 'Country', 'Happiness Score','Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)','Freedom', 'Generosity', 'Trust (Government Corruption)']].copy()

yr_19
yr_2020 = pd.read_csv("/kaggle/input/world-happiness-report/2020.csv")

yr_2020['Happiness Rank'] = yr_2020['Ladder score'].rank(ascending=False)

yr_20_1 = yr_2020[['Country name','Regional indicator','Ladder score','Standard error of ladder score', 'upperwhisker', 'lowerwhisker','Explained by: Log GDP per capita', 'Explained by: Social support','Explained by: Healthy life expectancy','Explained by: Freedom to make life choices','Explained by: Generosity', 'Explained by: Perceptions of corruption','Dystopia + residual', 'Happiness Rank']].copy()

yr_20_1.rename(columns={'Country name':'Country','Regional indicator':'Region','Ladder score':'Happiness Score','Standard error of ladder score':'Standard Error', 'upperwhisker':'Upper Confidence Interval', 'lowerwhisker':'Lower Confidence Interval','Explained by: Log GDP per capita':'Economy (GDP per Capita)', 'Explained by: Social support':'Family','Explained by: Healthy life expectancy':'Health (Life Expectancy)','Explained by: Freedom to make life choices':'Freedom','Explained by: Generosity':'Generosity', 'Explained by: Perceptions of corruption':'Trust (Government Corruption)','Dystopia + residual':'Dystopia Residual',},inplace=True)

yr_20 = yr_20_1[['Happiness Rank', 'Country', 'Happiness Score','Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)','Freedom', 'Generosity', 'Trust (Government Corruption)']].copy()

yr_20
yr_15_rank = yr_15[['Country','Happiness Rank']][:10]

yr_15_rank.rename(columns={'Happiness Rank':'2015'},inplace=True)

yr_16_rank = yr_16[['Country','Happiness Rank']][:10]

yr_16_rank.rename(columns={'Happiness Rank':'2016'},inplace=True)

yr_17_rank = yr_17[['Country','Happiness Rank']][:10]

yr_17_rank.rename(columns={'Happiness Rank':'2017'},inplace=True)

yr_18_rank = yr_18[['Country','Happiness Rank']][:10]

yr_18_rank.rename(columns={'Happiness Rank':'2018'},inplace=True)

yr_19_rank = yr_19[['Country','Happiness Rank']][:10]

yr_19_rank.rename(columns={'Happiness Rank':'2019'},inplace=True)

yr_20_rank = yr_20[['Country','Happiness Rank']][:10]

yr_20_rank.rename(columns={'Happiness Rank':'2020'},inplace=True)

merge1 = pd.merge(yr_15_rank,yr_16_rank,on='Country',how='outer')

merge2 = pd.merge(merge1,yr_17_rank,on='Country',how='outer')

merge3 = pd.merge(merge2,yr_18_rank,on='Country',how='outer')

merge4 = pd.merge(merge3,yr_19_rank,on='Country',how='outer')

merge5 = pd.merge(merge4,yr_20_rank,on='Country',how='outer')

merge5.fillna(0,inplace=True)

merge5.rename(columns={'Country':''},inplace=True)

final_top10 = merge5.T

new_header = final_top10.iloc[0] #grab the first row for the header

final_top10 = final_top10[1:] #take the data less the header row

final_top10.columns = new_header 

top10 = final_top10.T

top10
sns.set()

fig = plt.figure(figsize = (20, 10))

ax = plt.gca()





#plt.plot( 'Region','Happyness2015', data=RegionHappyness, marker='', color='blue', linewidth=4, linestyle='dashed', label="2015")

top10.plot(kind='bar',ax=ax,colormap='rainbow')

plt.xlabel("Countries")

plt.ylabel("Rank")

plt.title("Ranks of Top 10 Happy Countries 2015-2020")

plt.xticks(rotation=90)

plt.show()
yr_15_rank2 = yr_15[['Country','Happiness Rank']][-10:]

yr_15_rank2.rename(columns={'Happiness Rank':'2015'},inplace=True)

yr_16_rank2 = yr_16[['Country','Happiness Rank']][-10:]

yr_16_rank2.rename(columns={'Happiness Rank':'2016'},inplace=True)

yr_17_rank2 = yr_17[['Country','Happiness Rank']][-10:]

yr_17_rank2.rename(columns={'Happiness Rank':'2017'},inplace=True)

yr_18_rank2 = yr_18[['Country','Happiness Rank']][-10:]

yr_18_rank2.rename(columns={'Happiness Rank':'2018'},inplace=True)

yr_19_rank2 = yr_19[['Country','Happiness Rank']][-10:]

yr_19_rank2.rename(columns={'Happiness Rank':'2019'},inplace=True)

yr_20_rank2 = yr_20[['Country','Happiness Rank']][-10:]

yr_20_rank2.rename(columns={'Happiness Rank':'2020'},inplace=True)

merge1_bot = pd.merge(yr_15_rank2,yr_16_rank2,on='Country',how='outer')

merge2_bot = pd.merge(merge1_bot,yr_17_rank2,on='Country',how='outer')

merge3_bot = pd.merge(merge2_bot,yr_18_rank2,on='Country',how='outer')

merge4_bot = pd.merge(merge3_bot,yr_19_rank2,on='Country',how='outer')

merge5_bot = pd.merge(merge4_bot,yr_20_rank2,on='Country',how='outer')

botranks = pd.DataFrame()

botranks['Country'] = merge5_bot['Country']

botranks['2015'] = merge5_bot['2015'].rank(ascending=False,na_option='keep')

botranks['2016'] = merge5_bot['2016'].rank(ascending=False,na_option='keep')

botranks['2017'] = merge5_bot['2017'].rank(ascending=False,na_option='keep')

botranks['2018'] = merge5_bot['2018'].rank(ascending=False,na_option='keep')

botranks['2019'] = merge5_bot['2019'].rank(ascending=False,na_option='keep')

botranks['2020'] = merge5_bot['2020'].rank(ascending=False,na_option='keep')

botranks.rename(columns={'Country':''},inplace=True)

bottom10 = botranks.T

new_header_bot = bottom10.iloc[0] #grab the first row for the header

bottom10 = bottom10[1:] #take the data less the header row

bottom10.columns = new_header_bot

bottom10.fillna(0,inplace=True)

bottom10
sns.set()

fig = plt.figure(figsize = (20, 10))

ax = plt.gca()





#plt.plot( 'Region','Happyness2015', data=RegionHappyness, marker='', color='blue', linewidth=4, linestyle='dashed', label="2015")

bottom10.T.plot(kind='bar',ax=ax,colormap='rainbow')

plt.xlabel("Countries")

plt.ylabel("Rank")

plt.title("Ranks of Bottom 10 Happy Countries 2015-2020 (Rank 1 being least happy)")

plt.xticks(rotation=90)

plt.show()
fig, axarr = plt.subplots(2, 3, figsize=(20, 10))

plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.2, wspace=1.5)

plt.title("Correlating factors - Overall")

sns.heatmap(yr_15.corr(),ax=axarr[0][0],cmap="mako",linewidths=.3)

axarr[0][0].set_title("2015", fontsize=12)



sns.heatmap(yr_16.corr(),ax=axarr[0][1],cmap="mako",linewidths=.3)

axarr[0][1].set_title("2016", fontsize=12)



sns.heatmap(yr_17.corr(),ax=axarr[0][2],cmap="mako",linewidths=.3)

axarr[0][2].set_title("2017", fontsize=12)



sns.heatmap(yr_18.corr(),ax=axarr[1][0],cmap="mako",linewidths=.3)

axarr[1][0].set_title("2018", fontsize=12)



sns.heatmap(yr_19.corr(),ax=axarr[1][1],cmap="mako",linewidths=.3)

axarr[1][1].set_title("2019", fontsize=12)



sns.heatmap(yr_20.corr(),ax=axarr[1][2],cmap="mako",linewidths=.3)

axarr[1][2].set_title("2020", fontsize=12)
fig, axarr = plt.subplots(2, 3, figsize=(20, 10))

plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.2, wspace=1.5)

sns.heatmap(yr_15[:10].corr(),ax=axarr[0][0],cmap="mako",linewidths=.3)

axarr[0][0].set_title("2015", fontsize=12)



sns.heatmap(yr_16[:10].corr(),ax=axarr[0][1],cmap="mako",linewidths=.3)

axarr[0][1].set_title("2016", fontsize=12)



sns.heatmap(yr_17[:10].corr(),ax=axarr[0][2],cmap="mako",linewidths=.3)

axarr[0][2].set_title("2017", fontsize=12)



sns.heatmap(yr_18[:10].corr(),ax=axarr[1][0],cmap="mako",linewidths=.3)

axarr[1][0].set_title("2018", fontsize=12)



sns.heatmap(yr_19[:10].corr(),ax=axarr[1][1],cmap="mako",linewidths=.3)

axarr[1][1].set_title("2019", fontsize=12)



sns.heatmap(yr_20[:10].corr(),ax=axarr[1][2],cmap="mako",linewidths=.3)

axarr[1][2].set_title("2020", fontsize=12)
fig, axarr = plt.subplots(2, 3, figsize=(20, 10))

plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.2, wspace=1.5)

sns.heatmap(yr_15[-10:].corr(),ax=axarr[0][0],cmap="mako",linewidths=.3)

axarr[0][0].set_title("2015", fontsize=12)



sns.heatmap(yr_16[-10:].corr(),ax=axarr[0][1],cmap="mako",linewidths=.3)

axarr[0][1].set_title("2016", fontsize=12)



sns.heatmap(yr_17[-10:].corr(),ax=axarr[0][2],cmap="mako",linewidths=.3)

axarr[0][2].set_title("2017", fontsize=12)



sns.heatmap(yr_18[-10:].corr(),ax=axarr[1][0],cmap="mako",linewidths=.3)

axarr[1][0].set_title("2018", fontsize=12)



sns.heatmap(yr_19[-10:].corr(),ax=axarr[1][1],cmap="mako",linewidths=.3)

axarr[1][1].set_title("2019", fontsize=12)



sns.heatmap(yr_20[-10:].corr(),ax=axarr[1][2],cmap="mako",linewidths=.3)

axarr[1][2].set_title("2020", fontsize=12)