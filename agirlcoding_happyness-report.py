# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pyplot import figure, show



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
happy2015=pd.read_csv('/kaggle/input/world-happiness-report/2015.csv')

happy2016=pd.read_csv('/kaggle/input/world-happiness-report/2016.csv')

happy2017=pd.read_csv('/kaggle/input/world-happiness-report/2017.csv')

happy2018=pd.read_csv('/kaggle/input/world-happiness-report/2018.csv')

happy2019=pd.read_csv('/kaggle/input/world-happiness-report/2019.csv')

happy2020=pd.read_csv('/kaggle/input/world-happiness-report/2020.csv')



Score2015=happy2015[['Country', 'Region','Happiness Score']]

Score2016=happy2016[['Country', 'Happiness Score']]

Score2017=happy2017[['Country', 'Happiness.Score']]

Score2018=happy2018[['Country or region', 'Score']]

Score2019=happy2019[['Country or region', 'Score']]

Score2020=happy2020[['Country name', 'Ladder score']]
Average2015=happy2015["Happiness Score"].mean()

Average2016=happy2016["Happiness Score"].mean()

Average2017=happy2017["Happiness.Score"].mean()

Average2018=happy2018["Score"].mean()

Average2019=happy2019["Score"].mean()

Average2020=happy2020["Ladder score"].mean()

print(Average2015,Average2016,Average2017,Average2018,Average2019,Average2020)
Score2015.rename(columns={'Happiness Score': 'Happyness2015'}, 

                 inplace=True)

Score2016.rename(columns={'Happiness Score': 'Happyness2016'}, 

                 inplace=True)

Score2017.rename(columns={'Happiness.Score': 'Happyness2017'}, 

                 inplace=True)

Score2018.rename(columns={'Country or region':'Country', 'Score': 'Happyness2018'}, 

                 inplace=True)

Score2019.rename(columns={'Country or region':'Country', 'Score': 'Happyness2019'}, 

                 inplace=True)

Score2020.rename(columns={'Country name':'Country', 'Ladder score': 'Happyness2020'}, 

                 inplace=True)
Score2015.sort_values(by=['Country'])

Score2016.sort_values(by=['Country'])

Score2017.sort_values(by=['Country'])

Score2018.sort_values(by=['Country'])

Score2019.sort_values(by=['Country'])

Score2020.sort_values(by=['Country'])
mergedStuff = pd.merge(Score2015, Score2016, on=['Country'], how='inner')

mergedStuff.head()



mergedStuff1 = pd.merge(Score2017, Score2018, on=['Country'], how='inner')

mergedStuff1.head()



mergedStuff2 = pd.merge(Score2019, Score2020, on=['Country'], how='inner')

mergedStuff2.head()



mergedStuff3 = pd.merge(mergedStuff, mergedStuff1, on=['Country'], how='inner')

mergedStuff3.head()



mergedStuff4 = pd.merge(mergedStuff3, mergedStuff2, on=['Country'], how='inner')

mergedStuff4.head()



RegionHappyness=mergedStuff4.groupby(['Region']).mean()

RegionHappyness.info

RegionHappyness.reset_index(inplace=True)

RegionHappyness.head()
fig = plt.figure(figsize = (20, 10))

plt.xticks(rotation=90)

plt.plot( 'Region','Happyness2015', data=RegionHappyness, marker='', color='blue', linewidth=4, linestyle='dashed', label="2015")

plt.plot( 'Region','Happyness2016', data=RegionHappyness, marker='', color='blueviolet', linewidth=4, linestyle='dashed', label="2016")

plt.plot( 'Region','Happyness2017', data=RegionHappyness, marker='', color='darkgreen', linewidth=4, linestyle='dashed', label="2017")

plt.plot( 'Region','Happyness2018', data=RegionHappyness, marker='', color='teal', linewidth=4, linestyle='dashed', label="2018")

plt.plot( 'Region','Happyness2019', data=RegionHappyness, marker='', color='dodgerblue', linewidth=4, linestyle='dashed', label="2019")

plt.plot( 'Region','Happyness2020', data=RegionHappyness, marker='', color='magenta', linewidth=4, linestyle='dashed', label="2020")

plt.legend()

fig = plt.figure(figsize = (20, 10))

(ax1, ax2, ax3), (ax4, ax5, ax6) = fig.subplots(2,3)

plt.xticks(rotation=90)

fig.subplots_adjust(hspace=1)



RegionHappyness.plot(kind='bar', x='Region', y='Happyness2015', ax=ax1, legend=False, color='Indigo')

ax1.yaxis.set_label_text("Happyness Score")

ax1.set_title("Happyness by Region in 2015", fontweight='bold', color = 'black', fontsize='12')



RegionHappyness.plot(kind='bar', x='Region', y='Happyness2016', ax=ax2, legend=False, color='darkorchid')

ax2.set_title("Happyness by Region in 2016", fontweight='bold', color = 'black', fontsize='12')



RegionHappyness.plot(kind='bar', x='Region', y='Happyness2017', ax=ax3, legend=False, color='darkmagenta')

ax3.set_title("Happyness by Region in 2017", fontweight='bold', color = 'black', fontsize='12')



RegionHappyness.plot(kind='bar', x='Region', y='Happyness2018', ax=ax4, legend=False, color='darkgreen')

ax4.yaxis.set_label_text("Happyness Score")

ax4.set_title("Happyness by Region in 2018", fontweight='bold', color = 'black', fontsize='12')



RegionHappyness.plot(kind='bar', x='Region', y='Happyness2019', ax=ax5, legend=False, color='forestgreen')

ax1.yaxis.set_label_text("Countries")

ax5.set_title("Happyness by Region in 2019", fontweight='bold', color = 'black', fontsize='12')



RegionHappyness.plot(kind='bar', x='Region', y='Happyness2020', ax=ax6, legend=False, color='lightseagreen')

ax1.yaxis.set_label_text("Happyness Score")

ax6.set_title("Happyness by Region in 2020", fontweight='bold', color = 'black', fontsize='12')



#top5countries per year

top2015=happy2015.head()

top2016=happy2016.head()

top2017=happy2017.head()

top2018=happy2018.head()

top2019=happy2019.head()

top2020=happy2020.head()
fig = plt.figure(figsize = (20, 10))

(ax1, ax2, ax3), (ax4, ax5, ax6) = fig.subplots(2,3)



top2015.sort_values('Happiness Score',inplace=True)

top2015.plot(kind='barh', x='Country', y='Happiness Score', ax=ax1, legend=False, cmap='plasma')

ax1.yaxis.set_label_text("Countries")

ax1.set_title("Happyness 2015", fontweight='bold', color = 'teal', fontsize='12')



top2016.sort_values('Happiness Score',inplace=True)

top2016.plot(kind='barh', x='Country', y='Happiness Score', ax=ax2, legend=False, cmap='plasma')

ax2.yaxis.set_label_text("")

ax2.set_title("Happyness 2016", fontweight='bold', color = 'teal', fontsize='12')

fig.subplots_adjust(wspace=0.5)



top2017.sort_values('Happiness.Score',inplace=True)

top2017.plot(kind='barh', x='Country', y='Happiness.Score', ax=ax3, legend=False, cmap='plasma')

ax3.yaxis.set_label_text("")

ax3.set_title("Happyness 2017", fontweight='bold', color = 'teal', fontsize='12')

fig.subplots_adjust(hspace=0.5)



top2018.sort_values('Score',inplace=True)

top2018.plot(kind='barh', x='Country or region', y='Score', ax=ax4, legend=False, cmap='plasma')

ax4.yaxis.set_label_text("Countries")

ax4.set_title("Happyness 2018", fontweight='bold', color = 'teal', fontsize='12')

fig.subplots_adjust(hspace=0.5)



top2019.sort_values('Score',inplace=True)

top2019.plot(kind='barh', x='Country or region', y='Score', ax=ax5, legend=False, cmap='plasma')

ax5.yaxis.set_label_text("")

ax5.set_title("Happyness 2019", fontweight='bold', color = 'teal', fontsize='12')

fig.subplots_adjust(hspace=0.5)



top2020.sort_values('Ladder score',inplace=True)

top2020.plot(kind='barh', x='Country name', y='Ladder score', ax=ax6, legend=False, cmap='plasma')

ax6.yaxis.set_label_text("")

ax6.set_title("Happyness 2020", fontweight='bold', color = 'teal', fontsize='12')

fig.subplots_adjust(hspace=0.5)
#let´s check which values affected happyness the most in different years.



fig = plt.figure(figsize = (20, 10))

(ax1, ax2, ax3), (ax4, ax5, ax6) = fig.subplots(2,3)



corr2015 = happy2015.corr()

corr2016 = happy2016.corr()

corr2017 = happy2017.corr()

corr2018 = happy2018.corr()

corr2019 = happy2019.corr()

corr2020 = happy2020.corr()







ax1 = sns.heatmap(corr2015, 

                 vmax = 1, 

                 vmin = 0,

                 square = True,  

                 cmap = "viridis",  ax=ax1)

fig.subplots_adjust(hspace=1, wspace=1)



ax2 = sns.heatmap(corr2016,  

                 vmax = 1, 

                 vmin = 0, 

                 square = True,  

                 cmap = "viridis",  ax=ax2)



ax3 = sns.heatmap(corr2017, 

                 vmax = 1, 

                 vmin = 0, 

                 square = True,  

                 cmap = "viridis",  ax=ax3)



ax4 = sns.heatmap(corr2018, 

                 vmax = 1, 

                 vmin = 0,

                square = True,

                 cmap = "viridis",  ax=ax4)



ax5 = sns.heatmap(corr2019, 

                 vmax = 1, 

                 vmin = 0,

                 square = True,  

                 cmap = "viridis",  ax=ax5)



ax6 = sns.heatmap(corr2020,              

                 vmax = 1, 

                 vmin = 0,

                 square = True,  

                 cmap = "viridis",  ax=ax6)

fig.subplots_adjust(hspace=1.5, wspace=1.5)

ax1.yaxis.set_label_text("Countries")

ax1.set_title("Happyness 2015", fontweight='bold', color = 'teal', fontsize='12')

ax2.set_title("Happyness 2016", fontweight='bold', color = 'teal', fontsize='12')

ax3.set_title("Happyness 2017", fontweight='bold', color = 'teal', fontsize='12')

ax4.set_title("Happyness 2018", fontweight='bold', color = 'teal', fontsize='12')

ax5.set_title("Happyness 2019", fontweight='bold', color = 'teal', fontsize='12')

ax6.set_title("Happyness 2020", fontweight='bold', color = 'teal', fontsize='12')
