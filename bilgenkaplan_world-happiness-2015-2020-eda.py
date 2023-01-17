# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

import scipy.stats as sp

import seaborn as sns

import pandas_profiling as pp
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



files=list()

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        full_dir = os.path.join(dirname, filename)

        files.append(full_dir)

files.sort()

files



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
all_df = {}

x=15

for file in files:

    df_name = "World Happiness Report 20{0}".format(x)

    all_df[df_name] = pd.read_csv(file)

    x += 1

    print(df_name, all_df[df_name].shape)

    display(all_df[df_name].head())
for df_name, df in all_df.items():

    print(df_name)

    print(df.isnull().sum() , "\n")
for df_name, df in all_df.items():

    print(df_name)

    print(df.columns , "\n")
columns_to_drop = ['Logged GDP per capita',

                   'Social support',

                   'Ladder score in Dystopia',

                   'Healthy life expectancy',

                   'Freedom to make life choices',

                   'Generosity',

                   'Perceptions of corruption'

                  ]



all_df["World Happiness Report 2020"] = all_df["World Happiness Report 2020"].drop(columns=columns_to_drop)
columns_name_to_change = {'Happiness.Rank': 'Happiness Rank', 

                          'Happiness.Score': 'Happiness Score',

                          'Economy..GDP.per.Capita.': 'Economy (GDP per Capita)',

                          'Health..Life.Expectancy.' : 'Health (Life Expectancy)',

                          'Trust..Government.Corruption.' : 'Trust (Government Corruption)',

                          'Dystopia.Residual' : 'Dystopia Residual',

                          'Score' : 'Happiness Score',

                          'GDP per capita' : 'Economy (GDP per Capita)',

                          'Healthy life expectancy' : 'Health (Life Expectancy)',

                          'Freedom' : 'Freedom to make life choices',

                          'Perceptions of corruption' : 'Trust (Government Corruption)',

                          'Overall rank' : 'Happiness Rank',

                          'Country or region' : 'Country',

                          'Country name' : 'Country',

                          'Regional indicator' : 'Region',

                          'Ladder score' : 'Happiness Score',

                          'Explained by: Log GDP per capita' : 'Economy (GDP per Capita)',

                          'Explained by: Healthy life expectancy' : 'Health (Life Expectancy)',

                          'Explained by: Social support' : 'Social Support',

                          'Explained by: Freedom to make life choices': 'Freedom to make life choices',

                          'Explained by: Generosity' : 'Generosity',

                          'Explained by: Perceptions of corruption' : 'Trust (Government Corruption)',

                          'Standard error of ladder score': 'Standard Error',

                          'upperwhisker': "Wisker High",

                          'Whisker.high' : "Wisker High",

                          'Whisker.low' : "Wisker Low",

                          'lowerwhisker' : "Wisker Low",

                          'Ladder score in Dystopia' : "Happiness Score in Dystopia",

                          'Dystopia + residual' : 'Dystopia Residual',

                         }



for _, df in all_df.items():

    df.rename(columns=columns_name_to_change, inplace=True)



print("COLUMNS AFTER RENAMING\n")

for df_name, df in all_df.items():

    print(df_name)

    print(df.columns , "\n")
all_df["World Happiness Report 2020"]['Happiness Rank'] = all_df["World Happiness Report 2020"]['Happiness Score'].rank(method='dense', ascending=False).astype(int)

all_df["World Happiness Report 2020"][['Happiness Score','Happiness Rank']] 
columns_corr = ['Happiness Score',

                'Economy (GDP per Capita)', 

                'Health (Life Expectancy)', 

                'Family',

                'Freedom to make life choices',

                'Generosity', 

                'Trust (Government Corruption)',

                'Generosity',

                'Dystopia Residual']
fig, axs = plt.subplots(6,1, figsize=(16,32))

fig.tight_layout(pad=15)

i=0





for df_name, df in all_df.items():

    axs[i].set_title(df_name)

    sns.heatmap(df.loc[:, df.columns.isin(columns_corr)].corr(), annot=True, ax=axs[i])

    i += 1
columns = ['Happiness Score',

           'Economy (GDP per Capita)', 

           'Health (Life Expectancy)', 

           'Freedom to make life choices',

           'Generosity', 

           'Trust (Government Corruption)']

sns.pairplot(all_df["World Happiness Report 2020"][columns])
for df_name, df in all_df.items():

    fig, ax = plt.subplots(1, 1, figsize = (10, 6))

    plt.title(df_name)

    sns.barplot(x = "Happiness Score", y = "Country", data=df.head(30))

    ax.set_ylabel('')
year=2015

top_num = 20

df_name_="World Happiness Report 2015"

init_count = all_df["World Happiness Report 2015"].shape[0]

happiness_rank_count= init_count

list_data_happiness_rank_15_20 = [all_df["World Happiness Report 2015"]["Happiness Rank"]]

columns_happiness_rank_15_20 = ["Happiness Rank"]





for df_name, df in all_df.items():

    if df.shape[0] > happiness_rank_count:

        happiness_rank_count = df.shape[0]

        df_name_ = df_name

    list_data_happiness_rank_15_20.append(df["Country"].loc[df["Happiness Rank"] <= top_num])

    columns_happiness_rank_15_20.append(str(year))

    year += 1



df_happiness_rank_15_20 = pd.concat(list_data_happiness_rank_15_20, keys=columns_happiness_rank_15_20, axis=1)



if happiness_rank_count != init_count:

    df_happiness_rank_15_20["Happiness Rank"] = all_df[df_name_]["Happiness Rank"]

    

df_happiness_rank_15_20.head(top_num).style.hide_index()
pp.ProfileReport(all_df["World Happiness Report 2020"])