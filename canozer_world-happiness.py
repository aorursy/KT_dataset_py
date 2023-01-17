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
y2015 = pd.read_csv("/kaggle/input/world-happiness/2015.csv")
y2016 = pd.read_csv("/kaggle/input/world-happiness/2016.csv")
y2017 = pd.read_csv("/kaggle/input/world-happiness/2017.csv")
y2018 = pd.read_csv("/kaggle/input/world-happiness/2018.csv")
y2019 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")
tr5 = y2015[y2015.Country == "Turkey"]
tr6 = y2016[y2016.Country == "Turkey"]
tr7 = y2017[y2017.Country == "Turkey"]
tr8 = y2018[y2018["Country or region"] == "Turkey"]
tr9 = y2019[y2019["Country or region"] == "Turkey"]
tr5.loc[:,"Year"] = "2015"
tr6.loc[:,"Year"] = "2016"
tr7.loc[:,"Year"] = "2017"
tr8.loc[:,"Year"] = "2018"
tr9.loc[:,"Year"] = "2019"
tr5.columns
tr6.columns
tr7.rename(columns = {'Health..Life.Expectancy.': 'Health (Life Expectancy)','Happiness.Rank':'Happiness Rank','Happiness.Score':'Happiness Score','Economy..GDP.per.Capita.':'Economy (GDP per Capita)',
                     'Trust..Government.Corruption.':'Trust (Government Corruption)','Dystopia.Residual':'Dystopia Residual',
                     'Whisker.low':'Lower Confidence Interval','Whisker.high':'Upper Confidence Interval'}, inplace = True)
tr7.columns
tr8.rename(columns = {'Healthy life expectancy': 'Health (Life Expectancy)','Overall rank':'Happiness Rank', 'Country or region':'Country', 'Score':'Happiness Score', 'GDP per capita':'Economy (GDP per Capita)',
       'Healthy life expectancy':'Health (Life Expectancy)',
       'Freedom to make life choices':'Freedom','Perceptions of corruption':'Trust (Government Corruption)'}, inplace = True)
tr8.columns
tr9.rename(columns = {'Healthy life expectancy': 'Health (Life Expectancy)','Overall rank':'Happiness Rank', 'Country or region':'Country', 'Score':'Happiness Score', 'GDP per capita':'Economy (GDP per Capita)',
       'Healthy life expectancy':'Health (Life Expectancy)',
       'Freedom to make life choices':'Freedom','Perceptions of corruption':'Trust (Government Corruption)'}, inplace = True)
tr9.columns
Tr = pd.concat([tr5,tr6,tr7,tr8,tr9], ignore_index = True)
Tr.info()
Tr.drop(columns = ["Region","Standard Error"],inplace = True)
Tr.columns
Tr_all = Tr[['Country','Year','Happiness Rank', 'Happiness Score','Economy (GDP per Capita)','Family', 'Health (Life Expectancy)','Freedom', 'Trust (Government Corruption)', 'Generosity',
   'Social support']]
Tr_all
import seaborn as sns
import matplotlib.pyplot as plt
y2019.head()
t = y2019
sns.lmplot(x = 'Score', y = "GDP per capita"  , data = t)
sns.lmplot(x = 'Score', y = 'Healthy life expectancy', data = t)
sns.lmplot(x = 'Score', y = 'Freedom to make life choices', data = t)
sns.lmplot(x = 'Score', y = 'Social support', data = t)
sns.lmplot(x = 'Score', y = 'Generosity', data = t)
sns.lmplot(x = 'Score', y = 'Perceptions of corruption', data = t)
