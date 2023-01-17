# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
happiness2015 = pd.read_csv("../input/2015.csv", index_col=0)
print(happiness2015.columns)
happiness2016 = pd.read_csv("../input/2016.csv", index_col=0)
print(happiness2016.columns)
happiness2017 = pd.read_csv("../input/2017.csv", index_col=0)
print(happiness2017.columns)
# Check for missing values
columns_missing_values_2015 = [col for col in happiness2015.columns 
                                 if happiness2015[col].isnull().any()]
columns_missing_values_2016 = [col for col in happiness2016.columns 
                                 if happiness2016[col].isnull().any()]
columns_missing_values_2017 = [col for col in happiness2017.columns 
                                 if happiness2017[col].isnull().any()]
print(columns_missing_values_2015)
print(columns_missing_values_2016)
print(columns_missing_values_2017)
import seaborn as sns
import matplotlib.pyplot as plt


# 2015 HeatMap
heatmap2015 = sns.heatmap(
    happiness2015.loc[:, ['Happiness Score',
       'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
       'Freedom', 'Trust (Government Corruption)', 'Generosity',
       'Dystopia Residual']].corr(method='pearson'),
    cmap="YlGnBu",
    annot=True
)

plt.title("2015 HeatMap")
plt.show(heatmap2015)

# 2016 HeatMap
heatmap2016 = sns.heatmap(
    happiness2016.loc[:, [ 'Happiness Score',
       'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
       'Freedom', 'Trust (Government Corruption)', 'Generosity',
       'Dystopia Residual']].corr(method='pearson'),
    cmap="YlGnBu",
    annot=True
)

plt.title("2016 HeatMap")
plt.show(heatmap2016)


# 2017 HeatMap
heatmap2017 = sns.heatmap(
    happiness2017.loc[:, ['Happiness.Score',
       'Economy..GDP.per.Capita.', 'Family', 'Health..Life.Expectancy.',
       'Freedom', 'Trust..Government.Corruption.', 'Generosity',
       'Dystopia.Residual']].corr(method='pearson'),
    cmap="YlGnBu",
    annot=True
)


plt.title("2017 HeatMap")
plt.show(heatmap2017)
