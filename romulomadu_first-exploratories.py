# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

from unidecode import unidecode


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def bar(acumm_data):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    ax = sns.barplot(x=acumm_data.index, y=acumm_data.values, palette='tab20b', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()
def stackedbar(data, rotation=90):
    ax = data.loc[:,data.columns].plot.bar(stacked=True, figsize=(10,7))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    plt.tight_layout()
    plt.show()  
def lower_undescore_unicode(lst):
    return [unidecode(re.sub(' ','_',x.lower())) for x in lst]

df = pd.read_csv('../input/IHMStefanini_industrial_safety_and_health_database.csv', parse_dates=['Data'])
df.info()
df.head()
df.columns = lower_undescore_unicode(df.columns)
df.accident_level.unique()
df.potential_accident_level.unique()
convert_roman = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6}
df.accident_level = df.accident_level.replace(convert_roman)
df.potential_accident_level = df.potential_accident_level.replace(convert_roman)
df.head()
df.info()
accum_genre = df.genre.value_counts()
bar(accum_genre)
print(f'Percentage of Male: {round(100*accum_genre["Male"]/(accum_genre["Male"]+accum_genre["Female"]),2)}%')
print(f'Percentage of Female: {round(100*accum_genre["Female"]/(accum_genre["Male"]+accum_genre["Female"]),2)}%')
bar(df.industry_sector.value_counts())
# remove Others and print bar
bar(df.risco_critico[~df['risco_critico'].isin(['Others'])].value_counts())
bar(df.employee_ou_terceiro.value_counts())

freq_matrix = pd.crosstab(df.accident_level, df.potential_accident_level)
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(freq_matrix, square=True, cmap='Purples', ax=ax, annot=True)

df.set_index('data')['local'].resample('W').count().plot()
plt.show()
df['weekday'] = df.data.dt.dayofweek+1
df.weekday.value_counts().sort_index().plot()
plt.show()
freq_matrix = pd.crosstab(df.genre, df.accident_level)
stackedbar(freq_matrix.T)
freq_matrix = pd.crosstab(df.industry_sector, df.accident_level)
stackedbar(freq_matrix.T)
freq_matrix1 = pd.crosstab(df.local, df.accident_level)
freq_matrix2 = pd.crosstab(df.local, df.potential_accident_level)
fig, ax = plt.subplots(figsize=(20,10),ncols=2)
ax[0] = sns.heatmap(freq_matrix1, square=True, cmap='Purples', annot=True, ax=ax[0])
ax[1] = sns.heatmap(freq_matrix2, square=True, cmap='Purples', annot=True, ax=ax[1])
ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=0)
ax[1].set_yticklabels(ax[1].get_yticklabels(), rotation=0)

plt.show()

df.genre.unique()
df.genre = df.genre.replace({'Male': 1, 'Female': 0})
