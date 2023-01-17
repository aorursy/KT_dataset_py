# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

from scipy import stats



# File related

import zipfile



# Plotting

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



plt.rcParams['axes.labelsize'] = 20

plt.rcParams['axes.titlesize'] = 20

plt.rcParams['xtick.labelsize'] = 18

plt.rcParams['ytick.labelsize'] = 18

plt.rcParams['legend.fontsize'] = 14



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
with zipfile.ZipFile('../input/data/fandango.zip','r') as file: file.extractall('.')

    

print(check_output(["ls", "fandango"]).decode("utf8"))



dataframe = pd.read_csv('fandango/fandango_score_comparison.csv')

dataframe.head()
films_sorted = sorted(dataframe['FILM'])
dataframe.keys()
dataframe.rename(columns={'Metacritic_user_nom':'Metacritic_user_norm'}, inplace = True)

dataframe.keys()
dataframe.set_index('FILM')

dataframe.sort_values(by='FILM', ascending=True, inplace=True)

dataframe.reset_index(drop=True,inplace= True)

dataframe.head()
# Comparing rating distribution of fandango with IMDB

# From the histogram plot it is visible that fandango rating is negatively skewed.



fig, axes = plt.subplots(figsize=(8.,6.))

dataframe['Fandango_Stars'].plot.hist(alpha=0.5,bins=5,label='Fandango_Stars')

dataframe['IMDB_norm'].plot.hist(alpha=0.5,bins=10,label='IMDB_norm')

plt.legend()

# Comparing rating distribution of fandango with Rotten Tomatoes

# From the histogram plot it is visible that fandango rating is negatively skewed.

fig, axes = plt.subplots(figsize=(8.,6.))

dataframe['Fandango_Stars'].plot.hist(alpha=0.5,bins=5,label='Fandango_Stars')

dataframe['RT_norm'].plot.hist(alpha=0.5,bins=10,label='RT_norm')

plt.legend()
# Comparing rating distribution of fandango with Metacritic

# From the histogram plot it is visible that fandango rating is negatively skewed.

fig, axes = plt.subplots(figsize=(8.,6.))

dataframe['Fandango_Stars'].plot.hist(alpha=0.5,bins=5,label='Fandango_Stars')

dataframe['Metacritic_norm'].plot.hist(alpha=0.5,bins=10,label='Metacritic_norm')

plt.legend()
# Comparing through boxplots

fig, axes = plt.subplots()

rankings_lst = ['Fandango_Stars','RT_user_norm','RT_norm','IMDB_norm','Metacritic_user_norm','Metacritic_norm']

dataframe[rankings_lst].boxplot(vert=False)

axes.set_xlabel('Stars')

plt.show()

plt.close()
# Comparing rating and value 

fig, axes = plt.subplots()

dataframe[['Fandango_Stars', 'Fandango_Ratingvalue']].boxplot(vert=False)

axes.set_xlabel('Stars')

plt.show()

plt.close()


