# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# This notebook can contain fake data and this only for practice purpose



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # Data visualization

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/corona-virus-time-series-dataset/COVID-19/csse_covid_19_data/csse_covid_19_time_series/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Paths

dir_confirmed = r'/kaggle/input/corona-virus-time-series-dataset/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

dir_recovered = r'/kaggle/input/corona-virus-time-series-dataset/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv'

dir_deaths = r'/kaggle/input/corona-virus-time-series-dataset/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

# Data frame creation

df_confirmed = pd.read_csv(dir_confirmed)

df_recovered = pd.read_csv(dir_recovered )

df_deaths = pd.read_csv(dir_deaths)

#Columns renaming for convenience 

rename_dict = {'Country/Region':'Country', 'Province/State':'State'}

df_confirmed = df_confirmed.rename(rename_dict, axis = 'columns')

df_recovered = df_recovered.rename(rename_dict, axis = 'columns')

df_deaths = df_deaths.rename(rename_dict, axis = 'columns')
#Data sorting

def sortTopTen(df, typ, val_to_sort):

    df[f'Total_{typ}'] = df.iloc[:, -1:]

    df_new = df.iloc[:, [1,-1]].sort_values(by = [f'Total_{typ}'], ascending=False)

    return df_new[:val_to_sort]

#Data visualization

def dataVis(df, x_nme, y_nme):

    f,ax=plt.subplots(figsize=(20,6))

    sns.set(style='whitegrid')

    bar = sns.barplot(x = x_nme, y = y_nme, data = df, palette='rainbow_r',capsize=.5)

    bar.set_xlabel(x_nme)

    bar.set_ylabel(y_nme)

    plt.title(f'Showing result of {x_nme} in all countries')

    plt.show()
#Confirmed results with bar chart

df_confirmed_new = sortTopTen(df_confirmed, 'confirmed', 15)

dataVis(df_confirmed_new, 'Total_confirmed', 'Country')
#Death results with bar chart

df_deaths_new = sortTopTen(df_deaths, 'died', 10)

dataVis(df_deaths_new, 'Total_died', 'Country')
#Recovered results with bar chart

df_recovered_new = sortTopTen(df_recovered, 'recovered', 10)

dataVis(df_recovered_new, 'Total_recovered', 'Country')