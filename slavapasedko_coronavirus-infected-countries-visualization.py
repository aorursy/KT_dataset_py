# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/corona-virus-report/novel corona virus situation report who.csv')
df.head(10)
def plot_growth(dates, country):

    plt.rcParams["figure.figsize"] = (20,10)

    plt.plot_date(dates, country)

    plt.xlabel('Date', fontsize=18)

    plt.ylabel('Infected People', fontsize=16)
countries = df.drop('Date', axis=1)

for country in countries.columns:

    plot_growth(df['Date'], df[country])



for country in countries.columns:

    fig = px.pie(df, values=country, names='Date', title=f'Count Infected in {country}')

    fig.show()