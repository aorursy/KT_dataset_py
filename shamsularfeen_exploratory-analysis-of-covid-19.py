# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import matplotlib as mpl

import seaborn as sns

from geopy.geocoders import Nominatim

color = sns.color_palette()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

%matplotlib inline

# Any results you write to the current directory are saved as output.



from subprocess import check_output

# Any results you write to the current directory are saved as output.



df = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv", encoding = "ISO-8859-1", parse_dates=["Date"])

unknown_count = df.isna().sum().drop_duplicates()

unknown_count[unknown_count>0]

# Japan vs South Korea vs Italy vs Pakistan

# Create figure and plot space

fig, ax = plt.subplots(figsize=(20, 8))



# Add x-axis and y-axis

ax.plot(df.loc[df['Country/Region'] == 'Afghanistan','Date'],

        df.loc[df['Country/Region'] == 'Afghanistan', 'ConfirmedCases'],

        label = 'Afghanistan',

        color='purple')

ax.plot(df.loc[df['Country/Region'] == 'India', 'Date'],

        df.loc[df['Country/Region'] == 'India', 'ConfirmedCases'],

        label = 'India',

        color='orange')

ax.plot(df.loc[df['Country/Region'] == 'Pakistan','Date'],

        df.loc[df['Country/Region'] == 'Pakistan', 'ConfirmedCases'],

        label = 'Pakistan',

        color='green')



# Set title and labels for axes

ax.set(xlabel="Date",

       ylabel="ConfirmedCases", 

       title="Afghanistan vs India vs Pakistan")

plt.xticks(rotation=90)

plt.legend(loc='upper left')

plt.show()