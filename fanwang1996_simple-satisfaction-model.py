# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def prepare_country_stats(oecd_bli, gdp_per_capita):

    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]

    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")

    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)

    gdp_per_capita.set_index("Country", inplace=True)

    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,

                                  left_index=True, right_index=True)

    full_country_stats.sort_values(by="GDP per capita", inplace=True)

    remove_indices = [0, 1, 6, 8, 33, 34, 35]

    keep_indices = list(set(range(36)) - set(remove_indices))

    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
# Code example

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import sklearn.linear_model



# Load the data

oecd_bli = pd.read_csv("/kaggle/input/oecd_bli_2015.csv", thousands=',')

gdp_per_capita = pd.read_csv("/kaggle/input/gdp_per_capita.csv",thousands=',',delimiter='\t',

                             encoding='latin1', na_values="n/a")
# Prepare the data

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)

X = np.c_[country_stats["GDP per capita"]]

y = np.c_[country_stats["Life satisfaction"]]
# Visualize the data

country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')

plt.show()
# Select a linear model

model = sklearn.linear_model.LinearRegression()
# Train the model

model.fit(X, y)
# Make a prediction for Cyprus

X_new = [[22587]]  # Cyprus' GDP per capita

print(model.predict(X_new)) # outputs [[ 5.96242338]]