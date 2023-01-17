import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np



# https://www.kaggle.com/medyasun/corona-virus-complete-dataset

#  https://cowid.netlify.com/data/new_cases.csv

filename = '../input/new_cases.csv'

data = pd.read_csv(filename)

data = data.fillna(0)
%matplotlib inline
data
def indexedAt100(col):

    # take a column and take the slice starting from 100

    # cumulative cases

    start = 0

    cumulative = 0

    i = 0

    for cell in col:

        cumulative = cumulative + cell

        # print(cumulative, i)

        if cumulative > 100:

            return col[i:].tolist()

        i = i + 1

    return []

# len(indexedAt100(data['United States']))
iData = []

countries = []

for column in data:

    if column != 'date' and column != 'World' and column != 'China':

        col = indexedAt100(data[column])

        if len(col) > 10:

            iData.append(col)

            countries.append(column)
plt.rcParams['figure.figsize'] = [15, 10]



# Add a 3 day moving average to visualize the trend

iDf = pd.DataFrame(iData, countries).transpose().rolling(window=3).mean()



plot = iDf.plot(title='New Cases After The 100th Case')

plot.set(xlabel="Day", ylabel="New Cases")