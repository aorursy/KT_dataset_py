%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import geopandas as gpd



pd.options.display.float_format = '{:,.2f}'.format

sns.set(color_codes=True, font_scale=1.5)

sns.set_style("white")

plt.rcParams["figure.figsize"] = [12, 9]
honeyproduction_csv = "../input/honeyproduction.csv"
honeyproduction = pd.read_csv(honeyproduction_csv)

honeyproduction.sample(10)
honeyproduction['year'] = honeyproduction['year'].astype(np.str)
honeyproduction.info()
honeyproduction.groupby(by=['year'])['state'].count()
honeyproduction.describe().T
honeyproduction.describe(include=['O']).transpose()
def plot_stats(d, ax): 

    desc = d.describe()

    

    mean = desc["mean"]

    std = desc["std"]

    

    ax.axvline(mean, color = "r")

    ax.axvline(desc["25%"], color = "g")

    ax.axvline(desc["50%"], color = "g")

    ax.axvline(desc["75%"], color = "g")

    ax.axvline(mean - std, color = "r")

    ax.axvline(mean + std, color = "r")
def plot_dist(d, ax):

    plot_stats(d, ax)    

    sns.distplot(d, ax=ax, kde=False)
numeric_columns = ['yieldpercol','totalprod','stocks','priceperlb','prodvalue']

numeric_columns
nrows = len(numeric_columns)

fig, axs = plt.subplots(nrows=nrows, figsize = [14, nrows * 5])

fig.tight_layout(pad=4)

sns.despine()



for i,column in enumerate(numeric_columns):

    plot_dist(honeyproduction[column], axs[i])
for column in numeric_columns:

    plt.figure(figsize=(14,6))

    ax = sns.barplot(x='year', y = column, data=honeyproduction, color='b')
g = sns.PairGrid(honeyproduction[numeric_columns])

g = g.map(plt.scatter)