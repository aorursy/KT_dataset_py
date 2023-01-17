

import pandas as pd # python's library for data manipulation and preprocessing

import numpy as np  # python's library for number crunching

import matplotlib            # python's library for visualisation

import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12

import seaborn as sns        # also python's library for visualisations

color = sns.color_palette()

sns.set_style('darkgrid')



import sklearn                #python's machine learning library

from sklearn.model_selection import train_test_split

#read.csv(file = "../input/titanic/train.csv", header=TRUE)

housing = pd.read_csv('../input/housing.csv')   # reading the data into a pandas dataframe

housing.describe()

housing.head()

#housing.shape

housing.shape

housing.describe()

housing.info()

housing.groupby(['ocean_proximity']).size()

hist = housing[['total_rooms']].hist(bins=3)

import plotly.express as px

fig = px.histogram(housing, x="total_rooms")

fig.show()

fig = px.histogram(housing, x="households")

fig.show()

housing1 = housing.iloc[0:20640, 0:9]

housing1.head()

housing1.shape

labels.shape

labels=housing1.corr()

type(housing1)

type(labels)

sns.heatmap(labels,annot=labels,cmap='RdYlGn')

sns.distplot(housing.median_income)

plt.show()

#,fmt="",cmap="RdYlgn",linewidths=0.3,ax=ax)

plt.show()
housing["median_income_category"] = np.ceil(housing["median_income"]/1.5)

housing.median_income_category

housing.median_income_category.value_counts().sort_index()

housing.median_income_category.where(housing["median_income_category"]<5,5.0,inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)



for train_index, test_index in split.split(housing, housing["median_income_category"]):

    strat_train_set = housing.iloc[train_index]

    strat_test_set = housing.iloc[test_index]

strat_test_set.head()    

def income_cat_props(data):

    return data['median_income_category'].value_counts() /len(data)



income_cat_props(housing)

income_cat_props(strat_test_set)

income_cat_props(strat_train_set)



income_cat_props(housing)





train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)



compare_props = pd.DataFrame({

    "Overall": income_cat_props(housing),

    "Stratified": income_cat_props(strat_test_set),

    "Random": income_cat_props(test_set),

}).sort_index()



compare_props



compare_props["Rand. %error"] = 100*compare_props["Random"]/compare_props["Overall"] - 100

compare_props["Strat. %error"] = 100*compare_props["Stratified"]/compare_props["Overall"] - 100

compare_props

np.random.seed(19680801)

# example data

mu = 28.64  # mean of distribution

sigma = 12.6  # standard deviation of distribution

x = mu + sigma * np.random.randn(20640)

num_bins = 50

fig, ax = plt.subplots()

# the histogram of the data

n, bins, patches = ax.hist(x, num_bins, density=1)



# add a 'best fit' line

y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *

     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

ax.plot(bins, y, '--')

ax.set_xlabel('Smarts')

ax.set_ylabel('Probability density')

ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel

fig.tight_layout()

plt.show()