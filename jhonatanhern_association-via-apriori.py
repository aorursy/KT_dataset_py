!pip install squarify
# for basic operations

import numpy as np

import pandas as pd



# for visualizations

import matplotlib.pyplot as plt

import squarify

import seaborn as sns

plt.style.use('fivethirtyeight')



# for defining path

import os

print(os.listdir('../input/'))



# for market basket analysis

from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules

data = pd.read_csv('../input/mobile-parts-for-association/mobileStuff.csv',names = list(range(0,9)), header = None, error_bad_lines=False)

# the names parameter is necessary since pd.read_csv tends to infer number of columns from the first rows,

# which allows errors when the first rows have a limited number of lines 



# let's check the shape of the dataset

data.shape
# looking at the frequency of most popular items 



plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.copper(np.linspace(0, 1, 4))

data[0].value_counts().head(40).plot.bar(color = color)

plt.title('frequency of most popular items', fontsize = 20)

plt.xticks(rotation = 90 )

plt.grid()

plt.show()
# checking the head of the data



data.sample(10)
# let's describe the dataset



data.describe()
# making each customers shopping items an identical list

trans = []

for i in range(0, 25):

    trans.append([str(data.values[i,j]) for j in range(0, 7)])



# conveting it into an numpy array

trans = np.array(trans)



# checking the shape of the array

print(trans.shape)
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder



te = TransactionEncoder()

data = te.fit_transform(trans)

data = pd.DataFrame(data, columns = te.columns_)



# getting the shape of the data

print(data.head())

data.shape
import warnings

warnings.filterwarnings('ignore')

data = data.loc[:, ['Battery', 'Camera', 'Charger', 'HeadPhones', 'HomeButton', 'Lences', 'Processor', 'Screen', 'Speakers', 'Splitter']]



data.shape
# getting the head of the data



data.head()
from mlxtend.frequent_patterns import apriori



#Now, let us return the items and itemsets with at least 40% support:

frequent_itemsets = apriori(data, min_support = 0.40, use_colnames = True)

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

frequent_itemsets
frequent_itemsets[ (frequent_itemsets['length'] == 1) ]
frequent_itemsets[ (frequent_itemsets['length'] == 2) | (frequent_itemsets['length'] == 3) ]
frequent_itemsets[ (frequent_itemsets['length'] == 3) ]