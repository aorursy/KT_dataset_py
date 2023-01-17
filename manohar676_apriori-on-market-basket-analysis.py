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
# reading the dataset



data = pd.read_csv('../input/veggie-data/Market_Basket_Optimisation.csv', header = None)

data.head()
data.tail()
data.describe()
data.isnull().sum()
data.dtypes
plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.magma(np.linspace(0, 1, 40))

data[0].value_counts().head(40).plot.bar(color = color)

plt.title('frequency of most popular items', fontsize = 20)

plt.xticks(rotation = 90 )

plt.grid()

plt.show()
y = data[0].value_counts().head(50).to_frame()

y.index



# plotting a tree map



plt.rcParams['figure.figsize'] = (20, 20)

color = plt.cm.RdYlGn(np.linspace(0, 1, 50))

squarify.plot(sizes = y.values, label = y.index, alpha=.8, color = color)

plt.title('Tree Map for Popular Items')

plt.axis('off')

plt.show()
data['food'] = 'Food'

food = data.truncate(before = -1, after = 15)





import networkx as nx



food = nx.from_pandas_edgelist(food, source = 'food', target = 0, edge_attr = True)







import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (20, 20)

pos = nx.spring_layout(food)

color = plt.cm.autumn(np.linspace(0, 15, 1))

nx.draw_networkx_nodes(food, pos, node_size = 15000, node_color = color)

nx.draw_networkx_edges(food, pos, width = 3, alpha = 0.6, edge_color = 'black')

nx.draw_networkx_labels(food, pos, font_size = 20, font_family = 'sans-serif')

plt.axis('off')

plt.grid()

plt.title('Top 15 First Choices', fontsize = 40)

plt.show()
data['secondchoice'] = 'Second Choice'

secondchoice = data.truncate(before = -1, after = 15)

secondchoice = nx.from_pandas_edgelist(secondchoice, source = 'food', target = 1, edge_attr = True)







import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (20, 20)

pos = nx.spring_layout(secondchoice)

color = plt.cm.summer(np.linspace(0, 15, 1))

nx.draw_networkx_nodes(secondchoice, pos, node_size = 15000, node_color = color)

nx.draw_networkx_edges(secondchoice, pos, width = 3, alpha = 0.6, edge_color = 'Yellow')

nx.draw_networkx_labels(secondchoice, pos, font_size = 20, font_family = 'sans-serif')

plt.axis('off')

plt.grid()

plt.title('Top 15 Second Choices', fontsize = 40)

plt.show()
data['thirdchoice'] = 'Third Choice'

secondchoice = data.truncate(before = -1, after = 15)

secondchoice = nx.from_pandas_edgelist(secondchoice, source = 'food', target = 2, edge_attr = True)

import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (20, 20)

pos = nx.spring_layout(secondchoice)

color = plt.cm.Wistia(np.linspace(0, 15, 1))

nx.draw_networkx_nodes(secondchoice, pos, node_size = 15000, node_color = color)

nx.draw_networkx_edges(secondchoice, pos, width = 3, alpha = 0.6, edge_color = 'Yellow')

nx.draw_networkx_labels(secondchoice, pos, font_size = 20, font_family = 'white')

plt.axis('off')

plt.grid()

plt.title('Top 15 Third Choices', fontsize = 40)

plt.show()




# making each customers shopping items an identical list

trans = []

for i in range(0, 7501):

    trans.append([str(data.values[i,j]) for j in range(0, 20)])



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

data.shape
import warnings

warnings.filterwarnings('ignore')



# getting correlations for 121 items would be messy 

# so let's reduce the items from 121 to 50



data = data.loc[:, ['mineral water', 'burgers', 'turkey', 'chocolate', 'frozen vegetables', 'spaghetti',

                    'shrimp', 'grated cheese', 'eggs', 'cookies', 'french fries', 'herb & pepper', 'ground beef',

                    'tomatoes', 'milk', 'escalope', 'fresh tuna', 'red wine', 'ham', 'cake', 'green tea',

                    'whole wheat pasta', 'pancakes', 'soup', 'muffins', 'energy bar', 'olive oil', 'champagne', 

                    'avocado', 'pepper', 'butter', 'parmesan cheese', 'whole wheat rice', 'low fat yogurt', 

                    'chicken', 'vegetables mix', 'pickles', 'meatballs', 'frozen smoothie', 'yogurt cake']]



# checking the shape

data.shape
from mlxtend.frequent_patterns import apriori



#Now, let us return the items and itemsets with at least 5% support:

apriori(data, min_support = 0.01, use_colnames = True)
frequent_itemsets = apriori(data, min_support = 0.05, use_colnames=True)

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

frequent_itemsets
# getting th item sets with length = 2 and support more han 10%



frequent_itemsets[ (frequent_itemsets['length'] == 2) &

                   (frequent_itemsets['support'] >= 0.01) ]
# getting th item sets with length = 2 and support more han 10%



frequent_itemsets[ (frequent_itemsets['length'] == 1) &

                   (frequent_itemsets['support'] >= 0.01) ]
frequent_itemsets[ frequent_itemsets['itemsets'] == {'eggs', 'mineral water'} ]
frequent_itemsets[ frequent_itemsets['itemsets'] == {'mineral water'} ]
frequent_itemsets[ frequent_itemsets['itemsets'] == {'chicken'} ]
frequent_itemsets[ frequent_itemsets['itemsets'] == {'frozen vegetables'} ]
frequent_itemsets[ frequent_itemsets['itemsets'] == {'chocolate'} ]