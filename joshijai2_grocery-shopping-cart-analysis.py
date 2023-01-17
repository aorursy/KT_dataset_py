# for basic operations
import numpy as np
import pandas as pd

# for visualizations
import matplotlib.pyplot as plt
import squarify
import seaborn as sns
from wordcloud import WordCloud
import networkx as nx

# for market basket analysis
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth, fpmax, fpcommon 

# reading the dataset

data = pd.read_csv('../input/market-basket-optimization/Market_Basket_Optimisation.csv', header = None)

# let's check the shape of the dataset
data.shape
# checking the head of the data

data.head()
# checkng the tail of the data

data.tail()
# checking the random entries in the data

data.sample(10)
# let's describe the dataset

data.describe()
plt.rcParams['figure.figsize'] = (15, 15)
wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(str(data[0]))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Most Popular Items',fontsize = 20)
plt.show()
# looking at the frequency of most popular items 

plt.rcParams['figure.figsize'] = (18, 7)
color = plt.cm.copper(np.linspace(0, 1, 40))
data[0].value_counts().head(40).plot.bar(color = color)
plt.title('frequency of most popular items', fontsize = 20)
plt.xticks(rotation = 90 )
plt.grid()
plt.show()
y = data[0].value_counts().head(50).to_frame()
y.index
# plotting a tree map

plt.rcParams['figure.figsize'] = (20, 20)
color = plt.cm.cool(np.linspace(0, 1, 50))
squarify.plot(sizes = y.values, label = y.index, alpha=.8, color = color)
plt.title('Tree Map for Popular Items')
plt.axis('off')
plt.show()
data['food'] = 'Food'
food = data.truncate(before = -1, after = 15)

food = nx.from_pandas_edgelist(food, source = 'food', target = 0, edge_attr = True)
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (20, 20)
pos = nx.spring_layout(food)
color = plt.cm.Wistia(np.linspace(0, 15, 1))
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
color = plt.cm.Blues(np.linspace(0, 15, 1))
nx.draw_networkx_nodes(secondchoice, pos, node_size = 15000, node_color = color)
nx.draw_networkx_edges(secondchoice, pos, width = 3, alpha = 0.6, edge_color = 'brown')
nx.draw_networkx_labels(secondchoice, pos, font_size = 20, font_family = 'sans-serif')
plt.axis('off')
plt.grid()
plt.title('Top 15 Second Choices', fontsize = 40)
plt.show()
data['thirdchoice'] = 'Third Choice'
secondchoice = data.truncate(before = -1, after = 10)
secondchoice = nx.from_pandas_edgelist(secondchoice, source = 'food', target = 2, edge_attr = True)
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (20, 20)
pos = nx.spring_layout(secondchoice)
color = plt.cm.Reds(np.linspace(0, 15, 1))
nx.draw_networkx_nodes(secondchoice, pos, node_size = 15000, node_color = color)
nx.draw_networkx_edges(secondchoice, pos, width = 3, alpha = 0.6, edge_color = 'pink')
nx.draw_networkx_labels(secondchoice, pos, font_size = 20, font_family = 'sans-serif')
plt.axis('off')
plt.grid()
plt.title('Top 10 Third Choices', fontsize = 40)
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

data = data.loc[:, list(y.index)]

# checking the shape
data.shape
# getting the head of the data

data.head()
frequent_itemsets = apriori(data, min_support = 0.01, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3).iloc[:,:-3]
rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))
rules.sort_values("confidence")
frequent_itemsets = apriori(data, min_support = 0.03, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.25).iloc[:,:-3]
rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))
rules.sort_values("confidence")
frequent_itemsets = apriori(data, min_support = 0.05, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2).iloc[:,:-3]
rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))
rules.sort_values("confidence")
frequent_itemsets = fpgrowth(data, min_support = 0.01, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3).iloc[:,:-3]
rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))
rules.sort_values("confidence")
frequent_itemsets = fpgrowth(data, min_support = 0.03, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.25).iloc[:,:-3]
rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))
rules.sort_values("confidence")
frequent_itemsets = fpgrowth(data, min_support = 0.05, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2).iloc[:,:-3]
rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))
rules.sort_values("confidence")
from timeit import repeat
testcases = [''' 
def fn(): 
    return apriori(data, min_support = 0.01, use_colnames=True)
''',
''' 
def fn(): 
    return fpgrowth(data, min_support = 0.01, use_colnames=True)
''']

res_apriori = repeat(stmt=testcases[0], repeat=5)
res_fpgrowth = repeat(stmt=testcases[1], repeat=5)

results1 = [res_apriori,res_fpgrowth]
testcases = [''' 
def fn(): 
    return apriori(data, min_support = 0.03, use_colnames=True)
''',
''' 
def fn(): 
    return fpgrowth(data, min_support = 0.03, use_colnames=True)
''']

res_apriori = repeat(stmt=testcases[0], repeat=5)
res_fpgrowth = repeat(stmt=testcases[1], repeat=5)

results3 = [res_apriori,res_fpgrowth]
testcases = [''' 
def fn(): 
    return apriori(data, min_support = 0.05, use_colnames=True)
''',
''' 
def fn(): 
    return fpgrowth(data, min_support = 0.05, use_colnames=True)
''']

res_apriori = repeat(stmt=testcases[0], repeat=5)
res_fpgrowth = repeat(stmt=testcases[1], repeat=5)

results5 = [res_apriori,res_fpgrowth]
plt.figure(figsize=(10,6))
plt.boxplot(results1, labels=["Apriori", "FP Growth"], showmeans=True)
plt.show()
plt.figure(figsize=(10,6))
plt.boxplot(results3, labels=["Apriori", "FP Growth"], showmeans=True)
plt.show()
plt.figure(figsize=(10,6))
plt.boxplot(results5, labels=["Apriori", "FP Growth"], showmeans=True)
plt.show()
plt.figure(figsize=(10,6))
sns.lineplot(["1%","3%","5%"],[np.mean(results1[0]),np.mean(results3[0]),np.mean(results5[0])], label="Apriori")
sns.lineplot(["1%","3%","5%"],[np.mean(results1[1]),np.mean(results3[1]),np.mean(results5[1])], label="FP Growth")

plt.show()
