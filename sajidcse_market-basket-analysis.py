#Data manipulation libraries

import pandas as pd

import numpy as np



#Visualizations

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("dark")

import squarify

import matplotlib



#for market basket analysis (using apriori)

from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules



#for preprocessing

from mlxtend.preprocessing import TransactionEncoder





#to print all the interactive output without resorting to print, not only the last result.

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
data = pd.read_csv('../input/market.csv',header=None)
#shape

data.shape
#head of data



data.head()





#tail of data



data.tail()
#converting into required format of TransactionEncoder()

trans=[]

for i in range(0,7501):

    trans.append([str(data.values[i,j]) for j in range(0,20)])



trans=np.array(trans)



print(trans.shape)
### Using TransactionEncoder



t=TransactionEncoder()

data=t.fit_transform(trans)

data=pd.DataFrame(data,columns=t.columns_,dtype=int)



data.shape
##here we also find nan as one of the columns so lets drop that column



data.drop('nan',axis=1,inplace=True)
#now lets check shape

data.shape



#lets verify whether nan is present in columns

'nan' in data.columns

#so its proved that nan is not in columns 
data.head()
##Lets consider the top 20 items purchased freequently

r=data.sum(axis=0).sort_values(ascending=False)[:20]

#altering the figsize

plt.figure(figsize=(20,10))

s=sns.barplot(x=r.index,y=r.values)

s.set_xticklabels(s.get_xticklabels(), rotation=90)
# create a color palette, mapped to these values

my_values=r.values

cmap = matplotlib.cm.Blues

mini=min(my_values)

maxi=max(my_values)

norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)

colors = [cmap(norm(value)) for value in my_values]





#treemap of top 20 frequent items

plt.figure(figsize=(10,10))

squarify.plot(sizes=r.values, label=r.index, alpha=.7,color=colors)

plt.title("Tree map of top 20 items")

plt.axis('off')

#let us return items and ietmsets with atleast 5% support:

freq_items=apriori(data,min_support=0.05,use_colnames=True)
freq_items
#Now let's generate association rules



res=association_rules(freq_items,metric="lift",min_threshold=1.3)
res
frequent_itemsets = apriori(data, min_support = 0.05, use_colnames=True)

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

frequent_itemsets
# getting th item sets with length = 2 and support more han 10%



frequent_itemsets[ (frequent_itemsets['length'] == 2) &

                   (frequent_itemsets['support'] >= 0.01) ]
# getting th item sets with length = 2 and support more han 10%



frequent_itemsets[ (frequent_itemsets['length'] == 1) &

                   (frequent_itemsets['support'] >= 0.01) ]
#Importing Libraries



from mlxtend.frequent_patterns import fpgrowth
#running the fpgrowth algorithm

res=fpgrowth(data,min_support=0.05,use_colnames=True)
res
res=association_rules(res,metric="lift",min_threshold=1)
res
import time

l=[0.01,0.02,0.03,0.04,0.05]

t=[]

for i in l:

    t1=time.time()

    apriori(data,min_support=i,use_colnames=True)

    t2=time.time()

    t.append((t2-t1)*1000)
l=[0.01,0.02,0.03,0.04,0.05]

f=[]

for i in l:

    t1=time.time()

    fpgrowth(data,min_support=i,use_colnames=True)

    t2=time.time()

    f.append((t2-t1)*1000)
sns.lineplot(x=l,y=f,label="fpgrowth")

sns.lineplot(x=l,y=t,label="apriori")

plt.xlabel("Min_support Threshold")

plt.ylabel("Run Time in ms")