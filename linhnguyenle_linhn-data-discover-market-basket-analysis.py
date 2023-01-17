# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd 
data = pd.read_csv('../input/BreadBasket_DMS.csv')
data.head(5)
# data['Dates'] = data['Date']
# data = data.drop('Date', axis=1)
# data.head(5)
data.dtypes
data.shape
import datetime as dt
data['Date'] = pd.to_datetime(data['Date'])
# data['Date'] = data['Dates'].dt.date
# data.head(5)
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Hour'] = data['Time'].apply(lambda x:int(str(pd.to_datetime(x).round('H')).split(" ")[1].split(":")[0]))
data.head(5)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
data['Item'].value_counts().head(10).plot(ax = ax, kind='bar')
# drop None:
# Drop row of NONE items
data.drop(data[data['Item']=='NONE'].index,inplace=True)
plt.hist(data['Month'], bins=np.arange(data['Month'].min(), data['Month'].max()+1))
data.groupby('Hour')['Transaction'].nunique().plot(figsize=(8,5))
data.groupby('Day')['Transaction'].nunique().plot(figsize=(8,5))
#Busiest day of week
#Get total transaction for each date
byday=data.groupby('Date')['Transaction'].nunique().reset_index()
#Create dayofweek column
byday['DayofWeek'] = byday['Date'].apply(lambda x:pd.to_datetime(x).weekday())

#Plot average transactions per day
byday.groupby('DayofWeek')['Transaction'].mean().plot(figsize=(8,5))
plt.title("Average number of transactions per weekday")
plt.ylabel("# of Transactions")
fig=plt.xticks(np.arange(7),['Mon','Tue','Wed','Thur','Fri','Sat','Sun'])
# get all the items in each transaction 
transactions = []
for i in data['Transaction'].unique():
    itemlist = list(set(data[data['Transaction']==i]['Item']))
    if len(itemlist) > 0:
        transactions.append(itemlist)
transactions[:5]
len(transactions) == data['Transaction'].nunique()
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
rules.sort_values('confidence', ascending=False).head(10)
import random

support=rules.as_matrix(columns=['support'])
confidence=rules.as_matrix(columns=['confidence'])

for i in range (len(support)):
   support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
   confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
 
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()

def draw_graph(rules, rules_to_show):
  import networkx as nx  
  G1 = nx.DiGraph()
   
  color_map=[]
  N = 50
  colors = np.random.rand(N)    
  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']   
   
   
  for i in range (rules_to_show):      
    G1.add_nodes_from(["R"+str(i)])
    
     
    for a in rules.iloc[i]['antecedents']:
                
        G1.add_nodes_from([a])
        
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)
       
    for c in rules.iloc[i]['consequents']:
             
            G1.add_nodes_from([c])
            
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)
 
  for node in G1:
       found_a_string = False
       for item in strs: 
           if node==item:
                found_a_string = True
       if found_a_string:
            color_map.append('yellow')
       else:
            color_map.append('green')       
 
 
   
  edges = G1.edges()
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]
 
  pos = nx.spring_layout(G1, k=16, scale=1)
  nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
  for p in pos:  # raise text positions
           pos[p][1] += 0.07
  nx.draw_networkx_labels(G1, pos)
  plt.show()
draw_graph(rules, 20)
