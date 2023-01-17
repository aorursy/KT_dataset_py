import pandas as pd

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

import networkx as nx

from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import apriori,association_rules

import matplotlib.pyplot as plt

plt.style.use('default')
data = pd.read_csv("../input/market-basket-optimization/Market_Basket_Optimisation.csv", header=None)

data.shape
data.head()
data.describe()
data[1]
# 1. Gather All Items of Each Transactions into Numpy Array

transaction = []

for i in range(0, data.shape[0]):

    for j in range(0, data.shape[1]):

        transaction.append(data.values[i,j])



transaction = np.array(transaction)



# 2. Transform Them a Pandas DataFrame

df = pd.DataFrame(transaction, columns=["items"]) 

df["incident_count"] = 1 # Put 1 to Each Item For Making Countable Table, to be able to perform Group By



# 3. Delete NaN Items from Dataset

indexNames = df[df['items'] == "nan" ].index

df.drop(indexNames , inplace=True)



# 4. Final Step: Make a New Appropriate Pandas DataFrame for Visualizations  

df_table = df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()



# 5. Initial Visualizations

df_table.head(10).style.background_gradient(cmap='Blues')
df_table["all"] = "all" # to have a same origin



fig = px.treemap(df_table.head(30), path=['all', "items"], values='incident_count',

                  color=df_table["incident_count"].head(30), hover_data=['items'],

                  color_continuous_scale='Blues',

                  )

fig.show()
# Transform Every Transaction to Seperate List & Gather Them into Numpy Array

# By Doing So, We Will Be Able To Iterate Through Array of Transactions



transaction = []

for i in range(data.shape[0]):

    transaction.append([str(data.values[i,j]) for j in range(data.shape[1])])

    

transaction = np.array(transaction)



# Create a DataFrame In Order To Check Status of Top20 Items



top20 = df_table["items"].head(20).values

array = []

df_top20_multiple_record_check = pd.DataFrame(columns=top20)



for i in range(0, len(top20)):

    array = []

    for j in range(0,transaction.shape[0]):

        array.append(np.count_nonzero(transaction[j]==top20[i]))

        if len(array) == len(data):

            df_top20_multiple_record_check[top20[i]] = array

        else:

            continue

            



df_top20_multiple_record_check.head(10)

df_top20_multiple_record_check.describe()
# 1. Gather Only First Choice of Each Transactions into Numpy Array

# Similar Pattern to Above, Only Change is the Column Number "0" in Append Function

transaction = []

for i in range(0, data.shape[0]):

    transaction.append(data.values[i,0])



transaction = np.array(transaction)



# 2. Transform Them a Pandas DataFrame

df_first = pd.DataFrame(transaction, columns=["items"])

df_first["incident_count"] = 1



# 3. Delete NaN Items from Dataset

indexNames = df_first[df_first['items'] == "nan" ].index

df_first.drop(indexNames , inplace=True)



# 4. Final Step: Make a New Appropriate Pandas DataFrame for Visualizations  

df_table_first = df_first.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()

df_table_first["food"] = "food"

df_table_first = df_table_first.truncate(before=-1, after=15) # Fist 15 Choice

import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (20, 20)

first_choice = nx.from_pandas_edgelist(df_table_first, source = 'food', target = "items", edge_attr = True)

pos = nx.spring_layout(first_choice)

nx.draw_networkx_nodes(first_choice, pos, node_size = 12500, node_color = "lavender")

nx.draw_networkx_edges(first_choice, pos, width = 3, alpha = 0.6, edge_color = 'black')

nx.draw_networkx_labels(first_choice, pos, font_size = 18, font_family = 'sans-serif')

plt.axis('off')

plt.grid()

plt.title('Top 15 First Choices', fontsize = 25)

plt.show()
# 1. Gather Only Second Choice of Each Transaction into Numpy Array



transaction = []

for i in range(0, data.shape[0]):

    transaction.append(data.values[i,1])



transaction = np.array(transaction)



# 2. Transform Them a Pandas DataFrame

df_second = pd.DataFrame(transaction, columns=["items"]) 

df_second["incident_count"] = 1



# 3. Delete NaN Items from Dataset

indexNames = df_second[df_second['items'] == "nan" ].index

df_second.drop(indexNames , inplace=True)



# 4. Final Step: Make a New Appropriate Pandas DataFrame for Visualizations  

df_table_second = df_second.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()

df_table_second["food"] = "food"

df_table_second = df_table_second.truncate(before=-1, after=15) # Fist 15 Choice

import warnings

warnings.filterwarnings('ignore')



second_choice = nx.from_pandas_edgelist(df_table_second, source = 'food', target = "items", edge_attr = True)

pos = nx.spring_layout(second_choice)

nx.draw_networkx_nodes(second_choice, pos, node_size = 12500, node_color = "honeydew")

nx.draw_networkx_edges(second_choice, pos, width = 3, alpha = 0.6, edge_color = 'black')

nx.draw_networkx_labels(second_choice, pos, font_size = 18, font_family = 'sans-serif')

plt.rcParams['figure.figsize'] = (20, 20)

plt.axis('off')

plt.grid()

plt.title('Top 15 Second Choices', fontsize = 25)

plt.show()
# 1. Gather Only Third Choice of Each Transaction into Numpy Array

## For Column "2"

transaction = []

for i in range(0, data.shape[0]):

    transaction.append(data.values[i,2])



transaction = np.array(transaction)



# 2. Transform Them a Pandas DataFrame

df_third = pd.DataFrame(transaction, columns=["items"]) # Transaction Item Name

df_third["incident_count"] = 1 # Put 1 to Each Item For Making Countable Table, Group By Will Be Done Later On



# 3. Delete NaN Items from Dataset

indexNames = df_third[df_third['items'] == "nan" ].index

df_third.drop(indexNames , inplace=True)



# 4. Final Step: Make a New Appropriate Pandas DataFrame for Visualizations  

df_table_third = df_third.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()

df_table_third["food"] = "food"

df_table_third = df_table_third.truncate(before=-1, after=15) # Fist 15 Choice

fig = go.Figure(data=[go.Bar(x=df_table_third["items"], y=df_table_third["incident_count"],

            hovertext=df_table_third["items"], text=df_table_third["incident_count"], textposition="outside")])



fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.65)

fig.update_layout(title_text="Customers' Third Choices", template="plotly_dark")

fig.show()
# Transform Every Transaction to Seperate List & Gather Them into Numpy Array



transaction = []

for i in range(data.shape[0]):

    transaction.append([str(data.values[i,j]) for j in range(data.shape[1])])

    

transaction = np.array(transaction)

transaction
te = TransactionEncoder()

te_ary = te.fit(transaction).transform(transaction)

dataset = pd.DataFrame(te_ary, columns=te.columns_)

dataset
dataset.shape
first50 = df_table["items"].head(50).values # Select Top50

dataset = dataset.loc[:,first50] # Extract Top50

dataset
dataset.columns

# We extracted first 50 column successfully.
# Convert dataset into 1-0 encoding



def encode_units(x):

    if x == False:

        return 0 

    if x == True:

        return 1

    

dataset = dataset.applymap(encode_units)

dataset.head(10)
# Extracting the most frequest itemsets via Mlxtend.

# The length column has been added to increase ease of filtering.



frequent_itemsets = apriori(dataset, min_support=0.01, use_colnames=True)

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

frequent_itemsets
frequent_itemsets[ (frequent_itemsets['length'] == 2) &

                   (frequent_itemsets['support'] >= 0.05) ]
frequent_itemsets[ (frequent_itemsets['length'] == 3) ].head()
# We can create our rules by defining metric and its threshold.



# For a start, 

#      We set our metric as "Lift" to define whether antecedents & consequents are dependent our not.

#      Treshold is selected as "1.2" since it is required to have lift scores above than 1 if there is dependency.



rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))

rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))

rules.sort_values("lift",ascending=False)
# Sort values based on confidence



rules.sort_values("confidence",ascending=False)
rules[~rules["consequents"].str.contains("mineral water", regex=False) & 

      ~rules["antecedents"].str.contains("mineral water", regex=False)].sort_values("confidence", ascending=False).head(10)
rules[rules["antecedents"].str.contains("ground beef", regex=False) & rules["antecedents_length"] == 1].sort_values("confidence", ascending=False).head(10)