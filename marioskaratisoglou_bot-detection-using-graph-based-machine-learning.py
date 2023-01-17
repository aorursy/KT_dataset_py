import pandas as pd

import networkx as nx

from tqdm import tqdm

import collections

from multiprocessing import Pool

import time

import itertools

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from keras.models import Sequential

from keras.layers import Dense
df = pd.read_csv('/kaggle/input/ctuscenario9/capture20110817.binetflow')

print('Dataset size:')

print(df.shape)

print(df.head())
df_bot = df[df['Label'].str.contains('Botnet')]

df_benign = df[~df['Label'].str.contains('Botnet')]

df_benign = df_benign.sample(n=df_bot.shape[0])



df_prunned = pd.concat([df_bot, df_benign])

print('Prunned dataframe shape:')

print(df_prunned.shape)
df = df_prunned[['SrcAddr', 'DstAddr', 'TotPkts']]

print(df.head())

print(df.shape)
#Finding the duplicate source and destination addresses and keeping a single pair with the total packets summed

duplicateRowsDF = df[df.duplicated(['SrcAddr', 'DstAddr'])]

duplicateRows = list(duplicateRowsDF.index.values)

df = df.drop(duplicateRows)



df_temp = pd.merge(df, duplicateRowsDF, how='inner', on=['SrcAddr', 'DstAddr'])



sum_column = df_temp['TotPkts_x'] + df_temp['TotPkts_y']

df_temp['TotPkts'] = sum_column

df = df_temp[['SrcAddr', 'DstAddr', 'TotPkts']]



#Creating graph where each ip corresponds to a unique node

dg=nx.DiGraph()



#Removing duplicate ip entries in order to have one node for each ip

srcAddrLst = list(df['SrcAddr'])

srcAddrLst.extend(list(df['DstAddr']))

ip_nodes = list(set(srcAddrLst))

dg.add_nodes_from(ip_nodes)



dict_nodes = collections.defaultdict(dict)



for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="{Creating network graph...}"):

    e = (row['SrcAddr'], row['DstAddr'], row['TotPkts'])

    

    if dg.has_edge(*e[:2]):

        edgeData = dg.get_edge_data(*e)

        weight = edgeData['weight']

        dg.add_weighted_edges_from([(row['SrcAddr'], row['DstAddr'], row['TotPkts'] + weight)])

        dict_nodes[row['SrcAddr']]['out-degree-weight'] += row['TotPkts']

        dict_nodes[row['DstAddr']]['in-degree-weight'] += row['TotPkts']

        dict_nodes[row['SrcAddr']]['out-degree'] += 1

        dict_nodes[row['DstAddr']]['in-degree'] += 1

    else:

        dg.add_weighted_edges_from([(row['SrcAddr'], row['DstAddr'], row['TotPkts'])])

        dict_nodes[row['SrcAddr']]['out-degree-weight'] = row['TotPkts']

        dict_nodes[row['SrcAddr']]['in-degree-weight'] = 0

        dict_nodes[row['DstAddr']]['in-degree-weight'] = row['TotPkts']

        dict_nodes[row['DstAddr']]['out-degree-weight'] = 0

        dict_nodes[row['SrcAddr']]['out-degree'] = 1

        dict_nodes[row['SrcAddr']]['in-degree'] = 0

        dict_nodes[row['DstAddr']]['in-degree'] = 1

        dict_nodes[row['DstAddr']]['out-degree'] = 0



print('Number of nodes: ' + str(nx.number_of_nodes(dg)))

print('Number of edges: ' + str(nx.number_of_edges(dg)))

print('Network graph created')
def chunks(l, n):

    """Divide a list of nodes `l` in `n` chunks"""

    l_c = iter(l)

    while 1:

        x = tuple(itertools.islice(l_c, n))

        if not x:

            return

        yield x





def _betmap(G_normalized_weight_sources_tuple):

    """Pool for multiprocess only accepts functions with one argument.

    This function uses a tuple as its only argument. We use a named tuple for

    python 3 compatibility, and then unpack it when we send it to

    `betweenness_centrality_source`

    """

    return nx.betweenness_centrality_source(*G_normalized_weight_sources_tuple)





def betweenness_centrality_parallel(G, processes=None):

    """Parallel betweenness centrality  function"""

    p = Pool(processes=processes)

    node_divisor = len(p._pool) * 2

    node_chunks = list(chunks(G.nodes(), int(G.order() / node_divisor)))

    num_chunks = len(node_chunks)

    bt_sc = p.map(_betmap,

                  zip([G] * num_chunks,

                      [True] * num_chunks,

                      [True] * num_chunks,

                      node_chunks))



    # Reduce the partial solutions

    bt_c = bt_sc[0]

    for bt in bt_sc[1:]:

        for n in bt:

            bt_c[n] += bt[n]

    return bt_c
#Calculate the betweeness centrality for all nodes

print('Calculating the betweeness centrality for all nodes:')

start = time.time()

dict_bc = betweenness_centrality_parallel(dg, 2)

#dict_bc = nx.betweenness_centrality(dg, weight='weight')

print("\t--Time: %.4F" % (time.time() - start))



#Calculate the clustering coefficient for all nodes

print('Calculating the clustering coefficient for all nodes...')

start = time.time()

dict_lcc = nx.clustering(dg, weight='weight')

print("\t--Time: %.4F" % (time.time() - start))



#Calculate the Alpha Centrality (Katz Centrality) for all nodes

print('Calculating the Alpha Centrality (Katz Centrality) for all nodes..')

start = time.time()

dict_ac = nx.algorithms.centrality.katz_centrality_numpy(dg, weight='weight')

print("\t--Time: %.4F" % (time.time() - start))

bot_lst = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204', '147.32.84.205', '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']



for k in dict_nodes:

    if k in bot_lst:

        dict_nodes[k]['bot'] = 1

    else:

        dict_nodes[k]['bot'] = 0



    dict_nodes[k]['bc'] = dict_bc[k]

    dict_nodes[k]['lcc'] = dict_lcc[k]

    dict_nodes[k]['ac'] = dict_ac[k]
#Feature Normalization

for node in tqdm(ip_nodes, desc="{Normalizing features...}"):  

    N = 0 #N is a counter for the total neighbors of each node with D=2

    s_in_degree = 0 #s is the total sum of all the features of the node's neighbors

    s_out_degree = 0

    s_in_degree_weight = 0

    s_out_degree_weight = 0

    s_bc = 0

    s_lcc = 0

    s_ac = 0

    

    for neighbor in dg.neighbors(node):

        s_in_degree += dict_nodes[neighbor]['in-degree'] 

        s_out_degree += dict_nodes[neighbor]['out-degree']

        s_in_degree_weight += dict_nodes[neighbor]['in-degree-weight']

        s_out_degree_weight += dict_nodes[neighbor]['out-degree-weight']

        s_bc += dict_bc[neighbor]

        s_lcc += dict_lcc[neighbor]

        s_ac += dict_ac[neighbor]

        N += 1

        

        for n in dg.neighbors(neighbor):

            s_in_degree += dict_nodes[neighbor]['in-degree'] 

            s_out_degree += dict_nodes[neighbor]['out-degree']

            s_in_degree_weight += dict_nodes[neighbor]['in-degree-weight']

            s_out_degree_weight += dict_nodes[neighbor]['out-degree-weight']

            s_bc += dict_bc[neighbor]

            s_lcc += dict_lcc[neighbor]

            s_ac += dict_ac[neighbor]

            N += 1

   

    if N != 0:

        if s_in_degree != 0:

            dict_nodes[node]['in-degree'] = dict_nodes[node]['in-degree'] / (s_in_degree/N)

        

        if s_out_degree != 0:

            dict_nodes[node]['out-degree'] = dict_nodes[node]['out-degree'] / (s_out_degree/N)

        

        if s_in_degree_weight != 0:

            dict_nodes[node]['in-degree-weight'] = dict_nodes[node]['in-degree-weight'] / (s_in_degree_weight/N)

        

        if s_out_degree_weight != 0:

            dict_nodes[node]['out-degree-weight'] = dict_nodes[node]['out-degree-weight'] / (s_out_degree_weight/N)

        

        if s_bc != 0:

            dict_nodes[node]['bc'] = dict_nodes[node]['bc'] / (s_bc/N)

        

        if s_lcc != 0:

            dict_nodes[node]['lcc'] = dict_nodes[node]['lcc'] / (s_lcc/N)

        

        if s_ac != 0:

            dict_nodes[node]['ac'] = dict_nodes[node]['ac'] / (s_ac/N)

        

        

print('Feature normalization complete')        

#Creating a dataframe from nodes' dictionary for easier computations and later use on machine learning algorithms

graph_df = pd.DataFrame.from_dict(dict_nodes, orient='index')
print(graph_df.head())
df = pd.read_csv('/kaggle/input/botnet-graph-dataset/botnet_graph_dataset.csv')

print(df.head())
X = df[['out-degree-weight', 'in-degree-weight', 'out-degree', 'in-degree', 'bc', 'lcc', 'ac']]

y = df['bot']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print('Performing Decision Tree Classification...')



# Create Decision Tree classifer object

dtc = DecisionTreeClassifier()



# Train Decision Tree Classifer

dtc = dtc.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = dtc.predict(X_test)



print('Decision Tree classification report:')

print(metrics.classification_report(y_test, y_pred))
print('Performing Logistic Regression...')

#ipt = input('Press enter to proceed')



logreg = LogisticRegression(max_iter=1000)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)



print('Logistic Regression classification report:')

print(metrics.classification_report(y_test, y_pred))
print('Performing SVM classification...')

clf = SVC(gamma='auto')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



print('Support Vector Machine classification report:')

print(metrics.classification_report(y_test, y_pred))
clf = RandomForestClassifier(n_estimators=200, max_depth=2, random_state=0, min_samples_split=4)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



print('Random Forest classification report:')

print(metrics.classification_report(y_test, y_pred))
model = Sequential()

model.add(Dense(12, input_dim=7, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



# compile the keras model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# fit the keras model on the dataset

model.fit(X_train, y_train, epochs=150, batch_size=10)



#evaluate the keras model

_, accuracy = model.evaluate(X, y)

print('Accuracy: %.2f' % (accuracy*100))