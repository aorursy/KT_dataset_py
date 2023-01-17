# All Imports

import os

import sys

import networkx as nx

import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from tqdm import tqdm_notebook

from math import ceil



import warnings

warnings.filterwarnings('ignore')
# Loading Data from csv files

transactions = pd.read_csv("/kaggle/input/micro-amlsim-dataset/transactions.csv")

accounts = pd.read_csv("/kaggle/input/micro-amlsim-dataset/accounts.csv")

transactions = transactions.head(4000)
# Data preparation - categorical and scaling



def processing_dataframes():

    print("Dropping NaN columns")

    accounts.dropna(axis=1, how='all', inplace=True)

    print("Renaming columns wrt to Stellar Config")

    transactions.columns  = ['tran_id', 'source', 'target', 'tx_type', 'weight',

       'tran_timestamp', 'is_sar', 'alert_id']

    # Label encoder performing encoding for all objects

    print("Label encoding categorical features")

    le = LabelEncoder()

    for col in transactions.columns:

        if transactions[col].dtype == "O":

            transactions[col] = le.fit_transform(transactions[col].astype(str))   

    le = LabelEncoder()

    for col in accounts.columns:

        if accounts[col].dtype == "O":

            accounts[col] = le.fit_transform(accounts[col].astype(str))   

            

            

    scaler = MinMaxScaler()

    transactions['weight'] = scaler.fit_transform(transactions['weight'].values.reshape(-1,1))

    print('\n')

    print("--> Account df done!")

    display(accounts.head())

    print("--> Transaction df done!")

    display(transactions.head())    

processing_dataframes()
transactions.shape
# Contains GCN Code



# Helper functions

from sklearn.metrics import roc_auc_score

def one_hot_encode(y):

    mods = len(np.unique(y))

    y_enc = np.zeros((y.shape[0], mods))

    

    for i in range(y.shape[0]):

        y_enc[i, y[i]] = 1

    return y_enc



class GraphConvolutionNetwork():

    

    def __init__(self, node_dim=2, graph_dim=2, nb_classes=2, nmax=15, alpha=0.025):

        """

        Parameters of the model architecture

        

        """

        self.node_dim = node_dim

        self.graph_dim = graph_dim

        self.nb_classes = nb_classes

        self.nmax = nmax

        self.alpha = alpha

        

        self.build_model()

        

    def build_model(self):

        self.adjs = tf.placeholder(tf.float32, shape=[None, self.nmax, self.nmax])

        self.embeddings = tf.placeholder(tf.float32, shape=[None, self.nmax, self.node_dim])

        self.targets = tf.placeholder(tf.float32, shape=[None, self.nb_classes])

        

        A1 = tf.Variable(tf.random_normal([self.graph_dim, self.node_dim], seed=None))

        B1 = tf.Variable(tf.random_normal([self.graph_dim, self.node_dim]))

        W  = tf.Variable(tf.random_normal([self.graph_dim, self.nb_classes]))

        

        M1 = tf.einsum('adc,adb->abc', self.embeddings, self.adjs)

        H1 = tf.nn.relu(tf.tensordot(M1, A1, (2, 1)) + tf.tensordot(self.embeddings, B1, (2, 1)))

        G1 = tf.reduce_mean(H1, 1)

        

        Y_OUT = tf.matmul(G1,W)

        cost = tf.losses.softmax_cross_entropy(self.targets, Y_OUT)

        

        self.predictions = tf.argmax(Y_OUT, 1)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)

        self.train = optimizer.minimize(cost)

        

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        

    def fit(self, adj, embds, y, epochs=20, batch_size=10, shuffle=True):

        self.scores = []

        y_enc = one_hot_encode(y)

        minibatches = ceil(len(adj) / batch_size)

        

        j = 0

        for i in range(epochs):

            INDS = np.array(range(len(adj)))

            

            if shuffle:

                idx = np.random.permutation(y.shape[0]) 

                INDS = INDS[idx]

                

            mini = np.array_split(INDS, minibatches)

            

            for inds in mini:

                j+=1

                sys.stderr.write('\rEpoch: %d/%d' % (j, epochs*minibatches))

                sys.stderr.flush()

                self.sess.run(self.train, feed_dict={self.adjs:adj[inds], self.embeddings:embds[inds], 

                                                self.targets:y_enc[inds]})                

            self.scores.append(self.score(adj, embds, y))

            #

    def predict(self, adj, embds):

        return self.sess.run(self.predictions, feed_dict={self.adjs:adj, self.embeddings:embds})

    

    def score(self, adj, embds,y):

        y_ = self.predict(adj, embds)

        return 100*(y==y_).mean()



    def auc_score(self, adj, embds,y):

        from sklearn.metrics import roc_auc_score

        y_ = self.predict(adj, embds)

        return roc_auc_score(y,y_)

    

class MultiLaplacianGCN():

    

    def __init__(self, node_dim=2, graph_dim=[3,3], nb_classes=2, nmax=15, alpha=0.025):

        """

        Parameters of the model architecture

        

        """

        self.graph_dims = [node_dim] + graph_dim

        self.n_layers = len(graph_dim)

        self.nb_classes = nb_classes

        self.nmax = nmax

        self.alpha = alpha

        

        self.build_model()

        

    def build_model(self):

        self.adjs = tf.placeholder(tf.float32, shape=[None, self.nmax, self.nmax])

        self.targets = tf.placeholder(tf.float32, shape=[None, self.nb_classes])

        

        self.A = {i+1: tf.Variable(tf.random_normal([self.graph_dims[i+1], self.graph_dims[i]])) \

             for i in range(self.n_layers)}

        self.B = {i+1: tf.Variable(tf.random_normal([self.graph_dims[i+1], self.graph_dims[i]])) \

             for i in range(self.n_layers)}

        self.W  = tf.Variable(tf.random_normal([self.graph_dims[-1], self.nb_classes]))

        

        

        self.M, self.H, self.G = {}, {}, {}

        

        self.H[0] = tf.placeholder(tf.float32, shape=[None, self.nmax, self.graph_dims[0]])

        

        for i in range(1, self.n_layers+1):

        

            self.M[i] = tf.einsum('adc,adb->abc', self.H[i-1], self.adjs)

            self.H[i] = tf.nn.relu(tf.tensordot(self.M[i], self.A[i], (2, 1)) 

                                   + tf.tensordot(self.H[i-1], self.B[i], (2, 1)))

            self.G[i] = tf.reduce_mean(self.H[i], 1)

        

        Y_OUT = tf.matmul(self.G[self.n_layers], self.W)

        cost = tf.losses.softmax_cross_entropy(self.targets, Y_OUT)

        

        self.predictions = tf.argmax(Y_OUT, 1)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)

        self.train = optimizer.minimize(cost)

        

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        

    def fit(self, adj, embds, y, epochs=20, batch_size=10, shuffle=True):

        self.scores = []

        minibatches = ceil(len(adj) / batch_size)

        

        y_enc = one_hot_encode(y)

        

        j = 0

        for i in range(epochs):

            INDS = np.array(range(len(adj)))

            

            if shuffle:

                idx = np.random.permutation(y.shape[0]) 

                INDS = INDS[idx]

                

            mini = np.array_split(INDS, minibatches)

            

            for inds in mini:

                j+=1

                sys.stderr.write('\rEpoch: %d/%d' % (j, epochs*minibatches))

                sys.stderr.flush()

                self.sess.run(self.train, feed_dict={self.adjs:adj[inds], self.H[0]:embds[inds], 

                                                self.targets:y_enc[inds]})

                

            self.scores.append(self.score(adj, embds, y))

            

        

        

    def predict(self, adj, embds):

        return self.sess.run(self.predictions, feed_dict={self.adjs:adj, self.H[0]:embds})

    

    def score(self, adj, embds,y):

        y_ = self.predict(adj, embds)

        return 100*(y==y_).mean()

    

    def auc_score(self, adj, embds,y):

        from sklearn.metrics import roc_auc_score

        y_ = self.predict(adj, embds)

        return roc_auc_score(y,y_)
# Creating sub-graphs

graphs_aml = []

for idx, row in tqdm_notebook(transactions.iterrows(),total=transactions.shape[0]):

    source = row['source']

    tran_id = row['tran_id']

    df = transactions[(transactions['source']==source)]

    nodes_list = list(set(df['source'].tolist() + df['target'].tolist()))

    tmp_node = accounts[accounts['acct_id'].isin(nodes_list)].set_index('acct_id')

    graph = nx.from_pandas_edgelist(df, 'source', 'target', ['weight', 'tx_type'])

    graphs_aml.append(graph)

graphs = graphs_aml
nx.draw(graphs[993])
# Feature engineering source node with adjacent neighbour sub-graphs



def get_graph_features(G, all_embds, nmax=15):

    n = len(G.nodes())

    

    node2id = {node:i for i, node in enumerate(G.nodes())}

    id2node = {i:node for node,i in node2id.items()}



    adj = np.zeros((nmax,nmax))

    embds = np.zeros((nmax, all_embds.shape[1]))



    for i in G.nodes():

        embds[node2id[i]] = all_embds[i]

        for j in G.neighbors(i):

            adj[node2id[j],node2id[i]] = 1

    

    return adj, embds
EMBED_DIM = 5

NB_SAMPLES = transactions.shape[0]

VOCAB_SIZE = 325

MAX_LENGTH = 10

NB_CLASSES = 2

# PROBAS = [0.3, 0.4, 0.55]

# CENTERS  =[0.1, 0.15, 0.2]

SHARE = .50

GRAPH_DIM = 10





# Relation and Target data for feeding into Neural Network

graphs = graphs_aml

y = transactions['is_sar'].astype('uint')
# Generating random embeddings matrix based on Graph shapes // Should be equal to accounts nos.

embds = np.random.normal(size = (accounts.shape[0],EMBED_DIM))

embds.shape
# Creating features

Adjs, Ids = [], []

for graph in graphs:

    adj, embds_g = get_graph_features(graph, embds, nmax=VOCAB_SIZE)

    Adjs.append(adj)

    Ids.append(embds_g)
# Value counts for imbalance check

pd.DataFrame(y)['is_sar'].value_counts()
# Training data prep



ADJ = np.array(Adjs)

ID = np.array(Ids)



CUT = int(NB_SAMPLES * SHARE)

ADJ_train, y_train, ADJ_test, y_test = ADJ[:CUT], y[:CUT], ADJ[CUT:], y[CUT:]

ID_train, ID_test = ID[:CUT], ID[CUT:]
import tensorflow.compat.v1 as tf



tf.disable_v2_behavior()

# Accomodating change for tensorflow 1.x behaviour 



# Running GCN

model = GraphConvolutionNetwork(node_dim=EMBED_DIM, graph_dim=GRAPH_DIM, nb_classes=NB_CLASSES, 

             nmax=VOCAB_SIZE, alpha=0.025)

model.fit(ADJ_train, ID_train, y_train, epochs=10, batch_size=32)
# Train-test results

train_score = model.score(ADJ_train, ID_train, y_train)

test_score = model.score(ADJ_test, ID_test, y_test)

print(f"Training score : {train_score}")

print(f"Test score : {test_score}")
#AUC results



train_score = model.auc_score(ADJ_train, ID_train, y_train)

test_score = model.auc_score(ADJ_test, ID_test, y_test)

print(f"Training ROC AUC : {train_score}")

print(f"Test ROC AUC : {test_score}")
payload = {

    "source" : 92,

    "target" : 83,

    "weight" : 0.2,

    "tx_type": 3,

}

inference_df = transactions[['source','target','weight','tx_type']].copy()

inference_df.append(payload,ignore_index=True)

df = inference_df[(inference_df['source']==payload['source'])]

nodes_list = list(set(df['source'].tolist() + df['target'].tolist()))

tmp_node = accounts[accounts['acct_id'].isin(nodes_list)].set_index('acct_id')

graph = nx.from_pandas_edgelist(df, 'source', 'target', ['weight', 'tx_type'])

nx.draw(graph)
adj, embds_g = get_graph_features(graph, embds, nmax=VOCAB_SIZE)
adj = adj.reshape(1,325,325)

embds_g = embds_g.reshape(1,325,5)
model.predict(adj, embds_g)