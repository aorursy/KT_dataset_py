# load packages
import csv
import math
import time
import umap
import heapq
#!pip install hdbscan
#import hdbscan
import numpy as np
import scipy as sc
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import urllib.request as urllib
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from sklearn.manifold import TSNE
import sklearn.cluster as cluster
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
# load data
url = 'https://sixtusdakurah.com/projects/liger/stim_sparse_dfp.csv'
stim_sparse = pd.read_csv(url, sep=",", header=0)
stim_sparse = stim_sparse.rename(columns={'Unnamed: 0': 'gene'})
# check dimensions and get brief view
stim_sparse.head()
print("Shape of Stim: ", stim_sparse.shape)
ctr_E_df = stim_sparse.iloc[0:200, 0:201]
# convert to long format
ctr_E_df_long = pd.melt(ctr_E_df, id_vars=['gene'],var_name='cell', value_name='expression')
ctr_E_df.head()
ctr_E_df_long.head()
(ctr_E_df_long == 0).astype(int).sum(axis=0)
print("Unique genes: ", len((ctr_E_df_long['gene'].astype("category").cat.codes).drop_duplicates()))#.sort_values()
print("Unique cells: ",len((ctr_E_df_long['cell'].astype("category").cat.codes).drop_duplicates()))
# process data set
def process_dataset(df):

    # Convert cells names into numerical IDs
    df['cell_id'] = df['cell'].astype("category").cat.codes
    df['gene_id'] = df['gene'].astype("category").cat.codes

    
    gene_lookup = df[['gene_id', 'gene']].drop_duplicates()
    gene_lookup['gene_id'] = gene_lookup.gene_id.astype(str)

    # Grab the columns we need in the order we need them.
    df = df[['cell_id', 'gene_id', 'expression']]

    
    df_train, df_test = train_test_split(df) # 80 20

    
    cells = list(np.sort(df.cell_id.unique()))
    genes = list(np.sort(df.gene_id.unique()))

    
    rows = df_train.cell_id.astype(int)
    cols = df_train.gene_id.astype(int)

    values = list(df_train.expression)

    # Get all user ids and item ids.
    cids = np.array(rows.tolist())
    gids = np.array(cols.tolist())

    # Sample 100 negative interactions for each cell in our test data
    df_neg = ''#get_negatives(cids, gids, genes, df_test)

    return cids, gids, df_train, df_test, df_neg, cells, genes, gene_lookup, values
# sample a couple of negatives for each positive label
def get_negatives(cids, gids, genes, df_test):
    
    negativeList = []
    test_c = df_test['cell_id'].values.tolist()
    test_g = df_test['gene_id'].values.tolist()

    test_expression_ids = list(zip(test_c, test_g))
    zipped = set(zip(cids, gids))
    #print(len(genes))

    for (c, g) in test_expression_ids:
        negatives = []
        negatives.append((c, g))
        for t in range(10):# increase for better accuracy
            j = np.random.randint(len(genes)) # Get random gene id.
            while (c, j) in zipped: # Check if there is an interaction
                j = np.random.randint(len(genes)) # If yes, generate a new gene id
            negatives.append(j) # Once a negative interaction is found we add it.
            #print("J value is", j)
            #print(negatives)
        negativeList.append(negatives)

    df_neg = pd.DataFrame(negativeList)

    return df_neg
# mask the first gene to be used for testing
def mask_first(x):
    result = np.ones_like(x)
    result[0] = 0
    
    return result

# split into train and test set   
def train_test_split(df):

    df_test = df.copy(deep=True)
    df_train = df.copy(deep=True)

    df_test = df_test.groupby(['cell_id']).first()
    df_test['cell_id'] = df_test.index
    df_test = df_test[['cell_id', 'gene_id', 'expression']]
    df_test.index.name = None

    mask = df.groupby(['cell_id'])['cell_id'].transform(mask_first).astype(bool)
    df_train = df.loc[mask]

    return df_train, df_test

# combine mask and train test split
def get_train_instances():
    
    cell_input, gene_input, labels = [],[],[]
    zipped = set(zip(cids, gids))
    #progress = tqdm(total=len(zipped))
    #tracker = 0
    for (c, g) in zip(cids, gids):
        # Add our positive interaction
        cell_input.append(c)
        gene_input.append(g)
        labels.append((df_train[(df_train.cell_id==c)&(df_train.gene_id==g)]).expression.values[0])
        #labels.append(1)
        #print("gene value: ", g, " cell value: ", c)

        # Sample a number of random negative interactions
        for t in range(num_neg):
            j = np.random.randint(len(genes))
            #j = j if j!=32 else np.random.randint(len(genes)) # chainging to more than 1 
            while (c, j) in zipped:
                j = np.random.randint(len(genes))
                #j = rv+1 if rv==0 else rv
            if j!=df_test.gene_id[0]:
              cell_input.append(c)
              gene_input.append(j)
              #print("gene value: ", j, " cell value: ", c)
              #labels.append(0)
              labels.append((df_train[(df_train.cell_id==c)&(df_train.gene_id==j)]).expression.values[0])
        #progress.update(1)
        #progress.set_description('Sampled Training Instance' + str(tracker+1)) 
        #tracker+=1
    #progress.close()
    return cell_input, gene_input, labels

# for faster training
def random_mini_batches(C, G, L, mini_batch_size=20):

    mini_batches = []

    shuffled_C, shuffled_G, shuffled_L = shuffle(C, G, L, random_state=0)

    num_complete_batches = int(math.floor(len(C)/mini_batch_size))
    for k in range(0, num_complete_batches):
        mini_batch_C = shuffled_C[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_G = shuffled_G[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_L = shuffled_L[k * mini_batch_size : k * mini_batch_size + mini_batch_size]

        mini_batch = (mini_batch_C, mini_batch_G, mini_batch_L)
        mini_batches.append(mini_batch)

    if len(C) % mini_batch_size != 0:
        mini_batch_C = shuffled_C[num_complete_batches * mini_batch_size: len(C)]
        mini_batch_G = shuffled_G[num_complete_batches * mini_batch_size: len(C)]
        mini_batch_L = shuffled_L[num_complete_batches * mini_batch_size: len(C)]

        mini_batch = (mini_batch_C, mini_batch_G, mini_batch_L)
        mini_batches.append(mini_batch)

    return mini_batches
# evaluation
def get_hits(k_ranked, holdout):
    for gene in k_ranked:
        if gene == holdout:
            return 1
    return 0

def eval_rating(idx, test_expression, test_negatives, K):
    # test_expression = test_expression_ids
    map_gene_score = {}

    
    genes = test_negatives[idx]

    
    cell_idx = test_expression[idx][0]

    
    holdout = test_expression[idx][1]
    print("Holdout: ", holdout)

    
    genes.append(holdout)

    
    predict_cell = np.full(len(genes), cell_idx, dtype='int32').reshape(-1,1)
    print("Predict cell: ", predict_cell)
    np_genes = np.array(genes).reshape(-1,1)
    print("Genes: ", genes)

    
    predictions = session.run([output_layer], feed_dict={cell: predict_cell, gene: np_genes})
    print("Predictions: ", predictions)
    
    predictions = predictions[0].flatten().tolist()

    
    for i in range(len(genes)):
        current_gene = genes[i]
        map_gene_score[current_gene] = predictions[i]

    
    k_ranked = heapq.nlargest(K, map_gene_score, key=map_gene_score.get)
    print("K Ranked: ", k_ranked)

       
    hits = get_hits(k_ranked, holdout)

    return hits

# 
def evaluate(df_neg, K=10):

    hits = []

    test_c = df_test['cell_id'].values.tolist()
    test_g = df_test['gene_id'].values.tolist()

    test_expression_ids = list(zip(test_c, test_g))

    df_neg = df_neg.drop(df_neg.columns[0], axis=1)
    test_negatives = df_neg.values.tolist()
    #len(test_expression)-2
    for idx in range(len(test_expression_ids)):
        # For each idx, call eval_one_rating
        hitrate = eval_rating(idx, test_expression_ids, test_negatives, K)
        hits.append(hitrate)

    return hits
def make_recommendation(cell_ids=None, gene_ids = None, top=None):
    # make recommendations for all
    df = pd.DataFrame()
    df["Gene"] = (ctr_E_df_long.gene_id).unique()
    for cell_idx in np.sort((ctr_E_df_long.cell_id).unique()):
        # get the genes for the given cell
        genes = ((ctr_E_df_long[ctr_E_df_long.cell_id==cell_idx]).gene_id).values
        cell_ls = [cell_idx]# * len(genes)
        # create a full gene prediction for a cell
        predict_cell = np.full(len(genes), cell_idx, dtype='int32').reshape(-1,1)
        np_genes = np.array(genes).reshape(-1,1)
        
        #run with the given session
        predictions = session.run([output_layer], feed_dict={cell: predict_cell, gene: np_genes})
        #print("Predictions: ", predictions)

        predictions = predictions[0].flatten().tolist()
        df[cell_idx] = predictions
 
    return df

# try basic k-means clustering
def kMeans_clustering(k=10):
  kmeans_labels = cluster.KMeans(n_clusters=k).fit_predict(mlp_df)
  standard_embedding = umap.UMAP(random_state=42).fit_transform(mlp_df)
  newdf = pd.DataFrame(standard_embedding, columns = ["x1", "x2"])
  newdf["cluster"] = kmeans_labels
  # make the plot
  size = 80
  plt.figure(figsize=(16,10))
  sc = plt.scatter(newdf['x1'], newdf['x2'], s=size, c=newdf['cluster'], edgecolors='none')

  lp = lambda i: plt.plot([],color=sc.cmap(sc.norm(i)), ms=np.sqrt(size), mec="none",
                          label="Cluster {:g}".format(i), ls="", marker="o")[0]
  handles = [lp(i) for i in np.unique(newdf["cluster"])]
  plt.legend(handles=handles)
  plt.xlabel("Umap 1")
  plt.ylabel("Umap 2")
  plt.show()
cids, gids, df_train, df_test, df_neg, cells, genes, gene_lookup, values = process_dataset(ctr_E_df_long)
print("Unique train genes: ", len((df_train['gene_id']).drop_duplicates()))
print("Unique train cells: ", len((df_train['cell_id']).drop_duplicates()))
print("Unique test genes: ", len((df_test['gene_id']).drop_duplicates()))
print("Unique test cells: ",len((df_test['cell_id']).drop_duplicates()))
#(df_train[(df_train.cell_id==33)&(df_train.gene_id==32)]).expression.values[0]
df_test.head()
# define learning parameters
num_neg = 4
epochs = 20
batch_size = 20
learning_rate = 0.001
# get train instances
cell_input, gene_input, labels = cids, gids, values #get_train_instances()
#sum(labels)
graph = tf.Graph() # tensorflow term for building a pipeline

with graph.as_default():

    # define input placeholders for cell, gene and count=label.
    cell = tf.placeholder(tf.int32, shape=(None, 1))
    gene = tf.placeholder(tf.int32, shape=(None, 1))
    label = tf.placeholder(tf.int32, shape=(None, 1))

    # cell feature embedding
    c_var = tf.Variable(tf.random_normal([len(cells), 20], stddev=0.05), name='cell_embedding')
    cell_embedding = tf.nn.embedding_lookup(c_var, cell) # for each cell id, return a vector of length 20

    # gene feature embedding
    g_var = tf.Variable(tf.random_normal([len(genes), 20], stddev=0.05), name='gene_embedding')
    gene_embedding = tf.nn.embedding_lookup(g_var, gene) # for each gene id, return a vector of length 20

    # Flatten our cell and gene embeddings.
    cell_embedding = tf.keras.layers.Flatten()(cell_embedding)
    gene_embedding = tf.keras.layers.Flatten()(gene_embedding)

    # concatenate the two embedding vectors
    concatenated = tf.keras.layers.concatenate([cell_embedding, gene_embedding])

    
    # add a first dropout layer.
    dropout = tf.keras.layers.Dropout(0.2)(concatenated)

    # add four hidden layers along with batch
    # normalization and dropouts. use relu as the activation function.
    layer_1 = tf.keras.layers.Dense(64, activation='relu', name='layer1')(dropout)
    batch_norm1 = tf.keras.layers.BatchNormalization(name='batch_norm1')(layer_1)
    dropout1 = tf.keras.layers.Dropout(0.2, name='dropout1')(batch_norm1)

    layer_2 = tf.keras.layers.Dense(32, activation='relu', name='layer2')(layer_1)
    batch_norm2 = tf.keras.layers.BatchNormalization(name='batch_norm1')(layer_2)
    dropout2 = tf.keras.layers.Dropout(0.2, name='dropout1')(batch_norm2)

    layer_3 = tf.keras.layers.Dense(16, activation='relu', name='layer3')(layer_2)
    layer_4 = tf.keras.layers.Dense(8, activation='linear', name='layer4')(layer_3) # make linear

    # final single neuron output layer.
    output_layer = tf.keras.layers.Dense(1,
            kernel_initializer="lecun_uniform",
            name='output_layer')(layer_4)

    # our loss function as mse.
    labels = tf.cast(label, tf.float32)
    tf.print(labels)
    logits = output_layer # still leave as logits but values are not logits
    tf.print(logits)
    loss = tf.reduce_mean(tf.square(tf.subtract(
                labels,
                logits)))

    # train using the Adam optimizer to minimize loss.
    opt = tf.train.AdamOptimizer(learning_rate = learning_rate)
    step = opt.minimize(loss)

    # initialize all tensorflow variables.
    init = tf.global_variables_initializer()

session = tf.Session(config=None, graph=graph)
session.run(init)
for epoch in range(epochs):

    # Get our training input.
    cell_input, gene_input, labels = get_train_instances()

    # Generate a list of minibatches.
    minibatches = random_mini_batches(cell_input, gene_input, labels)

    # This has noting to do with tensorflow but gives
    # us a nice progress bar for the training
    progress = tqdm(total=len(minibatches))

    # Loop over each batch and feed our cells, genes and labels
    # into our graph. 
    for minibatch in minibatches:
        feed_dict = {cell: np.array(minibatch[0]).reshape(-1,1),
                    gene: np.array(minibatch[1]).reshape(-1,1),
                    label: np.array(minibatch[2]).reshape(-1,1)}
   
        # Execute the graph.
        _, l = session.run([step, loss], feed_dict)

        # Update the progress
        progress.update(1)
        progress.set_description('Epoch: %d - Loss: %.8f' % (epoch+1, l))

    progress.close()


# Calculate top@K    
hits = evaluate(df_neg)
print(np.array(hits).mean())
mlp_df = make_recommendation()
mlp_df.drop('Gene', axis=1, inplace=True)
mlp_df.head()
# try basic clustering
kMeans_clustering()
graph = tf.Graph()
latent_features = 20 # same as the paper specification

with graph.as_default():

    cell = tf.placeholder(tf.int32, shape=(None, 1))
    gene = tf.placeholder(tf.int32, shape=(None, 1))
    label = tf.placeholder(tf.int32, shape=(None, 1))

    
    c_var = tf.Variable(tf.random_normal([len(cells), latent_features],
                                         stddev=0.05), name='cell_embedding')
    cell_embedding = tf.nn.embedding_lookup(c_var, cell)


    g_var = tf.Variable(tf.random_normal([len(genes), latent_features],
                                         stddev=0.05), name='gene_embedding')
    gene_embedding = tf.nn.embedding_lookup(g_var, gene)
    
    # flatten the embeddings 
    cell_embedding = tf.keras.layers.Flatten()(cell_embedding)
    gene_embedding = tf.keras.layers.Flatten()(gene_embedding)

    # multiplying our cell and gene latent space vectors together 
    prediction_matrix = tf.multiply(cell_embedding, gene_embedding)

    
    output_layer = tf.keras.layers.Dense(1, 
            kernel_initializer="lecun_uniform",
            name='output_layer')(prediction_matrix)

    # loss function as a mse. 
    labels = tf.cast(label, tf.float32)
    loss = tf.reduce_mean(tf.square(tf.subtract(
                labels,
                output_layer)))
    
    # using the Adam optimizer to minimize loss.
    opt = tf.train.AdamOptimizer(learning_rate = learning_rate)
    step = opt.minimize(loss)

    # initialize all tensorflow variables.
    init = tf.global_variables_initializer()

session = tf.Session(config=None, graph=graph)
session.run(init)
for epoch in range(epochs):

    # Get our training input.
    cell_input, gene_input, labels = get_train_instances()

    # Generate a list of minibatches.
    minibatches = random_mini_batches(cell_input, gene_input, labels)

    # This has noting to do with tensorflow but gives
    # us a nice progress bar for the training
    progress = tqdm(total=len(minibatches))

    # Loop over each batch and feed our cells, genes and labels
    # into our graph. 
    for minibatch in minibatches:
        feed_dict = {cell: np.array(minibatch[0]).reshape(-1,1),
                    gene: np.array(minibatch[1]).reshape(-1,1),
                    label: np.array(minibatch[2]).reshape(-1,1)}
   
        # Execute the graph.
        _, l = session.run([step, loss], feed_dict)

        # Update the progress
        progress.update(1)
        progress.set_description('Epoch: %d - Loss: %.3f' % (epoch+1, l))

    progress.close()
mlp_df = make_recommendation()
mlp_df.drop('Gene', axis=1, inplace=True)
mlp_df.head()
# try basic clustering
kMeans_clustering()
graph = tf.Graph()

with graph.as_default():

    cell = tf.placeholder(tf.int32, shape=(None, 1))
    gene = tf.placeholder(tf.int32, shape=(None, 1))
    label = tf.placeholder(tf.int32, shape=(None, 1))

    
    mlp_c_var = tf.Variable(tf.random_normal([len(cells), 20], stddev=0.05),
            name='mlp_cell_embedding')
    mlp_cell_embedding = tf.nn.embedding_lookup(mlp_c_var, cell)

    
    mlp_g_var = tf.Variable(tf.random_normal([len(genes), 20], stddev=0.05),
            name='mlp_gene_embedding')
    mlp_gene_embedding = tf.nn.embedding_lookup(mlp_g_var, gene)

    
    gmf_c_var = tf.Variable(tf.random_normal([len(cells), latent_features],
        stddev=0.05), name='gmf_cell_embedding')
    gmf_cell_embedding = tf.nn.embedding_lookup(gmf_c_var, cell)

    # gene embedding for GMF
    gmf_g_var = tf.Variable(tf.random_normal([len(genes), latent_features],
        stddev=0.05), name='gmf_item_embedding')
    gmf_gene_embedding = tf.nn.embedding_lookup(gmf_g_var, gene)

    # flatten gmf embedding
    gmf_cell_embed = tf.keras.layers.Flatten()(gmf_cell_embedding)
    gmf_gene_embed = tf.keras.layers.Flatten()(gmf_gene_embedding)
    gmf_matrix = tf.multiply(gmf_cell_embed, gmf_gene_embed)

    # flatten mlp embedding
    mlp_cell_embed = tf.keras.layers.Flatten()(mlp_cell_embedding)
    mlp_gene_embed = tf.keras.layers.Flatten()(mlp_gene_embedding)
    mlp_concat = tf.keras.layers.concatenate([mlp_cell_embed, mlp_gene_embed])

    mlp_dropout = tf.keras.layers.Dropout(0.2)(mlp_concat)

    mlp_layer_1 = tf.keras.layers.Dense(64, activation='relu', name='layer1')(mlp_dropout)
    mlp_batch_norm1 = tf.keras.layers.BatchNormalization(name='batch_norm1')(mlp_layer_1)
    mlp_dropout1 = tf.keras.layers.Dropout(0.2, name='dropout1')(mlp_batch_norm1)

    mlp_layer_2 = tf.keras.layers.Dense(32, activation='relu', name='layer2')(mlp_dropout1)
    mlp_batch_norm2 = tf.keras.layers.BatchNormalization(name='batch_norm1')(mlp_layer_2)
    mlp_dropout2 = tf.keras.layers.Dropout(0.2, name='dropout1')(mlp_batch_norm2)

    mlp_layer_3 = tf.keras.layers.Dense(16, activation='relu', name='layer3')(mlp_dropout2)
    mlp_layer_4 = tf.keras.layers.Dense(8, activation='linear', name='layer4')(mlp_layer_3)

    # We merge the two networks together
    merged_vector = tf.keras.layers.concatenate([gmf_matrix, mlp_layer_4])

    # Our final single neuron output layer. 
    output_layer = tf.keras.layers.Dense(1,
            kernel_initializer="lecun_uniform",
            name='output_layer')(merged_vector)

    # Our loss function as mse. 
    labels = tf.cast(label, tf.float32)
    loss = tf.reduce_mean(tf.square(tf.subtract(
                labels,
                output_layer)))

    # Train using the Adam optimizer to minimize our loss.
    opt = tf.train.AdamOptimizer(learning_rate = learning_rate)
    step = opt.minimize(loss)

    # Initialize all tensorflow variables.
    init = tf.global_variables_initializer()


session = tf.Session(config=None, graph=graph)
session.run(init)
for epoch in range(epochs):

    # Get our training input.
    cell_input, gene_input, labels = get_train_instances()

    # Generate a list of minibatches.
    minibatches = random_mini_batches(cell_input, gene_input, labels)

    # This has noting to do with tensorflow but gives
    # us a nice progress bar for the training
    progress = tqdm(total=len(minibatches))

    # Loop over each batch and feed our cells, genes and labels
    # into our graph. 
    for minibatch in minibatches:
        feed_dict = {cell: np.array(minibatch[0]).reshape(-1,1),
                    gene: np.array(minibatch[1]).reshape(-1,1),
                    label: np.array(minibatch[2]).reshape(-1,1)}
   
        # Execute the graph.
        _, l = session.run([step, loss], feed_dict)

        # Update the progress
        progress.update(1)
        progress.set_description('Epoch: %d - Loss: %.3f' % (epoch+1, l))

    progress.close()


# Calculate top@K    
hits = evaluate(df_neg)
print(np.array(hits).mean())
mlp_df = make_recommendation()
#mlp_df.drop('Gene', axis=1, inplace=True)
mlp_df.head()
# try basic clustering
kMeans_clustering(9)
mlp_df.to_csv("stim_dense_short.csv", index = False)