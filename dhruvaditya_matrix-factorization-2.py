import tensorflow as tf
import numpy as np
!pip install Node2Vec
from node2vec import Node2Vec
import pandas as pd
data=pd.read_csv('../input/movielens/ratings.dat',delimiter='::',names=['userid','itemid','rating','timestamp'])
list_of_nodes=list(set(data['userid']))

import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from random import random
import numpy as np

G = nx.Graph()
number_of_nodes=len(list_of_nodes)

# list_of_nodes=np.arange(number_of_nodes)
G.add_nodes_from(list_of_nodes)

for i in range(len(list_of_nodes)):
    initial_prob=0.4
    print("\r {} /{} ".format(i,len(list_of_nodes)),end='')
    for j in range(i+1,number_of_nodes):
        a=random()
        if(a<=initial_prob):
            G.add_edge(list_of_nodes[i],list_of_nodes[j])
        else:
            initial_prob=initial_prob/2
from collections import defaultdict
probs=[0.15,0.5,0.85]
all_treatments=[]
for prob in probs:
    a=dict()

    for node in list_of_nodes:
        a[node]=0

    for edge in G.edges:
        a[edge[0]]+=1
        a[edge[1]]+=1

    tr=dict()
    for nodeno,node in enumerate(list_of_nodes):
        if(a[node]==0):
            if(random()>prob):
                tr[node]=1
            else:
                tr[node]=0
    treatments=[]

    for user in data['userid']:
        if(user in list(tr.keys())):
            treatments.append(tr[user])
        else:
            treatments.append(None)
    all_treatments.append(treatments)
for i in range(len(probs)):
    data['treatments'+str(i)]=all_treatments[i]
data=data.fillna(1)
data.head()
data.to_csv('movielenswithtreatment.csv',index=False)
node2vec = Node2Vec(G, dimensions=max(list(set(data['userid']))), walk_length=3, num_walks=20, workers=2)
model=node2vec.fit(window=10, min_count=1, batch_words=4)
embeddings=[]
for node in list_of_nodes:
    embeddings.append(model.wv.get_vector(str(node)))
from numpy import savez_compressed
savez_compressed('movie_lens_embeddings.npz', embeddings)
matrix=np.zeros(shape=(np.max(data['userid']),np.max(data['itemid'])),dtype=np.float32)
for i in range(len(data)):
    matrix[data['userid'][i]-1][data['itemid'][i]-1]=data['rating'][i]/5

treatment_matrix=np.zeros(shape=(np.max(data['userid']),np.max(data['itemid'])),dtype=np.float32)
for i in range(len(data)):
    treatment_matrix[data['userid'][i]-1][data['itemid'][i]-1]=data['treatments'][i]

users=np.max(data['userid'])
products=np.max(data['itemid'])
weight_initer = tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.01,seed=0)
features=8000

user_embedding = tf.compat.v1.get_variable(name="Weight1", dtype=tf.float32, shape=[users,features], initializer=weight_initer,)
prod_embedding =tf.compat.v1.get_variable(name="Weight2", dtype=tf.float32, shape=[features,products], initializer=weight_initer)
node_to_vec_embedding=tf.compat.v1.get_variable(name="Weight3", dtype=tf.float32, shape=[2,products], initializer=weight_initer)
regularizer1 = tf.nn.l2_loss(user_embedding)
regularizer2 = tf.nn.l2_loss(prod_embedding)

embeddings=np.asarray(embeddings)
y = tf.compat.v1.placeholder(tf.float32, shape=[users,products])

loss1=tf.reduce_mean(tf.square(y-tf.matmul(user_embedding,prod_embedding)-tf.matmul(embeddings[0:users,0:users],treatment_matrix[0:users,0:products])))
loss2=tf.reduce_mean(tf.square(y-tf.matmul(user_embedding,prod_embedding)))
# *treatment_matrix[0:users,0:products]
# +0.00000001*regularizer1+0.00000001*regularizer2
learning_rate=0.01
train_step1 = tf.train.AdamOptimizer(learning_rate).minimize(loss1)
train_step2 = tf.train.AdamOptimizer(learning_rate).minimize(loss2)
import time
print(" Loss Function With Embedding and Treatment :")
l1=[]
iterations=100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    for i in range(iterations):
        sess.run(train_step1, feed_dict={y:matrix[0:users,0:products]})
        print("\r Cost for iteration  {} /{} ={} ".format(i+1,iterations,sess.run(loss1,feed_dict={y:matrix[0:users,0:products]})),end='')
        l1.append(sess.run(loss1,feed_dict={y:matrix[0:users,0:products]}))
    print("--- %s seconds ---" % (time.time() - start_time))
    print(sess.run(tf.matmul(user_embedding,prod_embedding)))
    y_pred=sess.run(tf.matmul(user_embedding,prod_embedding))
    

print(" Loss Function Without Embedding :")
l2=[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    for i in range(iterations):
        sess.run(train_step2, feed_dict={y:matrix[0:users,0:products]})
        print("\r Cost for iteration  {} /{} ={} ".format(i+1,iterations,sess.run(loss2,feed_dict={y:matrix[0:users,0:products]})),end='')
        l2.append(sess.run(loss2,feed_dict={y:matrix[0:users,0:products]}))
    print("--- %s seconds ---" % (time.time() - start_time))
    print(sess.run(tf.matmul(user_embedding,prod_embedding)))
    
    
    
matrix[0]
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error :",mean_absolute_error(matrix[0:users,0:products],y_pred))

import operator
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 
total_recall=0
total_precision=0
count=0
top=5


for user_no in range(users):
    user_recommended=y_pred[user_no]*5
    user_actual=matrix[user_no,:products]*5
    products_relevant=[]
    for product_no,rating in enumerate(user_actual):
        if(rating>3.5):
            products_relevant.append(product_no)

    lst = user_recommended
    indexed = list(enumerate(lst)) 
    top_n = sorted(indexed, key=operator.itemgetter(1))[-top:]
    top_recommended=list(reversed([i for i, v in top_n]))
    
    if(len(products_relevant)!=0):
        count+=1
        recall=len(intersection(top_recommended,products_relevant))/len(products_relevant)
        precision=len(intersection(top_recommended,products_relevant))/top
        total_recall+=recall
        total_precision+=precision
    
print("Precision :",total_precision*100/count,'%')
print("Recall    :",total_recall*100/count,'%')
f1_score=(2*(total_precision*100/count)*(total_recall*100/count))/((total_precision*100/count)+(total_recall*100/count))
print(f1_score)

import scipy
x = 1.96
norm_cdf = scipy.stats.norm.cdf(x)
norm_cdf
s = np.random.normal(0.5, 0.1, 1)
s
