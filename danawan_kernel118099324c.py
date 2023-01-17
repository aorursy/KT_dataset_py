from IPython.display import Image

Image(filename='/kaggle/input/screen-shot/screen_shot.png', width=800)
!cp /kaggle/input/topic-code-v2/* ./
import numpy as np

import scipy.io as scio

import matplotlib.pyplot as plt

from scipy import sparse as sp

import time

from logger import Logger

#import cPickle as pickle

import pickle

import json

import os

from dic_top_vision import dic_topic_vision





realmin = 2.2e-308



train_data = sp.load_npz('/kaggle/input/cord-bow-data/cord19_10000.npz')

train_data = train_data.toarray()

np.random.seed(2018)

np.random.shuffle(train_data)

train_data = train_data.T

voc = np.load('/kaggle/input/cord-bow-data/voc_10000.npy')
from PGBN import PGBN



topic_model = PGBN(train_data,K=[128,64,32])

topic_model.train('./output',iteration = 1000)

topic_model.Phi_vis('./output',voc)
!cp /kaggle/input/topic-code/logger.py ./
!rm logger.py
import tensorflow as tf

writer = tf.summary.create_file_writer('logs')


import numpy as np

from graphviz import Digraph

import heapq

import pickle







#

def node(graph,layer,idx,num,voc,phi):

    max_index=heapq.nlargest(num, range(len(phi[:,idx])), phi[:,idx].take)

    label = ''

    for i in range(len(max_index)):

        label += str(voc[max_index[i]])+'\n'

    graph.node(str(layer)+'_'+str(idx),str(idx) + ' ' + label)



#

def weight_load(file_name):

    with open(file_name,"rb") as f:

        weight=pickle.load(f)

    return weight



# 

def voc_load(file_name):

    voc = []

    with open(file_name,'r') as f:

        lines = f.readlines()

        for idx,line in enumerate(lines):

            voc.append(line.strip())

    return voc





def plot(weight,voc,id,threshold,num):

    graph = Digraph()

    idx_2 = id 

    temp = threshold #

    #

    weight_0 = weight[0]

    weight_1 = weight[1]

    weight_2 = weight[2]

    phi_2 = weight_0.dot(weight_1).dot(weight_2)

    phi_1 = weight_0.dot(weight_1)

    phi_0 = weight[0]

    #

    node(graph,2,idx_2,num,voc,phi_2)

    idx_1 = np.where(weight_2[:,idx_2]>temp)

    for i in idx_1[0]:

        node(graph,1,i,10,voc,phi_1)

        graph.attr('edge',penwidth=str(weight_2[i][idx_2]*10))

        graph.edge('2_'+str(idx_2),'1_'+str(i))

        idx_0 = np.where(weight_1[:,i]>temp)

        for j in idx_0[0]:

            node(graph,0,j,10,voc,phi_0)

            graph.attr('edge',penwidth=str(weight_1[j][i]*10))

            graph.edge('1_'+str(i),'0_'+str(j))

    #graph.view()

    return graph

weight = weight_load('/kaggle/input/show-topic/Phi.pick')

voc = voc_load('/kaggle/input/show-topic/voc_10000.txt')

graph = plot(weight,voc,id=31,threshold=0.05,num=10) #
graph
from graphviz import Source

Source.from_file('./Digraph.gv')