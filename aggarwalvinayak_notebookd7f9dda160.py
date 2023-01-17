import csv

import numpy as np

import pandas as pd

from sklearn.metrics import f1_score

from sklearn.utils import shuffle

from collections import Counter

import imblearn

from imblearn.over_sampling import SMOTE,ADASYN

import copy 

from copy import deepcopy

from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout,LSTM, Bidirectional,Activation

from keras.layers.embeddings import Embedding

from keras.utils import Sequence

from keras import backend as K

from keras.wrappers.scikit_learn import KerasClassifier

import tensorflow as tf

from sklearn.model_selection import GridSearchCV,StratifiedKFold,KFold

from sklearn.metrics import make_scorer, accuracy_score, precision_recall_fscore_support,classification_report

from time import time

import statistics

import requests

import random
def csv_to_list(path):

    with open(path) as f:

        reader = csv.reader(f)

        data = list(reader)

        data = data[0]

        return data
!pip install biopython
data_1_3a = csv_to_list('../input/genesp/stress1a_genes.csv')

data_0_3b = csv_to_list('../input/genesp/stress3b_genes.csv')

data_1_2a = csv_to_list('../input/genesp/stress2a_genes.csv')

data_0_2b = csv_to_list('../input/genesp/stress2b_genes.csv')

data_1_1a = csv_to_list('../input/genesp/stress1a_genes.csv')

data_0_1b = csv_to_list('../input/genesp/stress1b_genes.csv')

data_n = csv_to_list('../input/genesp/stressnet_genes.csv')
from Bio import pairwise2

from Bio.Seq import Seq



dataup = data_1_3a +data_1_1a +data_1_2a

datadown =  data_0_3b +data_0_1b +data_0_2b

datanet = data_n

random.shuffle(datanet)

random.shuffle(dataup)

random.shuffle(datadown)



dataup=dataup[:100]

datadown=datadown[:100]

datanet=datanet[:100]





res = np.zeros((300, 300))

datatot=dataup+datadown+datanet

# print(len(datatot))

maxx = 0

for i in range(len(datatot)):

    for j in range(i,len(datatot)):



        if (i==j):

            res[i][i]=1

        else:

            alignnn =pairwise2.align.globalxx(datatot[i],datatot[j])

            minn = min(len(datatot[i]),len(datatot[j]))

            res[i][j] = round(((alignnn[0][2]/minn)*100),2)

        maxx = max(maxx,res[i][j])



# for i

print(maxx)

res



# sim = []

# lis = []

# seq1 = Seq(data_n[0].lower())

# seq3 = Seq(data_n[1].lower())

# seq4 = Seq(data_n[4].lower())

# lis.append(seq1)

# lis.append(seq3)

# lis.append(seq4)

# for i in range (2,len(data_n)):

#     num1 = 0

#     num2 = 0

#     num3 = 0

#     print('',end=".")

#     seq2 = Seq(data_n[i].lower())

#     alignments1 = pairwise2.align.globalxx(seq1, seq2)

#     alignments2 = pairwise2.align.globalxx(seq3, seq2)

#     alignments3 = pairwise2.align.globalxx(seq4, seq2)

#     l1 = min(len(seq1),len(seq2))

#     l2 = min(len(seq3),len(seq2))

#     l3 = min(len(seq4),len(seq2))

#     if (len(alignments1)>0):

#         num1 = round(((alignments1[0][2]/l1)*100),2)

#         sim.append(num1)

        

#     if (len(alignments2)>0):

#         num2 = round(((alignments2[0][2]/l2)*100),2)

#         sim.append(num2)

        

#     if (len(alignments3)>0):

#         num3 = round(((alignments3[0][2]/l3)*100),2)

#         sim.append(num3)

        

#     if (num1 < 95 and num2 < 95 and num3 < 95):

#         lis.append(data_n[i].lower())

df = pd.DataFrame(data=res)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "out.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df)