import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy import stats

import seaborn as sns

import kagglegym

import math



%matplotlib inline
env = kagglegym.make()



observation = env.reset()



train = observation.train
unique_ids = list(set(train.id))

len(unique_ids)
def getRankById(inst_id, method):

    data = train.loc[train.id == inst_id, train.columns]     

    

    data = data.ix[:,2:len(data.columns)]

    

    cor_with_y = data[data.columns[0:-1]].apply(lambda x: x.corr(data['y'], method=method))

    cors = cor_with_y.copy()

    cors_sq = cors**2 #square the results to handle highly negative correlations

    cors_sq.sort_values(inplace=True, ascending=False)

    return(cors_sq)

    

def plotRank(inst_id, num_features, cors_sq):

    pos = np.arange(num_features)

    width = 1.0     # gives histogram aspect to the bar diagram



    ax = plt.axes()

    ax.set_xticks(pos + (width / 2))

    ax.set_xticklabels(cors_sq[0:num_features].index.values, rotation=45)



    plt.bar(pos, cors_sq[0:num_features], width, color='r')

    plt.title("Correlation of " + str(inst_id) + " and y")

    plt.show()

    
inst_cors = {}

for inst_id in unique_ids[0:250]:

    cors = getRankById(inst_id, "spearman")

    inst_cors[inst_id] = cors

    if inst_id % 10 == 0:

        temp = plotRank(inst_id, 5, cors)
for inst_id in unique_ids[250:500]:

    cors = getRankById(inst_id, "spearman")

    inst_cors[inst_id] = cors

    if inst_id % 10 == 0:

        temp = plotRank(inst_id, 5, cors)
for inst_id in unique_ids[500:750]:

    cors = getRankById(inst_id, "spearman")

    inst_cors[inst_id] = cors

    if inst_id % 10 == 0:

        temp = plotRank(inst_id, 5, cors)
for inst_id in unique_ids[750:]:

    cors = getRankById(inst_id, "spearman")

    inst_cors[inst_id] = cors

    if inst_id % 10 == 0:

        temp = plotRank(inst_id, 5, cors)