import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import matplotlib

matplotlib.use(u'nbAgg')

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.manifold import TSNE

from sklearn import preprocessing

from tqdm import tqdm
asm_reduced_final = pd.read_csv("../input/asm_reduced_final.csv")

asm_reduced_final.head()
labels = pd.read_csv("../input/trainLabels.csv")
asm_data = asm_reduced_final.merge(labels, on = "Id")

asm_data.head()
asm_y = asm_data["Class"]

asm_x = asm_data.drop("Class", axis=1)
#Let's normalize the data.

def normalize(dataframe):

    #print("Here")

    test = dataframe.copy()

    for col in tqdm(test.columns):

        if(col != "Id" and col !="Class"):

            max_val = max(dataframe[col])

            min_val = min(dataframe[col])

            test[col] = (dataframe[col] - min_val) / (max_val-min_val)

    return test
asm_x = normalize(asm_x)
xtsne=TSNE(perplexity=30)

results=xtsne.fit_transform(asm_x.drop(['Id'], axis=1))

vis_x = results[:, 0]

vis_y = results[:, 1]

plt.scatter(vis_x, vis_y, c=asm_y, cmap=plt.cm.get_cmap("jet", 9))

plt.colorbar(ticks=range(10))

plt.clim(0.5, 9)

plt.show()
data_asm_byte_final = pd.read_csv("../input/data_asm_byte_final.csv", index_col = 0)

data_asm_byte_final.head()
asm_byte_y = data_asm_byte_final["Class"]

asm_byte_x = data_asm_byte_final.drop("Class", axis=1)
#Let's normalize the data.

def normalize(dataframe):

    #print("Here")

    test = dataframe.copy()

    for col in tqdm(test.columns):

        if(col != "Id" and col !="Class"):

            max_val = max(dataframe[col])

            min_val = min(dataframe[col])

            test[col] = (dataframe[col] - min_val) / (max_val-min_val)

    return test
asm_byte_x = normalize(asm_byte_x)
xtsne=TSNE(perplexity=30)

results=xtsne.fit_transform(asm_byte_x.drop(['Id'], axis=1))

vis_x = results[:, 0]

vis_y = results[:, 1]

plt.scatter(vis_x, vis_y, c=asm_byte_y, cmap=plt.cm.get_cmap("jet", 9))

plt.colorbar(ticks=range(10))

plt.clim(0.5, 9)

plt.show()