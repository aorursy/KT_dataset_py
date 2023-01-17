# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/googlenewsvectorsnegative300"))



# Any results you write to the current directory are saved as output.
pd.options.display.max_rows = 64

pd.options.display.max_columns = 512
label = pd.read_csv('../input/imet-2019-fgvc6/labels.csv')
label_list = list(label.attribute_name.str.split(pat='::').map(lambda x: x[1]))

label_dict = dict(zip(label_list, list(label.attribute_id)))
data = pd.read_csv('../input/imet-2019-fgvc6/train.csv')
from gensim import models
w2v = models.KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin.gz', binary=True)
map_label = {}

exceptions = []

for item in label_list:

    try:

        map_label[item] = w2v[item]

    except BaseException:

        try:

            item_C = item[0].upper() + item[1:]

            map_label[item] = w2v[item_C]

        except BaseException:

            try:

                item__ = item.replace(' ','_')

                map_label[item] = w2v[item__]

            except BaseException:

                exceptions.append(item)

print(len(exceptions),len(map_label))
exceptions
np.zeros(2)
no_embed = np.zeros(len(exceptions))

no_embeds = []

for item in exceptions:

    no_embeds.append(label_dict[item])
for item in data.attribute_ids:

    for num in item.split(' '):

        num = int(num)

        for i in range(len(exceptions)):

            if num == no_embeds[i]:

                no_embed[i] += 1

no_embed.astype(np.int32)
count = 0

for item in data.attribute_ids:

    for num in item.split(' '):

        num = int(num)

        count += 1

count
df_labels = pd.DataFrame(map_label).T

df_labels.head()
from sklearn.cluster import KMeans

n_clusters = 12

km = KMeans(n_clusters=n_clusters, random_state=42, n_jobs=4)
kmeans = km.fit(df_labels)
centers = kmeans.cluster_centers_

centers
np.concatenate((df_labels.values,centers),axis=0).shape
from sklearn.manifold import TSNE



label_embedded = TSNE(n_components=2,perplexity=100).fit_transform(np.concatenate((centers,df_labels.values),axis=0))

label_embedded
import matplotlib.cm as cm

import matplotlib.pyplot as plt



colors = cm.gist_rainbow(np.linspace(0, 1, n_clusters))

plt.style.use('ggplot')



X = label_embedded[:,0]

Y = label_embedded[:,1]

c = kmeans.labels_



plt.figure(figsize=[10,10])

for i in range(len(X)):

    if i >= n_clusters:

        plt.scatter(X[i],Y[i],color=colors[c[i-n_clusters]])

    else:

        plt.scatter(X[i],Y[i],color=colors[i],s=500,label=str(i))

plt.legend()
df_labels['C'] = kmeans.labels_

df_labels.C.value_counts()