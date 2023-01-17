import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

from matplotlib import pyplot as plt 

#%matplotlib inline
data = pd.read_csv('../input/train.csv')

data.describe()
from sklearn.model_selection import train_test_split

train_data, validation_data = train_test_split(data, test_size=0.3)



train_data.describe()
from gensim.models import word2vec

import gensim.models

import gensim.utils



sentences = data['text']

sentences = (sentences.map(lambda x: gensim.utils.simple_preprocess(x)).tolist())

print("# of sentences = {}".format(len(sentences)))



phrases = gensim.models.Phrases(sentences)

bigram = gensim.models.phrases.Phraser(phrases)

model = word2vec.Word2Vec(list(bigram[sentences]),size=100)



wv = model.wv

del model



similar = wv.most_similar(positive=['woman'], negative=['man'])

similar = map(lambda x: '{}\t\t\t{}'.format(x[0],x[1]),similar)

print('Similar Words:','\n' + '\n'.join(similar))



print('Family - Woman similarity:',wv.similarity('family','woman'))

print('Family - Man similarity:',wv.similarity('family','man'))



vocab = list(wv.vocab.keys())

features = wv[wv.vocab]



print('{} Words in model'.format(len(vocab)))
from sklearn.manifold import TSNE



features_2d = TSNE(n_components=2,perplexity = 1).fit_transform(features)



'''

plt.figure()

#plt.scatter(features_2d[:,0], features_2d[:,1])



for i in range(features_2d.shape[0]):

    plt.text(features_2d[i,0],features_2d[i,1], vocab[i],

             horizontalalignment='center', verticalalignment='center',clip_on=True)



plt.xlim(np.min(features_2d[:,0]), np.max(features_2d[:,0]))

plt.ylim(np.min(features_2d[:,1]), np.max(features_2d[:,1]))

plt.show()

'''
import matplotlib.cm

from sklearn.cluster import DBSCAN



cluster_alg = DBSCAN(eps=3)



#from sklearn.cluster



clusters = cluster_alg.fit_predict(features_2d)



nclusters = np.unique(clusters).shape[0]

print("{} clusters".format(nclusters))



norm = matplotlib.colors.Normalize(vmin=0, vmax=nclusters, clip=True)

mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Spectral)



plt.figure(figsize=(10,10),dpi=500)

#plt.scatter(features_2d[:,0], features_2d[:,1])



for i in range(features_2d.shape[0]):

    plt.text(features_2d[i,0],features_2d[i,1], vocab[i], color=mapper.to_rgba(clusters[i]),

             horizontalalignment='center', verticalalignment='center',clip_on=True,

                fontsize=3)

    #plt.scatter(features_2d[i,0],features_2d[i,1],color=mapper.to_rgba(clusters[i]))



plt.xlim(np.min(features_2d[:,0]) - 5, np.max(features_2d[:,0]) + 5)

plt.ylim(np.min(features_2d[:,1]) - 5, np.max(features_2d[:,1]) + 5)

plt.title('Clustered vocabulary')

plt.savefig('out.csv',format = 'png',dpi=1000)

print('Saved...')



av = np.array(vocab)



nc = len(np.unique(clusters))

for c in np.unique(clusters):

    

    cf = av[clusters == c]

    if (cf.shape[0] > 1):

        print("{}[{}] : {}".format(c, cf.shape[0], cf))