import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

#import plotly.offline as py

#py.init_notebook_mode(connected=True)

#import plotly.graph_objs as go

#import plotly.tools as tls

import pandas as pd

import fuzzywuzzy

from fuzzywuzzy import fuzz

import itertools

#import distance

#import cPickle

import pandas as pd

import numpy as np

#import utils

from fuzzywuzzy import fuzz

#from gensim.models import Word2Vec

import nltk

from nltk.corpus import stopwords



from scipy.stats import skew, kurtosis

from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

from nltk import word_tokenize

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from subprocess import check_output



%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls





df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")

df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")







df_train.head(1)
df_test.head(1)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df_train['same_security'] = le.fit_transform(df_train['same_security'])

df_test['same_security'] = le.fit_transform(df_test['same_security'])

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

len(df_all)
dfs = df_all[0:2658]

%%time



df_all['levinstein'] = [[]] * len(df_all)

for i in range(len(df_all)):

    df_all['levinstein'][i] = distance.levenshtein(df_all['description_x'][i], df_all['description_y'][i])

    

df_all['jaccard'] = [[]] * len(df_all)

for i in range(len(df_all)):

    df_all['jaccard'][i] = distance.jaccard(df_all['description_x'][i],df_all['description_y'][i])

    

df_all['sorensen'] = [[]] * len(df_all)

for i in range(len(df_all)):

    df_all['sorensen'][i] = distance.sorensen(df_all['description_x'][i], df_all['description_y'][i])

    

df_all['bleu'] = [[]] * len(df_all)

for i in range(len(df_all)):

    df_all['bleu'][i] = nltk.translate.bleu_score.sentence_bleu(df_all['description_x'][i],df_all['description_y'][i]) 
df_all['Common'] = df_all.apply(lambda row: len(list(set(row['description_x']).intersection(row['description_y']))), axis=1)

df_all['Average'] = df_all.apply(lambda row: 0.5*(len(row['description_x'])+len(row['description_y'])), axis=1)

df_all['Percentage'] = df_all.apply(lambda row: row['Common']*100.0/(row['Average']+1), axis=1)
desx, desy = df_train[['description_x']], df_test[['description_y']]

desx.columns = ['Des']

desy.columns = ['Des']

description = pd.concat((desx, desy), axis=0).fillna("")

description.shape
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

mq1 = TfidfVectorizer(max_features = 256).fit_transform(description['Des'].values)

mq1


diff_encodings = mq1[::2] - mq1[1::2]

diff_encodings
from sklearn.manifold import TSNE

tsne = TSNE(

    n_components=3,

    init='random', # pca

    random_state=101,

    method='barnes_hut',

    n_iter=200,

    verbose=2,

    angle=0.5

).fit_transform(diff_encodings.toarray())
trace1 = go.Scatter3d(

    x=tsne[:,0],

    y=tsne[:,1],

    z=tsne[:,2],

    mode='lines',

    marker=dict(

        sizemode='diameter',

        color = dfs['same_security'].values,

        colorscale = 'Portland',

        colorbar = dict(title = 'duplicate'),

        line=dict(color='rgb(255, 255, 255)'),

        opacity=0.75

    )

)



data=[trace1]

layout=dict(height=800, width=800, title='Text Similarity')

fig=dict(data=data, layout=layout)

py.iplot(fig, filename='3DBubble')