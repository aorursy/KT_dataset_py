# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt

data=pd.read_csv('/kaggle/input/indian-food-101/indian_food.csv')

data
sns.heatmap(data.isnull())
data['region'].value_counts()
f, axes = plt.subplots(2, 2, figsize=(15,15))

sns.countplot(x=data['diet'],ax=axes[0,0])

sns.countplot(x=data['flavor_profile'],ax=axes[0,1])

sns.countplot(x=data['course'],ax=axes[1,0])

sns.countplot(x=data['region'],ax=axes[1,1])

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2,figsize=(15,15))

# A standard pie plot

axs[0, 0].pie(data['diet'].value_counts().values, labels=data['diet'].value_counts().index, autopct='%1.1f%%', shadow=True)

axs[0, 1].pie(data['course'].value_counts().values, labels=data['course'].value_counts().index, autopct='%1.1f%%', shadow=True)

axs[1, 0].pie(data['region'].value_counts().values, labels=data['region'].value_counts().index, autopct='%1.1f%%', shadow=True)

axs[1, 1].pie(data['flavor_profile'].value_counts().values, labels=data['flavor_profile'].value_counts().index, autopct='%1.1f%%', shadow=True)

plt.show()
import geopandas as gpd

fp = "/kaggle/input/indiamap/Indian_States.shp"

map_df = gpd.read_file(fp)
desserts = data[data['course']=='dessert']
# desserts = data[data['course']=='dessert']

# #desserts

# des_df = desserts.state.value_counts().reset_index()

# des_df.columns = ['state','count']

# merged = map_df.set_index('st_nm').join(des_df.set_index('state'))

# merged

# fig, ax = plt.subplots(1, figsize=(10, 10))

# ax.axis('off')

# ax.set_title('State-wise Distribution of Indian Sweets',

#              fontdict={'fontsize': '15', 'fontweight' : '3'})

# fig = merged.plot(column='count', cmap='Wistia', linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)
desserts = data[data['state']=='Rajasthan'].reset_index(drop=True)

desserts
desserts = data[data['ingredients'].str.contains('yogurt') | data['ingredients'].str.contains('sugar')].reset_index(drop=True)

desserts         
desserts['ingredients'][2]
import pandas as pd

import gensim #the library for Topic modelling

from gensim.models.ldamulticore import LdaMulticore

from gensim import corpora, models

import pyLDAvis.gensim #LDA visualization library



from nltk.corpus import stopwords

import string

from nltk.stem.wordnet import WordNetLemmatizer



import warnings

warnings.simplefilter('ignore')

from itertools import chain
data=data[data['flavor_profile'] !='-1']
def seprate_string(cols):

    li = list(cols.split(", "))

    li = [string.replace(" ","").lower() for string in li]

    return li
data_food=data[['ingredients','flavor_profile']]

data_food['ingredients']=data_food['ingredients'].apply(seprate_string)
data_food
dictionary = corpora.Dictionary(data_food['ingredients'])
print(dictionary.num_nnz)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in data_food['ingredients'] ]

print(len(doc_term_matrix))
doc_term_matrix
lda = gensim.models.ldamodel.LdaModel
num_topics=3

%time ldamodel = lda(doc_term_matrix,num_topics=num_topics,id2word=dictionary,passes=50,minimum_probability=0)
ldamodel.print_topics(num_topics=num_topics)
lda_display = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary, sort_topics=False, mds='mmds')

pyLDAvis.display(lda_display)

lda_corpus = ldamodel[doc_term_matrix]

[doc for doc in lda_corpus]
scores = list(chain(*[[score for topic_id,score in topic] \

                      for topic in [doc for doc in lda_corpus]]))



threshold = sum(scores)/len(scores)

print(threshold)



cluster1 = [j for i,j in zip(lda_corpus,data_food.index) if i[0][1] > threshold]

cluster2 = [j for i,j in zip(lda_corpus,data_food.index) if i[1][1] > threshold]

cluster3 = [j for i,j in zip(lda_corpus,data_food.index) if i[2][1] > threshold]

#cluster4 = [j for i,j in zip(lda_corpus,data_food.index) if i[3][1] > threshold]

# cluster5 = [j for i,j in zip(lda_corpus,df.index) if i[4][1] > threshold]



print(len(cluster1))

print(len(cluster2))

print(len(cluster3))

#print(len(cluster4))

# print(len(cluster5))



cluster1
data_food.loc[cluster1]
data_food.loc[cluster2]
cluster3_df=data_food.loc[cluster3]
cluster3_df