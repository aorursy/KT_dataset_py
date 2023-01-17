# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Shakespeare_data.csv")

df.head()
df.info()
#Remove all stop works

import nltk

from nltk.corpus import stopwords # Import the stop word list

english_stop=stopwords.words("english") 

df['Player-Line']=df['Player-Line'].apply(lambda x:' '.join(w for w in nltk.word_tokenize(x.lower().strip()) if not w in english_stop) )

df.Word_tokens.head()

df_play_content=pd.DataFrame(df.groupby('Play')['Player-Line'].apply(lambda x: "{%s}" % ', '.join(x)))

df_play_content.head()


from wordcloud import WordCloud

def generateWordCloud(str1,title):



    wordcloud = WordCloud( background_color='black',width=900, height=800, max_font_size=40).generate(str1)

    wordcloud.recolor(random_state=0)

    plt.figure(figsize=(20, 15))

    plt.title(title, fontsize=60,color='red')

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis('off')

    plt.show()

    

#for cols in df_play_content.index:

    #generateWordCloud(df_play_content.loc[cols,'Player-Line'],cols)
wordcld = pd.Series(df['Player-Line'].tolist()).astype(str)

# Most frequent words in the data set. Just because. Using a beautiful wordcloud

from wordcloud import WordCloud 

cloud = WordCloud(width=900, height=800).generate(' '.join(wordcld.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
def build_corpus(data):

    corpus = []

    for sentence in data.iteritems():

            word_list = sentence[1].split(" ")

            corpus.append(word_list)

     

    return corpus

corpus = build_corpus(df['Player-Line'])
from gensim.models import word2vec



from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

%matplotlib inline



model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)
def tsne_plot(model):

    "Creates and TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(18, 18)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()

tsne_plot(model)
list_of_plays = []

names_of_plays = []

for play in df.Play.unique():

    _play = df[df.Play == play]

    _text = ""

    for index, row in _play.iterrows():

        _text += row["Player-Line"].lower()

        _text += " "

    list_of_plays.append(_text)

    names_of_plays.append(play)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans



tfidf = TfidfVectorizer(ngram_range=(2,2))

tfidf.fit(list_of_plays)

tfidf_vec = tfidf.transform(list_of_plays)

pca_red = PCA(n_components=2,random_state=1)

pca_red.fit(tfidf_vec.toarray())

tfidf_pca = pca_red.transform(tfidf_vec.toarray())



km = KMeans(n_clusters=5,random_state=1)

km.fit(tfidf_pca)



color_list = ["r","g","b","cyan","black"]

colormap = [color_list[color] for color in km.labels_]



plt.figure(figsize = (20,20))

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], marker = ".",c = "r",s=600)

for cluster,x,y in zip([0,1,2,3,4],km.cluster_centers_[:,0],km.cluster_centers_[:,1]):

    plt.annotate(cluster,xy=(x,y),fontsize=24,color="purple")

plt.scatter(tfidf_pca[:,0], tfidf_pca[:,1], marker="o",c=colormap,s=500,alpha=0.3)

for label, x, y in zip(names_of_plays,tfidf_pca[:,0], tfidf_pca[:,1]):

    plt.annotate(label,xy=(x,y))
labels = []

tokens = []



for word in model.wv.vocab:

    tokens.append(model[word])

    labels.append(word)

    

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

new_values = tsne_model.fit_transform(tokens)



x = []

y = []

for value in new_values:

    x.append(value[0])

    y.append(value[1])



km = KMeans(n_clusters=5,random_state=1)

km.fit(new_values)



color_list = ["r","g","b","cyan","black"]

colormap = [color_list[color] for color in km.labels_]



plt.figure(figsize = (20,20))

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], marker = ".",c = "r",s=600)

for cluster,x,y in zip([0,1,2,3,4],km.cluster_centers_[:,0],km.cluster_centers_[:,1]):

    plt.annotate(cluster,xy=(x,y),fontsize=24,color="purple")

plt.scatter(new_values[:,0], new_values[:,1], marker="o",c=colormap,s=500,alpha=0.3)

for label, x, y in zip(labels,new_values[:,0], new_values[:,1]):

    plt.annotate(label,xy=(x,y))