# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import seaborn as sns



import matplotlib.pyplot as plt



%matplotlib inline



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans



from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
shakespeare_df = pd.read_csv("../input/Shakespeare_data.csv")



# Drop NaN values to hide all stage directions and non-spoken parts

shakespeare_df = shakespeare_df.dropna(axis=0)

list_of_plays = []

names_of_plays = []

for play in shakespeare_df.Play.unique():

    _play = shakespeare_df[shakespeare_df.Play == play]

    _text = ""

    for index, row in _play.iterrows():

        _text += row["Player-Line"].lower()

        _text += " "

    list_of_plays.append(_text)

    names_of_plays.append(play)
tfidf = TfidfVectorizer(ngram_range=(2,2))

tfidf.fit(list_of_plays)

tfidf_vec = tfidf.transform(list_of_plays)

pca_red = PCA(n_components=2,random_state=1)

pca_red.fit(tfidf_vec.toarray())

tfidf_pca = pca_red.transform(tfidf_vec.toarray())



km = KMeans(n_clusters=4,random_state=1)

km.fit(tfidf_pca)



color_list = ["r","g","b","cyan"]

colormap = [color_list[color] for color in km.labels_]



plt.figure(figsize = (20,20))

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], marker = ".",c = "r",s=600)

for cluster,x,y in zip([0,1,2,3],km.cluster_centers_[:,0],km.cluster_centers_[:,1]):

    plt.annotate(cluster,xy=(x,y),fontsize=24,color="purple")

plt.scatter(tfidf_pca[:,0], tfidf_pca[:,1], marker="o",c=colormap,s=500,alpha=0.3)

for label, x, y in zip(names_of_plays,tfidf_pca[:,0], tfidf_pca[:,1]):

    plt.annotate(label,xy=(x,y))

    
combined_texts = ["","","",""]

for index,text in zip(km.labels_,list_of_plays):

    combined_texts[index] += text



for text in combined_texts:

    plt.figure(figsize=(12,10))

    wordcloud = WordCloud(max_words=150, width=1000,height=600).generate(text)

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.show()