 ##THE LYBRARIES USED IN THIS NOTEBOOK



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn import linear_model

from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.neighbors import NearestNeighbors

import seaborn as sns





#We will retrieve the Wikipedia Articles Dataset

dataset = '../input/people_wiki.csv'

people_wiki_df = pd.read_csv(dataset)

#Lets see what we have in the dataset

people_wiki_df.head(3)



#We get the text column from Bill

bill_clinton_text = people_wiki_df.loc[people_wiki_df['name']=='Bill Clinton', 'text'].tolist()

counter = CountVectorizer(stop_words='english')

count_matrix =counter.fit_transform(bill_clinton_text)

features = counter.get_feature_names()

#Create a series from the sparse matrix

clinton_counter = pd.Series(count_matrix.toarray().flatten(), 

              index = features).sort_values(ascending=False)

#We are gonna plot the most 50 frequent words in Bill's article, without taking under consideration  the stopwords.

bar_graph = clinton_counter[:50].plot(kind='bar', figsize=(18,8), alpha=1, fontsize=17, rot=90,edgecolor='black', linewidth=2,

            title='Bill Clinton Wikipedia Article Word Counts')

bar_graph.set_xlabel('Words')

bar_graph.set_ylabel('Occurrences')

bar_graph.title.set_size(18)



name = 'Bill Clinton'

#TfidfVectorizer: Converts a collection of raw documents to a matrix of TF-IDF features.

#min_df: When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.

#max_df: When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold.

#Apply this vectorizer to the full dataset to create normalized vectors

tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf =True, stop_words = 'english')

#tfidf_vectorizer.fit: Learn vocabulary and idf from training set.

tfidf_matrix = tfidf_vectorizer.fit_transform(people_wiki_df.text.values)

#tfidf_vectorizer.get_feature_names(): Array mapping from feature integer indices to feature name

features = tfidf_vectorizer.get_feature_names()

#tfidf_vectorizer.get_feature_names(): Array mapping from feature integer indices to feature name

#Get the row that belongs to Bill Clinton

row = people_wiki_df[people_wiki_df.name==name].index.tolist()[0]

#Create a series from the sparse matrix

clinton_matrix = pd.Series(tfidf_matrix.getrow(row).toarray().flatten(),index = features).sort_values(ascending=False)

tf_idf_plot = clinton_matrix[:20].plot(kind='bar', title='Bill Clinton Wikipedia Article Word TF-IDF Values',

            figsize=(10,6), alpha=1, fontsize=14, rot=80,edgecolor='black', linewidth=2 )

tf_idf_plot.title.set_size(18)

tf_idf_plot.set_xlabel('WORDS')

tf_idf_plot.set_ylabel('TF-IDF')



#Number of neighbors to use by default is 5 that's why we dont give to knn_neighbours any parameter

#weights: where  you detail if all points in each neighborhood are weighted equally or not.

#algorithm:‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute

#By default the distance that the algorithm uses is Mink

#and so on. For further information take a look at scikit

knn_neighbours = NearestNeighbors(n_neighbors=20)

knn_neighbours.fit(tfidf_matrix)

#closest_friends returns a list of lists. Where the first one belongs to the distances and the second one to the rows

closest_friends = knn_neighbours.kneighbors(tfidf_matrix.getrow(row), return_distance=True)

names_index = closest_friends[1][0]

names = [people_wiki_df.iloc[row]['name'] for row in names_index]

distances = closest_friends[0][0]

#If it is the case, we delete the name that has distance zero.

if distances[0]==0.0: distances,names = np.delete(distances,0),np.delete(names,0) 

data = pd.DataFrame({'Distances': distances,'Neighbours':names })

sns.set_style("whitegrid")

plt.figure(figsize=(20, 10))

sns.set(font_scale=1.5)

closest_neighbours_to_bill = sns.barplot(x='Distances', y="Neighbours", data= data ,linewidth=1,edgecolor=".2",palette="Blues",saturation=1)

closest_neighbours_to_bill.set_title('Closest neighbours to Bill')

closest_neighbours_to_bill.set(xlabel='DISTANCES', ylabel='NEIGHBOURS')

plt.show()








