# import the pandas library for handling CSVs and table-type data 

# import numpy for some (simple) linear algebra

import pandas as pd

import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
iris.data.shape
pd.DataFrame(iris.data, columns = iris.feature_names).head()
print(iris.target_names)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y = encoder.fit_transform(iris.target) #categorical index
from sklearn.model_selection import train_test_split
# splitting the data



X_train, X_test, y_train, y_test = train_test_split(iris.data, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(multi_class='multinomial',solver='lbfgs')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
pd.crosstab(y_pred,y_test)
#read in the data

data = pd.read_csv('http://sds-datacrunch.aau.dk/public/data/upwork_aom_300k.csv')
#let's check it

data.head()
#some descriptives



data.info()
# selecting the empty ones



data_subset_empty = data[data['main_category'].isnull()]
# selecting the complete ones



data_full = data[~data['main_category'].isnull()]
#Some descriptives of the complete table



data_full.info()
# Print out the different categories of tasks



for i in data_full['main_category'].unique():

    print(i)
# import encoders for the dependant variable



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder
# encode the dependant into dummy variables



encoder = LabelEncoder()

onehot = OneHotEncoder()



encoded = encoder.fit_transform(data_full['main_category']) #categorical index

y = onehot.fit_transform(encoded.reshape(-1,1)) # dummy matrix
# In caggle you have direct access to the large model, which is great

import spacy

nlp = spacy.load("en_core_web_lg")
# Let's try it out on a sentence

sentence = nlp("The weather in Beijing is great")
# this returns a collection of tokens such as:

sentence[3]
# Tokens contain many useful fetures

# Spacy recognizes parts of speech

# It also detects entities

print(sentence[3].pos_)

print(sentence.ents)

print(sentence.ents[0].label_)
# Each token has a vector representation

sentence[3].vector
# Sentences (several words) are represented as a mean of the 

# contained word-vectors

sentence.vector
tokens = nlp(u'dog cat banana')



for token1 in tokens:

    for token2 in tokens:

        print(token1.text, token2.text, token1.similarity(token2))
# Vectorizing the text-data

vector_list = []



for doc in nlp.pipe(data_full['as_opening_title'], n_threads=4, batch_size=10000):

    vector_list.append(doc.vector)
# Assamble the list of vectors into a matrix

X = np.vstack(vector_list)
X.shape
X[:1,:]
# splitting the data



X_train, X_test, y_train, y_test = train_test_split(X, encoded, test_size=0.2)
# Training a logistic regression



classifier = LogisticRegression(multi_class='multinomial',solver='lbfgs')

classifier.fit(X_train, y_train)
# How are we doing?



from sklearn.metrics import classification_report



y_pred = classifier.predict(X_test)



classes_list = y_test.tolist() + y_pred.tolist()



labels = sorted(set(classes_list))

targets = encoder.inverse_transform(labels)



print(classification_report(y_test, y_pred, target_names = targets))
# Importing the keras library for deep learning



import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout
classifier = Sequential() 



#### RED ####

classifier.add(Dense(units = 256, activation='relu', input_dim = 300))



#### BLUE ####

classifier.add(Dropout(rate = 0.3))

classifier.add(Dense(units = 512, activation='relu'))

classifier.add(Dropout(rate = 0.1))

classifier.add(Dense(units = 64,  activation='relu'))





#### GREEN ####

classifier.add(Dense(units = 13, activation='softmax'))





#### COMPILE ####

classifier.compile(optimizer="adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.summary()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
history = classifier.fit(X_train, y_train, batch_size= 500, epochs= 10, validation_data=(X_test, y_test))
pd.DataFrame(history.history)[['acc','val_acc']].plot()
from sklearn.metrics import classification_report



y_true = np.argmax(y_test, axis=1)

labels = sorted(set([x[0] for x in y_true.tolist()]))

targets = encoder.inverse_transform(labels)



y_pred = classifier.predict_classes(X_test)



print(classification_report(y_true, y_pred, target_names=targets))
# Extract a sample of 1000 rows from the empty-dataset

new_data = data_subset_empty.sample(1000)['as_opening_title']
# Reindex the sample (just to avoiod potential index-related-problems)

new_data.index = range(len(new_data))
# Vectorise the text with SpaCy



new_X = []



for doc in nlp.pipe(new_data, n_threads=4, batch_size=10000):

    new_X.append(doc.vector)



new_X = np.vstack(new_X)
# Make predictions with the neural network model



predictions = classifier.predict_classes(new_X)
# Write predictions into the dataset



result = pd.concat([new_data, pd.Series(predictions.tolist())], axis=1)
# Transform nummerical category-predictions into labels

result['category'] = encoder.inverse_transform(result[0])
result['category'].unique()
# Quick check?



result[result.category == 'Writing']
!pip3 install hdbscan
# loading the data?

data = pd.read_csv('http://sds-datacrunch.aau.dk/public/feelance_eda.csv')
# Quick data exploration

data.info()
data.head()
# How does one portfolio look like?

data[data.f_id == 78].sub_category
empty_list = []



print(empty_list)
empty_list.append(1)



print(empty_list)
empty_list.append("i don't want to be in that list")



print(empty_list)
empty_list.extend(['🐧','🍅','🤘'])



print(empty_list)
# individual freelancers

workers = data.f_id.unique()
#create empty list

stuff_people_do = []



for some_worker_id in workers: #initiate loop

  stuff = list(data[data.f_id == some_worker_id].sub_category) # extract portfolio for a single worker

  stuff_people_do.append((some_worker_id, stuff)) # append portfolio to the list of portfolios
#use pandas to make it into a datafrmae

portfolios = pd.DataFrame(stuff_people_do, columns = ['f_id', 'gig_portfolio'])

#Calculate the most common gig_activity

portfolios['max'] = portfolios['gig_portfolio'].map(lambda t: max(t))
#Gensim is actually an NLP library but we will use it here to construct BagOfJobs representations of freelancer portfolios.

import gensim

from gensim.corpora.dictionary import Dictionary
# First we ceate a dicitonary - an index-subcategory mapping object



dictionary = Dictionary(portfolios['gig_portfolio'])
# Create a "corpus of portfolios" in BoW format

corpus = [dictionary.doc2bow(sequence) for sequence in portfolios['gig_portfolio']]
# Transform corpus into a matrix 

portfolio_matrix = gensim.matutils.corpus2dense(corpus=corpus, num_terms=len(dictionary))
portfolio_matrix.shape
# swap rows and columns with a transponse

portfolio_matrix = portfolio_matrix.T
# portfolio of worker 0

portfolio_matrix[0]
# what's the taks sub-category of index 3?

dictionary.get(3)
# How many times did the worker perform this gig?

data[data.f_id == 0].sub_category
# Let's try to bring it all the way down to 5 dimensions



from sklearn.decomposition import NMF



model = NMF(n_components=5)



portfolio_matrix_reduced = model.fit_transform(portfolio_matrix)
# what are these components?

model.components_.shape
# Make a dataframe

components_df = pd.DataFrame(model.components_, columns=list(dictionary.values()))
components_df
# Select a component

component = components_df.iloc[0,:]



# Print result of nlargest

print(component.nlargest())
# Import clustering and dimensionality reduction

# HDBSCAN won't work with numpy < 1.16



import hdbscan

import umap



# Also, we will now need some visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns
# Let's try not to overdo things and maybe keep it at 20 components



model = NMF(n_components=20)



portfolio_matrix_reduced = model.fit_transform(portfolio_matrix)
# Note that the standard setting of UMAP will produce 2 dimensions

embedding = umap.UMAP(n_neighbors=15, metric='cosine').fit_transform(portfolio_matrix_reduced)
# Now, we will feed the 2-dimensional representation into HDBSCAN

# Warning can be ignored for now



clusterer = hdbscan.HDBSCAN(min_cluster_size=50, 

                            min_samples=50, 

                            leaf_size=40, 

                            #core_dist_n_jobs=16, 

                            prediction_data=True)

clusterer.fit(embedding)
pal = sns.color_palette("Paired", n_colors = len(set(clusterer.labels_)))[1:]

plt.figure(figsize=(12,12))

plt.rcParams.update({'font.size': 15})

clusterer.condensed_tree_.plot(select_clusters=True,

                               selection_palette=pal, label_clusters=True)
# Scatterplor of the UMAP embeddings with cluster-coloring

plt.rcParams.update({'font.size': 12})

plt.figure(figsize=(10,10))

g = sns.scatterplot(*embedding.T, 

                hue=clusterer.labels_, 

                legend='full',

                palette = 'Paired')

legend = g.get_legend()
portfolios['cluster'] = clusterer.labels_
import itertools

from collections import Counter
counter = Counter(list(itertools.chain(*portfolios[portfolios.cluster == 1]['gig_portfolio'])))

counter.most_common(20)