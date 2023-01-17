import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import spacy
data = pd.read_csv("../input/steam-reviews-dataset/steam_reviews.csv")
'''

Some basic EDA after which i've abandoned/alterated some of the initial ideas for this project (above is the final version)

'''
data.describe()
data.head()
data['review_length'] = data.apply(lambda row: len(str(row['review'])), axis=1)



data['recommendation_int'] = data['recommendation'] == 'Recommended'

data['recommendation_int'] = data['recommendation_int'].astype(int)
data.head()
len(data['title'].unique()), data['title'].unique()
reviews_count = data.groupby(['title'])['review'].count().sort_values(ascending=False)



reviews_count = reviews_count.reset_index()



sns.set(style="darkgrid")

plt.figure(figsize=(25,20))

sns.barplot(y=reviews_count['title'], x=reviews_count['review'], data=reviews_count,

            label="Total", color="r")



reviews_count_pos = data.groupby(['title', 'recommendation_int'])['review'].count().sort_values(ascending=False)

reviews_count_pos = reviews_count_pos.reset_index()

reviews_count_pos = reviews_count_pos[reviews_count_pos['recommendation_int'] == 1]

sns.barplot(y=reviews_count_pos['title'], x=reviews_count_pos['review'], data=reviews_count_pos,

            label="Total", color="b")





data.groupby(['title', 'recommendation_int'])['review'].count()

data[data['title'] == "Tom Clancy's Rainbow Six® Siege"]



# R6 Siege in reality has much more reviews and much more mixed score

#=> *pos/neg reviews distribution is completely unrepresentative for some games

#=> *quantity of  reviews distribution is completely unrepresentative for some games

polarity_count = data.groupby(['recommendation_int']).count()

polarity_count = polarity_count.reset_index()





ax = sns.barplot(x=polarity_count['recommendation_int'], y=polarity_count['review'],

            data=polarity_count, hue='recommendation_int')





'''

#Just a different take on visualization:



polarity_count_pos = polarity_count[polarity_count['recommendation_int'] == 1]

sns.barplot(x=polarity_count_pos['recommendation_int'], y=polarity_count_pos['review'], data=polarity_count_pos,

            label="Total", color="b")



polarity_count_neg = polarity_count[polarity_count['recommendation_int'] == 0]

sns.barplot(x=polarity_count_neg['recommendation_int'], y=polarity_count_neg['review'], data=polarity_count_neg,

            label="Total", color="r")

'''
polarity_count = data[data['helpful'] > 50].groupby(['recommendation_int']).count()

polarity_count = polarity_count.reset_index()





ax = sns.barplot(x=polarity_count['recommendation_int'], y=polarity_count['review'],

            data=polarity_count, hue='recommendation_int')

data.isnull().sum()
'''

 This cell can be used to prepare reviews data for classifiers

 

 Can take data for all products in general (to analyze typical product reviews)

 Can also take data just for a specific product (e.g. 'Grand Theft Auto V')

'''



clean_data = data.dropna()



#train = clean_data

train = clean_data[clean_data['title'] == 'Grand Theft Auto V']





X = train['review']

y = train['recommendation_int']



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=273, stratify=y)
'''

Custom tokenizer for lemmatization from spaCy tokenizer



Used in vectorizers in different tasks here



Prefer using on small datasets, or else processing will take too long

'''



import spacy



# load spacy language model

en_nlp = spacy.load('en', disable=['parser', 'ner'])

spacy_tokenizer = en_nlp.tokenizer



# create a custom tokenizer using the spaCy document processing pipeline

# (now using our own tokenizer)

def custom_tokenizer(document):

    doc_spacy = en_nlp(document)

    return [token.lemma_ for token in doc_spacy]
'''

 This cell can be used to build classifiers of positive/negative reviews

 which can be inspected to get insights about why certain reviews are positive/negative

 

 Can be used to analyse what people like/dislike about certain product

 or a selection of products in general

 

 Can possibly be used to label unlabeled comments about the product (like comments on forums)

 

 Deep Learning alternative: LSTM(uni/bi-directional) / 1D-CNN / MLP / their combinations

 can be used, although my limited tests on IMDB dataset(which is quite similair to this Steam Reviews dataset)

 have not found any reasonable justification to use Deep Learning over plain LogisticRegression

 because loss of explainability is obvious, hypothesis space is drasticaly increased, so optimization is harder,

 and, most of all, accuracy gain is negliable (89 LogReg vs 89-90 DL with a lot of tweaking)

'''



from time import time



#from sklearn.naive_bayes import MultinomialNB

#from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

#from sklearn.linear_model import SGDClassifier



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

#from sklearn.feature_extraction.text import TfidfTransformer



#from sklearn.model_selection import GridSearchCV

#from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report





# Zoo of vectorizers inside a pipline, some include comments

t0 = time()

text_clf = Pipeline([#('vect', CountVectorizer(min_df=5)),

                     #('vect', TfidfTransformer(norm=None)),

                     ('vect', TfidfVectorizer(max_df=0.99, norm='l2')), #< default, cuts some generic words

                     #('vect', TfidfVectorizer(max_df=0.2, norm='l2')), #< default, leaves generic words

                     #('vect', TfidfVectorizer(max_df=0.99, norm='l2', sublinear_tf=True, ngram_range=(2, 2), tokenizer=custom_tokenizer)), #500 sec journey

                     #('vect', TfidfVectorizer(max_df=0.99, norm='l2', sublinear_tf=True, tokenizer=custom_tokenizer)),

                     #('vect', TfidfVectorizer(max_df=0.99, norm='l2', ngram_range=(4, 4))), #< some useful info

                     #('vect', TfidfVectorizer(min_df=5, norm='l2', tokenizer=custom_tokenizer)),

                     #('vect', TfidfVectorizer(min_df=5, norm='l2', tokenizer=custom_tokenizer, max_features=10000)),

                     #('clf', MultinomialNB())

                     #('clf', LogisticRegression(solver='saga', fit_intercept=True, class_weight='balanced'))

                     ('clf', LogisticRegression(solver='saga', fit_intercept=True, class_weight='balanced', C=0.1)) #< reasonable

                     #('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=273, max_iter=5, tol=None)) #< SVM, SGD version, sometimes delivers good results

                    ])

print("preprocessing done in %0.3fs." % (time() - t0))





t0 = time()

text_clf.fit(X_train, y_train)

print("fitting done in %0.3fs." % (time() - t0))



t0 = time()

y_pred = text_clf.predict(X_test)

print("predicting done in %0.3fs." % (time() - t0))

#target_names = ['class 0', 'class 1', 'class 2']

print(classification_report(y_test, y_pred)) #, target_names=target_names))
'''

 This cell can be used to visualize LogisticRegression coefficients attributed to the tokens,

 which should give an idea of what make a review positive/negative, why customers like/dislike the product, etc.

'''



import eli5



eli5.show_weights(text_clf, vec=text_clf.named_steps["vect"], top=40)
'''

 This cell can be used to setup data for topic modeling

 

 Designed to prepare a corpus of negative/positive reviews for certain product

 to further extract latent features via topic modeling methods (LDA/NMF) 

 

 Train/test naming and split steps are present for cinsistency

'''



clean_data = data.dropna()



# Example: here we want to find out why customers who left negative reviews for certain product are not satisfied, 

train = clean_data[(clean_data['title'] == 'Grand Theft Auto V') & (clean_data['recommendation_int'] == 0)]



X = train['review']

y = train['recommendation_int']



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=273, stratify=y)
'''

 This cell can be used to visualize topics modeled by LDA/NMF.

 

 Designed to visualize hidden features of products which affect customer's satisfaction

 

 Default is the wordcloud visualization, 

 although topics' contents can be printed as lists of tokens (uncomment the respective lines)

 

 For colormaps list google: "Matplotlib colormap reference", some examples:

 -colormap = 'summer' < suitable for positive reviews

 -colormap = 'inferno' < suitable for negative reviews

 

'''



from wordcloud import WordCloud, STOPWORDS



def print_top_words(model, feature_names, n_top_words, colormap='viridis'):

    for topic_idx, topic in enumerate(model.components_):

        

        #to print topics' contents as lists of tokens

        #message = "Topic #%d: " % topic_idx

        

        message = " ".join([feature_names[i]

                             for i in topic.argsort()[:-n_top_words - 1:-1]])

        

        #to print topics' contents as lists of tokens

        #print(message + '\n')

        

        generate_wordcloud(message, colormap)

    print()    





def generate_wordcloud(text, colormap='viridis'):

    wordcloud = WordCloud(#font_path='/Library/Fonts/Verdana.ttf',

                          relative_scaling = 1.0,

                          colormap = colormap

                          #colormap = 'summer', #< suitable for positive reviews

                          #colormap = 'inferno', #< suitable for negative reviews

                          #stopwords = STOPWORDS

                          ).generate(text)

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.show()



# for testing the generate_wordcloud():

#text = 'all your base are belong to us all of your base base base'

#generate_wordcloud(text)
'''

This cell can be used for topic modeling w/ NMF



For the data used here NMF is often faster (so more preferable)

'''



from sklearn.decomposition import NMF



# Zoo of tested vectorizers:

tfidf_vect = TfidfVectorizer(max_df=.50) #< quick results

#tfidf_vect = TfidfVectorizer(max_df=.50, tokenizer=custom_tokenizer) #< uses spaCy tokenizer w/ lemmatization, good for smaller datasets

#tfidf_vect = TfidfVectorizer(ngram_range=(1, 2)) #< experimental, long to compute, questionable results, but can be insightful



# Transform dataset, extract topics

X_train_topical = tfidf_vect.fit_transform(X_train)



nmf = NMF(n_components=5, random_state=273,

          alpha=.1, l1_ratio=.5)



document_topics_nmf = nmf.fit_transform(X_train_topical)
# Get topic contents and visualize the topics, example here - for negative reviews

tfidf_vect_feature_names = tfidf_vect.get_feature_names()

print_top_words(nmf, tfidf_vect_feature_names, 100, colormap='inferno')
'''

Here's another data setup for topic modeling, but for different example:

extract topics from positive reviews using LDA

'''



clean_data = data.dropna()



# Example: here we want to find out why customers who left positive reviews for certain product are satisfied, 

train = clean_data[(clean_data['title'] == 'Grand Theft Auto V') & (clean_data['recommendation_int'] == 1)]



X = train['review']

y = train['recommendation_int']



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=273, stratify=y)
'''

This cell can be used for topic modeling w/ LDA



For the data used here NMF is often faster (so more preferable)



For unsupervised text document models, 

it is often good to remove very common words,

as they might otherwise dominate the analysis. 

We’ll remove words that appear in atleast  20  percent  of  the  documents,  

and  we’ll  limit  the  bag-of-words  model  to  the

10,000 words that are most common after removing the top 20 percent:

'''



from sklearn.decomposition import LatentDirichletAllocation



vect = CountVectorizer(max_features=10000, max_df=.20)

X_train_topical = vect.fit_transform(X_train)



lda = LatentDirichletAllocation(n_components=5, learning_method="batch",

                                max_iter=25, random_state=273)

# We build the model and transform the data in one step

# Computing transform takes some time,

# and we can save time by doing both at once

document_topics = lda.fit_transform(X_train_topical)
vect_feature_names = vect.get_feature_names()

print_top_words(lda, vect_feature_names, 100, colormap='summer')
'''

 This cell can be used to calculate the strength of positive/negative association 

 of the current product (e.g. 'Grand Theft Auto V') with other products

 being mentioned in the customers' reviews for this product

 

 Essentially it's an NER task with elements of Sentiment Analysis

 

 WIP, currently only shows entity counts from short handcrafted list

 across all the reviews

  

 [Assumptions: 

     Customers tend to mention other products in positive reviews for this product

     when they think this product is at least no worse than other products, 

     or maybe even better,

     when they like certain positive features from other products are found in this product

     etc.

     

     At the same time customers mention other products in negative reviews for this product

     when they think this product is at least no better than other products,

     or maybe even worse,

     when they dislike certain negative features from other products are found in this product

     etc.

     

     Ties can be broken based on majority ratio/percent difference threshold, etc.

     Cool idea for breaking ties and visualization - LogisticRegression w/ eli5 visualizer

     on both pos/neg reviews set for a preduct]

     

 WORK IN PROGRESS: 

     Requires heavy entity matching, coz no one writes 

     full correct names of the products like they are in the database;

     

     Demo example is provided for the whole set of reviews(all products mixed)

     with some custom entity matching

     

     Potentially cool implementation w/ LogisticRegression(see above)

'''



# Data preparation

clean_data = data.dropna()



# Usecase of the final version

#train = clean_data[(clean_data['title'] == 'Grand Theft Auto V') 

                  # & (clean_data['recommendation_int'] == 1)]  #< pick one

                  # & (clean_data['recommendation_int'] == 0)]  #  not both



# Demo usecase

train = clean_data



# Routine split for potential use in logistic regression

X = train['review']

y = train['recommendation_int']



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=273, stratify=y)







# Kind of a comprehensive list of entities (product names from Steam store), about 40k names,

# Requires entity matching to be of practical value

#steam_games = pd.read_csv("../input/steam-games-complete-dataset/steam_games.csv")

#ents = steam_games['name'].unique()



# Demo list of entities with matching 

ents = {'GTA': 0, 'GTA5':1, 'GTAV':2, 'GTA V':3, 'GTA 5':4, 'gta 5':5, 'gta5':6, 'PUBG':7, 'pubg':8}



# Making custom vocabulary from entities to do NER via BoW

ner = CountVectorizer(vocabulary=ents)

ner_fit=ner.fit_transform(X_train)



# Sum entity mentions across the corpus

counts = np.asarray(ner_fit.sum(axis=0)).ravel()



# Output entity names and number of mentions across the corpus

for ent_idx, ent in enumerate(ents):

    

    #DEBUG

    #print(ent_idx, ent)

    

    if (counts[ent_idx] != 0):

        print(ent, counts[ent_idx])