!pip install langdetect
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

from datetime import datetime, date, timedelta

import numpy as np

import re

import os



import matplotlib.pyplot as plt

import seaborn as sns



import nltk 

nltk.download('stopwords')

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

import gensim

from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit,train_test_split, GroupShuffleSplit

from langdetect import detect

from nltk.stem import PorterStemmer

from nltk.tokenize import word_tokenize

from keras.wrappers.scikit_learn import KerasClassifier



os.environ['KMP_DUPLICATE_LIB_OK']='True'



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# get NPI data

npis_csv = "/kaggle/input/covid19-challenges/npi_canada.csv"

raw_data = pd.read_csv(npis_csv,encoding = "ISO-8859-1")

# remove any rows that don't have a start_date, region, or intervention_category

df = raw_data.dropna(how='any', subset=['start_date', 'region', 'intervention_category'])

df['region'] = df['region'].replace('Newfoundland', 'Newfoundland and Labrador')

num_rows_removed = len(raw_data)-len(df)

print("Number of rows removed: {}".format(num_rows_removed))



# get all regions

regions = list(set(df.region.values))

print("Number of unique regions: {}".format(len(regions)))



# get all intervention categories

num_cats = list(set(df.intervention_category.values))

num_interventions = len(num_cats)

print("Number of unique intervention categories: {}".format(len(num_cats)))



# get earliest start date and latest start date

df['start_date'] = pd.to_datetime(df['start_date'], format='%Y-%m-%d')

earliest_start_date = df['start_date'].min()

latest_start_date = df['start_date'].max()

num_days = latest_start_date - earliest_start_date

print("Analyzing from {} to {} ({} days)".format(earliest_start_date.date(), latest_start_date.date(), num_days))

print("DONE READING DATA")

merged_tweets_csv = '/kaggle/input/npi-canada-tweets/tweets_to_intervention_category.merged.csv'

colnames = ["npi_record_id", "intervention_category", "oxford_government_response_category", "source_url", "id", "conversation_id", "created_at", "date", "time", "timezone", "user_id", "username", "name", "place", "tweet", "mentions", "urls", "photos", "replies_count", "retweets_count", "likes_count", "hashtags", "cashtags", "link", "retweet", "quote_url", "video", "near", "geo", "source", "user_rt_id", "user_rt", "retweet_id", "reply_to", "retweet_date", "translate", "trans_src", "trans_dest"]

tweets_df = pd.read_csv(merged_tweets_csv, encoding = "ISO-8859-1")

print(len(tweets_df))

# drop any rows without tweets - aka any interventions supported by non-tweeted media urls

tweets_df = tweets_df.dropna(how='any', subset=['npi_record_id', 'intervention_category', 'tweet'])

print(len(tweets_df))
# merge twitter dataset and the npi dataset

# npi dataset

data = []

for index, row in df.iterrows():

    data.append([row['intervention_category'], str(row['intervention_summary']) + ' ' + str(row['source_title']) + ' ' + str(row['source_full_text'])])

# tweet dataset

for index, row in tweets_df.iterrows():

    # detect only english tweets

    tweet = row['tweet'].strip()

    if tweet != "":

      language =""

      try:

          language = detect(tweet)

      except:

          language = "error"

      if language == "en":

        data.append([row['intervention_category'], tweet])

print(len(data))

# make it into a pandas dataframe

full_df = pd.DataFrame(data, columns=["intervention_category", "text"])
def get_binary_labels(df, target_category):

  '''Return binary labels in a list

  where 1 = category, 0 = everything else

  '''

  labels = pd.DataFrame(df['intervention_category'])

  labels.loc[labels.intervention_category == target_category, 'intervention_category'] = 1

  labels.loc[labels.intervention_category != 1, 'intervention_category'] = 0

  labels = labels.intervention_category.values.tolist()

  print("Number of {} = {}, Total = {} ".format(target_category, sum(labels), len(labels)))

  return labels



# binary labels for just one intervention category

y = get_binary_labels(full_df, "General case announcement")

y = np.asarray(y)
def preprocess(text):

  '''Return tokenized text with 

  removed URLs, usernames, hashtags, weird characters, repeated

  characters, stop words, and numbers

  '''

  text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text) # remove URLs

  text = re.sub("[^\w]", " ",  text).lower()

  text = re.sub('@[^\s]+', 'AT_USER', text) # removes any usernames in tweets

  text = re.sub(r'#([^\s]+)', r'\1', text) # remove the # in #hashtag

  text = re.sub('[^a-zA-Z0-9-*. ]', ' ', text) # remove any remaining weird characters

  words = word_tokenize(text)  # remove repeated characters (helloooooooo into hello)

  ignore = set(stopwords.words('english'))

  more_ignore = {'at', 'and', 'also', 'or', "http", "ca", "www", "https", "com", "twitter", "html", "news", "link"}

  ignore.update(more_ignore)

  #porter = PorterStemmer()

  #cleaned_words_tokens = [porter.stem(w) for w in words if w not in ignore]

  cleaned_words_tokens = [w for w in words if w not in ignore]

  cleaned_words_tokens = [w for w in cleaned_words_tokens if w.isalpha()]



  return cleaned_words_tokens
def get_tfidf_rep(df):

  '''Return array where each entry is a 

  tfidf represention of the NPI

  '''

  corpus = []

  for index, row in df.iterrows():

    # merge the texts and remove any stopwords, store as one string

    clean_tokens = preprocess(row['text'])

    clean_tokens_str = " ".join(clean_tokens)

    corpus.append(clean_tokens_str)



  # tf-idf representation

  vectorizer = TfidfVectorizer()

  d = vectorizer.fit_transform(corpus)

  X = d.toarray().tolist()



  data = []

  for l in X:

    data.append(np.asarray(l))



  return np.asarray(data)

def get_tagged_doc_corpus(X, labels, tokens_only=False):

  '''Return list of documents in gensim

  document object format'''

  corpus = []

  i = 0

  for index, row in X.iterrows():

    # merge the texts and remove any stopwords, store as one string

    clean_summary = preprocess(row["text"])

    # make binary label

    if tokens_only:

      corpus.append(clean_summary)

    else:

      corpus.append(gensim.models.doc2vec.TaggedDocument(clean_summary, [labels[i]]))

    i+=1

  return corpus



# CODE FROM: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-p

def train_doc2vec(df, y, category):

  '''Return a trained doc2vec model

  '''

  # split into train and test

  X_train, X_test, y_train, y_test = train_test_split(df, y, train_size=0.70,test_size=0.30, random_state=101)

  train_corpus = get_tagged_doc_corpus(X_train, y_train)

  test_corpus = get_tagged_doc_corpus(X_test, y_test, True)



  model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=40)

  model.build_vocab(train_corpus)

  model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)



  return model



def get_doc2vec_rep(model, y, df):

  '''Return entire dataframe in a list

  where each entry is the doc2vec vector

  for each NPI record'''

  X = []

  # gets tokenized versions of each text

  corpus = get_tagged_doc_corpus(df, y, True)

  for doc_id in range(len(corpus)):

    inferred_vector = model.infer_vector(corpus[doc_id])

    #sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    X.append(inferred_vector)

  return np.asarray(X)
reps = {}



# tfidf representation

reps["tfidf"] = get_tfidf_rep(full_df)



# doc2vec representation

model = train_doc2vec(full_df, y, "General case announcement")

reps["doc2vec"] = get_doc2vec_rep(model, y, full_df)
from sklearn.decomposition import PCA



def run_pca(X, labels, rep_name):

    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(X)

    principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

    finalDf = pd.concat([principalDf, labels], axis = 1)



    # choose a color palette with seaborn.

    num_classes = len(np.unique(labels))

    print(num_classes)

    palette = np.array(sns.color_palette("husl", num_classes))

    print(palette)



    plt.figure(figsize=(16,10))

    sns.scatterplot(

        x="pc1", y="pc2",

        hue="labels",

        palette=palette,

        s=60,

        data=finalDf,

        legend="full",

        alpha=0.5

    )



    margin = 0.05

    plt.xlim(min(finalDf['pc1'])-margin, max(finalDf['pc1'])+margin)

    plt.ylim(min(finalDf['pc2'])-margin, max(finalDf['pc2'])+margin)

    plt.title('PCA analysis ' + rep_name, fontsize = 15)

    plt.xlabel('Principal Component 1', fontsize=12)

    plt.ylabel('Principal Component 2', fontsize=12)



    plt.savefig('pca_' + rep_name + '.png', dpi=1000)

    plt.show()





# visualize representations

for r in reps:

    run_pca(reps[r], pd.DataFrame(y, columns = ['labels']), r)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

import pandas as pd

from sklearn.metrics import average_precision_score, precision_recall_curve



from keras.models import Sequential

from keras import layers

from sklearn.metrics import f1_score

 



def create_nn(X_train):

    input_dim = X_train.shape[1]  # Number of features





    model = Sequential()

    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', 

                  optimizer='adam', 

                  metrics=['accuracy'])

    return model



def custom_cross_val_score(model, X, labels, scoring, cv): 

    scores = []



    for train_index,test_index in cv.split(X,labels):

        x_train,x_test=X[train_index],X[test_index]

        y_train,y_test=labels[train_index],labels[test_index]



        model=create_nn(x_train)

        model.fit(x_train, y_train,epochs=20)



        y_pred = model.predict_classes(x_test)

        score = f1_score(y_pred, y_test, average='macro')

        scores.append(score)



    return scores



def find_best_models(X, labels, rep_name):

    neural_network = KerasClassifier(build_fn=create_nn, 

                                 epochs=10, 

                                 batch_size=100, 

                                 verbose=0)

    models = [

      RandomForestClassifier(n_estimators=20, max_depth=3, random_state=0),

      LinearSVC(),

      LogisticRegression(random_state=0),

      SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None),

      KNeighborsClassifier(n_neighbors=3),

      neural_network

    ]

    CV = 5

    cv_df = pd.DataFrame(index=range(CV * len(models)))

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

    entries = []

    max_acc = -9

    best_model = ""

    for model in models:

        model_name = model.__class__.__name__

        accuracies = []

        if model_name == "KerasClassifier":

            accuracies = custom_cross_val_score(model, X, labels, scoring='f1_macro', cv=sss)

        else:

            accuracies = cross_val_score(model, X, labels, scoring='f1_macro', cv=sss)

        if max(accuracies) > max_acc:

            max_acc = max(accuracies)

            best_model = model

        for fold_idx, accuracy in enumerate(accuracies):

            entries.append((model_name, fold_idx, accuracy))

        print("{} with {}".format(model_name, str(accuracies)))



    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'F1-score macro'])

    

    return cv_df, best_model



def plot_model_perf(cv_df, rep_name):

    # plotting

    plt.figure( figsize=(18, 10), dpi=200, facecolor='w', edgecolor='k')



    # configure plot

    SMALL_SIZE = 6

    MEDIUM_SIZE = 10

    BIGGER_SIZE = 20



    plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes

    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title

    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels

    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



    sns.boxplot(x='model_name', y='F1-score macro', data=cv_df)

    sns.stripplot(x='model_name', y='F1-score macro', data=cv_df, 

                size=8, jitter=True, edgecolor="gray", linewidth=4)

    plt.xlabel("Model", fontsize=15)

    plt.xticks(fontsize=11)

    plt.yticks(fontsize=11)

    plt.ylim([0.0, 1.0])

    plt.ylabel("F1-score macro", fontsize=11)

    plt.title("Model selection - binary classification\nPredict General Case Announcement\n"+rep_name, fontsize = 15)



    #plt.figure( figsize=(8, 11), dpi=200, facecolor='w', edgecolor='k')

    plt.savefig("model_selection." + rep_name + ".f1_score.png")



# =============================

# TRAIN MODELS 

# =============================

cv_scores = {}

best_models = {}

for r in reps:

    cv_scores[r], best_model = find_best_models(reps[r], y, r)

    plot_model_perf(cv_scores[r], r)

    best_models[r] = best_model