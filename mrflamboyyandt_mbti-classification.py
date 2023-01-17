# general imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

from scipy import stats



# nlp imports

import nltk

import nltk.data

from nltk.corpus import stopwords

import string

from gensim.models.word2vec import Word2Vec

from imblearn.combine import SMOTETomek





# model imports

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, RandomizedSearchCV, KFold

from sklearn import metrics

from sklearn.metrics import classification_report, accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.feature_extraction.text import CountVectorizer



# visualisation imports

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
from multiprocessing.dummy import Pool as ThreadPool 

pool = ThreadPool(4)
train = pd.read_csv('../input/train.csv')
train_df = train.copy()
test = pd.read_csv('../input/test.csv')
train.head()
dist = train['type'].value_counts()
plt.figure(figsize=(10,8),edgecolor='b')

plt.hlines(y=list(range(16)), xmin=0, xmax=dist, color='skyblue')

plt.plot(dist, list(range(16)), "D")

plt.yticks(list(range(16)), dist.index)

plt.xlabel('Number of Observations')

plt.ylabel('MBTI Type')

plt.title('Class Distribution')

plt.show()
def encode_type(df):

    # Function to convert 'type' column to four binary columns,

    # i.e. introvert becomes 0 and extrovert becomes 1.

    # Returns a dataframe.



    # grab the 'type' column in a list

    listy = list(df['type'])



    # separate the four letters into the four categories

    mind = [x[0] for x in listy]

    energy = [x[1] for x in listy]

    nature =  [x[2] for x in listy]

    tactics = [x[3] for x in listy]



    # create new columns

    df['mind'] = mind

    df['energy'] = energy

    df['nature'] = nature

    df['tactics'] = tactics



    # assigning integer values to categories

    df['mind'] = df['mind'].apply(lambda x: 1 if x == 'E' else 0)    

    df['energy'] = df['energy'].apply(lambda x: 1 if x == 'N' else 0)

    df['nature'] = df['nature'].apply(lambda x: 1 if x == 'T' else 0)

    df['tactics'] = df['tactics'].apply(lambda x: 1 if x == 'J' else 0)



    return df.drop('type', axis=1)
train = encode_type(train)
train.head()
def post_to_wordlist(post, remove_stopwords=False):

    # Function to convert a document to a sequence of words,

    # optionally removing stop words.  Returns a list of words.

    #

    # 1. Remove HTML

    post_text = post

    #

    # 2. Remove non-letters

    post_text = re.sub("[^a-zA-Z]", " ", post_text)

    #

    # 3. Convert words to lower case and split them

    words = post_text.lower().split()

    #

    # 4. Optionally remove stop words (false by default)

    if remove_stopwords:

        stops = set(stopwords.words("english"))

        words = [w for w in words if w not in stops]

    #

    # 5. Return a list of words

    return(words)
nltk.download('punkt')   





# Load the punkt tokenizer

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')





# Define a function to split a review into parsed sentences

def post_to_sentences(post, tokenizer=tokenizer, remove_stopwords=False):

    # Function to split a post into parsed sentences. Returns a 

    # list of sentences, where each sentence is a list of words

    #

    # 1. Use the NLTK tokenizer to split the paragraph into sentences

    raw_sentences = tokenizer.tokenize(post.strip())

    #

    # 2. Loop over each sentence

    sentences = []

    for raw_sentence in raw_sentences:

        # If a sentence is empty, skip it

        if len(raw_sentence) > 0:

            # Otherwise, call review_to_wordlist to get a list of words

            sentences.append(post_to_wordlist(raw_sentence, remove_stopwords))

    #

    # Return the list of sentences (each sentence is a list of words,

    # so this returns a list of lists

    return sentences
sentences = []  # Initialize an empty list of sentences



print("Parsing sentences from training set")

for post in train['posts']:

    sentences += post_to_sentences(post, tokenizer)



print("Parsing sentences from testing set")

for post in test['posts']:

    sentences += post_to_sentences(post, tokenizer)
# Check if sentences exist.

len(sentences)
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',

                    level=logging.INFO)



# Set values for various parameters

num_features = 1000   # Word vector dimensionality

min_word_count = 10   # Minimum word count

num_workers = 4       # Number of threads to run in parallel

context = 10          # Context window size

downsampling = 1e-3   # Downsample setting for frequent words



# Initialize and train the model (this will take some time)

print("Training model...")

w2v_model = Word2Vec(sentences, workers=num_workers,

                     size=num_features, min_count=min_word_count,

                     window=context, sample=downsampling)



# calling init_sims will make the model much more memory-efficient.

w2v_model.init_sims(replace=True)
w2v_model.most_similar('sad')
model = w2v_model
def makeFeatureVec(words, model, num_features):

    # Function to average all of the word vectors in a given

    # paragraph

    #

    # Pre-initialize an empty numpy array (for speed)

    featureVec = np.zeros((num_features,), dtype="float32")

    #

    nwords = 0.

    #

    # Index2word is a list that contains the names of the words in

    # the model's vocabulary. Convert it to a set, for speed

    index2word_set = set(model.wv.vocab.keys())

    #

    # Loop over each word in the post and, if it is in the model's

    # vocaublary, add its feature vector to the total

    for word in words:

        if word in index2word_set:

            nwords = nwords + 1.

            featureVec = np.add(featureVec, model[word])

    #

    # Divide the result by the number of words to get the average

    featureVec = np.divide(featureVec, nwords)

    return featureVec





def getAvgFeatureVecs(posts, model, num_features):

    # Given a set of posts (each one a list of words), calculate

    # the average feature vector for each one and return a 2D numpy array

    #

    # Initialize a counter

    counter = 0.

    #

    # Preallocate a 2D numpy array, for speed

    postFeatureVecs = np.zeros((len(posts), num_features), dtype="float32")

    #

    # Loop through the posts

    for post in posts:

        #

        # Print a status message every 1000th post

        if counter % 50000. == 0.:

            print("Post %d of %d" % (counter, len(posts)))

        #

        # Call the function (defined above) that makes average feature vectors

        postFeatureVecs[int(counter)] = makeFeatureVec(post, model,

                                                       num_features)

        #

        # Increment the counter

        counter = counter + 1.

    return postFeatureVecs
print("Creating average feature vecs for train posts")

# Initialize an empty list

clean_train_posts = []



# Use our multithreaded 

clean_train_posts = pool.map(post_to_wordlist, train['posts'])



trainDataVecs = getAvgFeatureVecs( clean_train_posts, model, num_features )



print("Creating average feature vecs for test posts")

clean_test_posts = []

clean_test_posts = pool.map(post_to_wordlist, test['posts'])



testDataVecs = getAvgFeatureVecs( clean_test_posts, model, num_features )
len(testDataVecs)
y = train[['mind', 'energy', 'nature', 'tactics']]

y_mind = train['mind']

y_energy = train['energy']

y_nature = train['nature']

y_tactics = train['tactics']

X = trainDataVecs

X_test = testDataVecs
test_ids = test.id
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LogisticRegression(solver='saga', class_weight='balanced', penalty='elasticnet')

abc = AdaBoostClassifier()

gbc = GradientBoostingClassifier()



params_lr = {'C': [1, 100], 'l1_ratio': [0.7, 0.9]}

params_abc = {'n_estimators': [16, 32]}

params_gbc =  {'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0]}
def train_tune(model, params, X, y, X_validate=X_validate):

    

    gs_mind = GridSearchCV(model, params, n_jobs=4, verbose=3)

    gs_energy = GridSearchCV(model, params, n_jobs=4, verbose=3)

    gs_nature = GridSearchCV(model, params, n_jobs=4, verbose=3)

    gs_tactics = GridSearchCV(model, params, n_jobs=4, verbose=3)

    

    mind = gs_mind.fit(X, y['mind'])

    energy = gs_energy.fit(X, y['energy'])

    nature = gs_nature.fit(X, y['nature'])

    tactics = gs_tactics.fit(X, y['tactics'])

    

    print('Best parameters for mind: {}'.format(gs_mind.best_params_))

    print('Best parameters for energy: {}'.format(gs_energy.best_params_))

    print('Best parameters for nature: {}'.format(gs_nature.best_params_))

    print('Best parameters for tactics: {}'.format(gs_tactics.best_params_))

    

    return (mind, energy, nature, tactics)
def score_model(mind, energy, nature, tactics):

    

    score_m = accuracy_score(y_validate['mind'], mind.predict(X_validate))

    score_e = accuracy_score(y_validate['energy'], energy.predict(X_validate))

    score_n = accuracy_score(y_validate['nature'], nature.predict(X_validate))

    score_t = accuracy_score(y_validate['nature'], nature.predict(X_validate))



    

    print('Mind accuracy: {}'.format(score_m))

    print('Energy accuracy: {}'.format(score_e))

    print('Nature accuracy: {}'.format(score_n))

    print('Tactics accuracy: {}'.format(score_t))

    

    return [score_m, score_e, score_n, score_t]

    
mind_lr, energy_lr, nature_lr, tactics_lr = train_tune(lr, params_lr, X = X_train, y=y_train)



print('Logistic Regression Scoring:')

lr_scores = score_model(mind_lr, energy_lr, nature_lr, tactics_lr)

mind_abc, energy_abc, nature_abc, tactics_abc = train_tune(abc, params_abc, X = X_train, y=y_train)



print('AdaBoost Classifier Scoring:')

abc_scores = score_model(mind_abc, energy_abc, nature_abc, tactics_abc)
mind_gbc, energy_gbc, nature_gbc, tactics_gbc = train_tune(gbc, params_gbc, X = X_train, y=y_train)



print('Gradient Boosting Classifier Scoring:')

gbc_scores = score_model(mind_gbc, energy_gbc, nature_gbc, tactics_gbc)
score_df = pd.DataFrame({

    'LogisticRegression': lr_scores,

    'AdaBoost': abc_scores,

    'GradientBoost': gbc_scores

}, index = ['mind', 'energy', 'nature', 'tactics'])
score_df
mind_preds = mind_abc.predict(X_test)

energy_preds = energy_abc.predict(X_test)

nature_preds = nature_lr.predict(X_test)

tactics_preds = tactics_lr.predict(X_test)
submit_df = pd.DataFrame({

    'id': test_ids,

    'mind': mind_preds,

    'energy': energy_preds,

    'nature': nature_preds,

    'tactics': tactics_preds

})
submit_df.to_csv('nowsubmit.csv', index=False)