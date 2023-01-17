## MANDATORY 
## [IMPORT NECESSARY LIBRARIES]

import gensim
import nltk
import sklearn
import pandas as pd
import numpy as np
import matplotlib

import re
import codecs
import itertools
import matplotlib.pyplot as plt

print ('DONE [IMPORT NECESSARY LIBRARIES]')
## MANDATORY 
## [ETL] Import Data

input_file = codecs.open("../input/nlp-starter-test/socialmedia_relevant_cols.csv", "r",encoding='utf-8', errors='replace')

# read_csv will turn CSV files into dataframes
questions = pd.read_csv(input_file)

#let's give names to the columns of our dataframe
questions.columns=['text', 'choose_one', 'class_label']

print ('DONE - [ETL] Import Data')
## [EDA] Explore Imported Data

questions.head()
#questions.head(10)
#questions.tail()
#questions.describe()
## MANDATORY 
## [PREPROCESS] Text Cleaning

def standardize_text(df, text_field):
    # normalize by turning all letters into lowercase
    df[text_field] = df[text_field].str.lower()
    # get rid of URLS
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"http\S+", "", elem))  
    return df

# call the text cleaning function
clean_questions = standardize_text(questions, "text")

print ('DONE - [PREPROCESS] Text Cleaning')
## [EDA] Explore Cleaned Data

clean_questions.head()
#clean_questions.tail()
## [EDA] Explore Class Labels

clean_questions.groupby("class_label").count()
## MANDATORY
## [PREPROCESS] Tokenize

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

clean_questions["tokens"] = clean_questions["text"].apply(tokenizer.tokenize)
clean_questions.head()
## [EDA] Explore words and sentences

all_words = [word for tokens in clean_questions["tokens"] for word in tokens]

sentence_lengths = [len(tokens) for tokens in clean_questions["tokens"]]

VOCAB = sorted(list(set(all_words)))

print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
# [EDA] Explore Vocabulary

# What are the words in the vocabulary
print (VOCAB[0:100])

# What are the most commonly occuring words
from collections import Counter
count_all_words = Counter(all_words)

# get the top 100 most common occuring words
count_all_words.most_common(100)
## MANDATORY 
## [CLASSIFY] Train test Split

from sklearn.model_selection import train_test_split

list_corpus = clean_questions["text"]
list_labels = clean_questions["class_label"]

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, random_state=40)

print("Training set: %d samples" % len(X_train))
print("Test set: %d samples" % len(X_test))
## [CLASSIFY] Check Data to be Trained

print (X_train[:10])
## [CLASSIFY] Check the Training Labels

print (y_train[:10])
## MANDATORY FOR BOW EMBEDDING
## [EMBEDDING] Tranform Tweets to BOW Embedding

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w+')

bow = dict()
bow["train"] = (count_vectorizer.fit_transform(X_train), y_train)
bow["test"]  = (count_vectorizer.transform(X_test), y_test)
print(bow["train"][0].shape)
print(bow["test"][0].shape)
## MANDATORY FOR TFIDF EMBEDDING
## [EMBEDDING] Transform Tweets to TFIDF Embedding

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w+')

tfidf = dict()
tfidf["train"] = (tfidf_vectorizer.fit_transform(X_train), y_train)
tfidf["test"]  = (tfidf_vectorizer.transform(X_test), y_test)

print(tfidf["train"][0].shape)
print(tfidf["test"][0].shape)
## MANDATORY FOR WORD2VEC EMBEDDING
## [EMBEDDING] Load Word2Vec Pretrained Corpus

word2vec_path = "../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

print ('DONE [Load Word2Vec Pretrained Corpus]')
## MANDATORY FOR WORD2VEC EMBEDDING
## [EMBEDDING] Get Word2Vec values for a Tweet

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_questions_tokens, generate_missing=False):
    embeddings = clean_questions_tokens.apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)

# Call the functions
embeddings = get_word2vec_embeddings(word2vec, clean_questions['tokens'])

print ('[EMBEDDING] Get Word2Vec values for a Tweet')
## MANDATORY FOR WORD2VEC EMBEDDING
## [CLASSIFY] Word2Vec Train Test Split

X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = train_test_split(embeddings, list_labels, 
                                                                    test_size=0.2, random_state=40)

w2v = dict()
w2v["train"] = (X_train_w2v, y_train_w2v)
w2v["test"]  = (X_test_w2v, y_test_w2v)

print ('DONE - [CLASSIFY] Word2Vec Train Test Split]')
## MANDATORY FOR LOGISTIC REGRESSION CLASSIFIER
## [CLASSIFY] Initialize Logistic Regression

from sklearn.linear_model import LogisticRegression

lr_classifier = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', random_state=40)

print ('DONE - [CLASSIFY] Initialize Logistic Regression')
## MANDATORY FOR SUPPORT VECTOR MACHINE CLASSIFIER
## [CLASSIFY] Initialize Support Vector Machine Classifier

from sklearn.svm import LinearSVC

lsvm_classifier = LinearSVC(C=1.0, class_weight='balanced', multi_class='ovr', random_state=40)

print ('[CLASSIFY] Initialize Support Vector Machine Classifier')
## MANDATORY FOR NAIVE BAYES CLASSIFIER
## [CLASSIFY] Initialize Naive Bayes
## NOTE - Does not work with Word2Vec Embedding

from sklearn.naive_bayes import MultinomialNB

nb_classifier = MultinomialNB()

print ('DONE - [CLASSIFY] Initialize Naive Bayes')
## MANDATORY FOR DECISION TREE
## [CLASSIFY] Initialize Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)

print ('DONE - [CLASSIFY] Initialize Decision Tree')
## MANDATORY 
## [EVALUATE] Prepare Metrics

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

print ('DONE - [EVALUATE] Prepare Metrics')
## MANDATORY
## [EVALUATE] Confusion Matrix

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt

print ('DONE - [EVALUATE] Confusion Matrix')
## MANDATORY 
## [EMBEDDING] CHOOSE EMBEDDING

embedding = bow                  # bow | tfidf | w2v

print ('DONE - [EMBEDDING] CHOOSE EMBEDDING')
## MANDATORY 
## [CLASSIFY] CHOOSE CLASSIFIER

classifier = lr_classifier     # lr_classifier | lsvm_classifier | nb_classifier| dt_classifier

print ('DONE - [CLASSIFY] CHOOSE CLASSIFIER')
## MANDATORY 
## [CLASSIFY] Train Classifier on Embeddings

classifier.fit(*embedding["train"])
y_predict = classifier.predict(embedding["test"][0])

print ('DONE - [CLASSIFY] Train Classifier on Embeddings')
## MANDATORY 
## [EVALUATE] Score chosen model

accuracy, precision, recall, f1 = get_metrics(embedding["test"][1], y_predict)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
## MANDATORY 
## [EVALUATE] Confusion matrix for chosen model

cm = confusion_matrix(embedding["test"][1], y_predict)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['Irrelevant','Disaster', 'Unsure'], normalize=False, title='Confusion Matrix')
plt.show()
## MANDATORY for COMPETITION
## [ETL] Load competition Test Data

test_X = pd.read_csv('../input/nlp-starter-test/test.csv')
test_corpus = test_X["Tweet"]
test_Id = test_X["Id"]

print ('DONE [ETL] Load competition Test Data')
## MANDATORY for COMPETITION
## [PREPROCESS] Tokenize Competition Data

# tokenize the test_corpus
test_corpus_tokens = test_corpus.apply(tokenizer.tokenize)

print ('[PREPROCESS] Tokenize Competition Data')
## MANDATORY for COMPETITION 
## [EMBEDDING] Apply Chosen Embeddings to the Tweets

vectorized_text = dict()
vectorized_text['test']  = (count_vectorizer.transform(test_corpus))  # see options in the above cell

print ('DONE - [EMBEDDING] Apply Chosen Embeddings to the Tweets')
## MANDATORY for COMPETITION  
## [CLASSIFY] Apply Chosen Classifier to the Embedding

embedding = vectorized_text                
classifier = lr_classifier     # lr_classifier | lsvm_classifier | nb_classifier | dt_classifier
predicted_sentiment = classifier.predict(embedding['test']).tolist()

print ('DONE - [CLASSIFY] Apply Chosen Classifier to the Embedding')
## MANDATORY for COMPETITION  
## [PREPARE SUBMISSION]


results = pd.DataFrame(
    {'Id': test_Id,
     'Expected': predicted_sentiment
    })

# Write your results for submission.
# Make sure to put in a meaningful name for the 'for_submission.csv 
# to distinguish your submission from other teams.

results.to_csv('for_submission_sample.csv', index=False)

print ('DONE - [PREPARE SUBMISSION]')