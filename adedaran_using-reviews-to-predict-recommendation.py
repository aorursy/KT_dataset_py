import nltk
import numpy as np
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from pandas import read_csv
import collections
dataset = read_csv('../input/reviews/Womens Clothing E-Commerce Reviews.csv')
reviews = dataset['Review Text'].astype('str')
recommend = dataset['Recommended IND']

print(dataset.shape)
print(reviews.head())
print(recommend.head())
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def review_prepare(review):
    review = review.lower()# lowercase text
    review = re.sub(REPLACE_BY_SPACE_RE," ",review)# replace REPLACE_BY_SPACE_RE symbols by space in text
    review = re.sub(BAD_SYMBOLS_RE,"",review)# delete symbols which are in BAD_SYMBOLS_RE from text
    review = re.sub(' +',' ',review)
    review = " ".join([word for word in review.split() if word not in STOPWORDS]) # delete stopwords from text
    return review

reviews_prepared = [review_prepare(review) for review in reviews]

print(reviews[3])
print(reviews_prepared[3])
counters_reviews = [collections.Counter(re.findall(r'\w+', review)) for review in reviews_prepared]
counter_all = sum(counters_reviews, collections.Counter())
most_common_word = sorted(counter_all.items(), key=lambda x: x[1] ,reverse=True)
print(counters_reviews[3])
print(most_common_word[:5])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(reviews_prepared,recommend,test_size=0.2)
print('Number of Recommend:',len(y_test[y_test==True]))
print('Number of Unrecommend:',len(y_test[y_test==False]))
DICT_SIZE = 5000
N = DICT_SIZE
WORDS_TO_INDEX = dict(most_common_word[:N])
ALL_WORDS = WORDS_TO_INDEX.keys()

def my_bag_of_words(review, words_to_index, dict_size):    
    words_list = list(sorted(words_to_index.keys()))
    result_vector = np.zeros(dict_size)
    for word in review.split():
        if word in words_list:
            result_vector[words_list.index(word)] +=1
    return result_vector
from scipy import sparse as sp_sparse

X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(review, WORDS_TO_INDEX, DICT_SIZE)) for review in X_train])
X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(review, WORDS_TO_INDEX, DICT_SIZE)) for review in X_test])
print('X_train shape ', X_train_mybag.shape)
print('X_test shape ', X_test_mybag.shape)
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(X_train, X_test):    
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), 
                                       max_df = 0.9, min_df=5, token_pattern='(\S+)')
    tfidf_vectorizer.fit(X_train)
    
    X_train =  tfidf_vectorizer.transform(X_train)
    X_test =  tfidf_vectorizer.transform(X_test)
    
    return X_train, X_test, tfidf_vectorizer.vocabulary_
X_train_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score, confusion_matrix, r2_score,classification_report
def train_classifier(X_train, y_train):
    classifier = LogisticRegression()
    
    classifier.fit(X_train, y_train)
    
    return classifier
classifier_mybag = train_classifier(X_train_mybag, y_train)
classifier_tfidf = train_classifier(X_train_tfidf, y_train)
y_predicted_labels_mybag = classifier_mybag.predict(X_test_mybag)
y_predicted_scores_mybag = classifier_mybag.decision_function(X_test_mybag)

y_predicted_labels_tfidf = classifier_tfidf.predict(X_test_tfidf)
y_predicted_scores_tfidf = classifier_tfidf.decision_function(X_test_tfidf)
print('r2 (bow):',r2_score(y_test,y_predicted_labels_mybag))
print('r2 (tfidf):',r2_score(y_test,y_predicted_labels_tfidf))
print('confusion_matrix (bow):\n',confusion_matrix(y_test,y_predicted_labels_mybag))
print('confusion_matrix (tfidf):\n',confusion_matrix(y_test,y_predicted_labels_tfidf))
print('classification_report (bow):\n',classification_report(y_test,y_predicted_labels_mybag))
print('classification_report (tfidf):\n',classification_report(y_test,y_predicted_labels_tfidf))
