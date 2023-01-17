import os
print(os.listdir("../input"))
!head "../input/amazon_cells_labelled.txt"
#Q1 soln.
import pandas as pd
import csv

# imdb - movie review
imdb = pd.read_csv(filepath_or_buffer='../input/imdb_labelled.txt',
                   sep='\t',
                   names=['text', 'sentiment'],
                   quoting=csv.QUOTE_NONE)

# yelp - restaurant review
yelp = pd.read_csv(filepath_or_buffer='../input/yelp_labelled.txt',
                   sep='\t',
                   names=['text', 'sentiment'],
                   quoting=csv.QUOTE_NONE)

# combine yelp and imdb to form a training dataset
train_df = pd.concat([imdb, yelp], ignore_index=True)

# read test set - amazon - product review
test_df = pd.read_csv(filepath_or_buffer='../input/amazon_cells_labelled.txt',
                      sep='\t',
                      names=['text', 'sentiment'],
                      quoting=csv.QUOTE_NONE)
train_df.info()
test_df.info()
train_df.head()
test_df.head()
#Q2 soln.
# create a copy of train_df
train_df_clean = train_df
train_df_clean['text'] = train_df_clean['text'].str.lower()
train_df_clean.head()
train_df_clean.loc[train_df_clean['text'].str.contains('\w-\w'), 'text'] = train_df_clean['text'].str.replace('-', ' ')
train_df_clean['text'] = train_df_clean['text'].str.replace(r'[^\w\s\']', '')
train_df_clean.head()
train_words = ' '.join(train_df_clean['text']).split()
train_words_frequency = pd.Series(train_words).value_counts()
train_words_frequency[:20]
from nltk.corpus import stopwords

# build the stopwords list
stopwords_list = [r'\b' + stopword + r'\b' for stopword in stopwords.words('english')]
stopwords_list = '|'.join(stopwords_list)
stopwords_list
# replace stopwords with blank
train_df_clean['text'] = train_df_clean['text'].str.replace(stopwords_list, '')
train_df_clean['text'] = train_df_clean['text'].str.replace('[^\w\s]', '')
train_df_clean.loc[train_df['text'].str.contains('movie|film')][:10]
train_df_clean.loc[train_df['text'].str.contains('restaurant|food')][:10]
domain_spec_words = r'\bmovie\b|\bfilm\b|\brestaurant\b|\bfood\b'
train_df_clean['text'] = train_df_clean['text'].str.replace(domain_spec_words, '')

train_df_clean.head()
train_words_frequency[:10]
train_words_updated = ' '.join(train_df_clean['text']).split()
train_words_updated_frequency = pd.Series(train_words_updated).value_counts()
train_words_updated_frequency[:10]
from sklearn.feature_extraction.text import CountVectorizer

count_vectoriser = CountVectorizer()
train_words_count = count_vectoriser.fit_transform(train_df_clean['text'])
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
train_words_tfidf = tfidf_transformer.fit_transform(train_words_count)
from sklearn.naive_bayes import MultinomialNB

# fit the model to our training data
nb_classifier = MultinomialNB().fit(train_words_tfidf, train_df_clean['sentiment'])

# predict
y_pred_nb = nb_classifier.predict(tfidf_transformer.transform(count_vectoriser.transform(test_df['text'])))
from sklearn.linear_model import LogisticRegression

# fit the model to our training data
log_classifier = LogisticRegression(random_state=123).fit(train_words_tfidf, train_df_clean['sentiment'])

# predict
y_pred_log = log_classifier.predict(tfidf_transformer.transform(count_vectoriser.transform(test_df['text'])))
from sklearn.svm import LinearSVC

# fit the model to our training data
svm_classifier = LinearSVC(random_state=123).fit(train_words_tfidf, train_df_clean['sentiment'])

# predict
y_pred_svm = svm_classifier.predict(tfidf_transformer.transform(count_vectoriser.transform(test_df['text'])))
#Q4 soln.

from sklearn.metrics import accuracy_score, precision_score, recall_score
# accuracy score: number of correct prediction / number of test sentences
# precision score: number of correct positive prediction / number of positive prediction
# recall score: number of correct positive prediction / number of actual positive samples

initial_performance = pd.DataFrame(data={'naive_bayes' : [accuracy_score(test_df['sentiment'], y_pred_nb), 
                                                          precision_score(test_df['sentiment'], y_pred_nb),
                                                          recall_score(test_df['sentiment'], y_pred_nb)],
                                         'logistic_regression' : [accuracy_score(test_df['sentiment'], y_pred_log), 
                                                                  precision_score(test_df['sentiment'], y_pred_log),
                                                                  recall_score(test_df['sentiment'], y_pred_log)],
                                         'linear_svm' : [accuracy_score(test_df['sentiment'], y_pred_svm), 
                                                         precision_score(test_df['sentiment'], y_pred_svm),
                                                         recall_score(test_df['sentiment'], y_pred_svm)]},
                                   index=['accuracy', 'precision', 'recall'])

initial_performance
count_vectoriser = CountVectorizer(ngram_range=(1,2), min_df = 2)
tfidf_transformer = TfidfTransformer()

train_words_count = count_vectoriser.fit_transform(train_df_clean['text'])
train_words_tfidf = tfidf_transformer.fit_transform(train_words_count)
nb_classifier = MultinomialNB().fit(train_words_tfidf, train_df_clean['sentiment'])
y_pred_nb = nb_classifier.predict(tfidf_transformer.transform(count_vectoriser.transform(test_df['text'])))

log_classifier = LogisticRegression(random_state=123).fit(train_words_tfidf, train_df_clean['sentiment'])
y_pred_log = log_classifier.predict(tfidf_transformer.transform(count_vectoriser.transform(test_df['text'])))

svm_classifier = LinearSVC(random_state=123).fit(train_words_tfidf, train_df_clean['sentiment'])
y_pred_svm = svm_classifier.predict(tfidf_transformer.transform(count_vectoriser.transform(test_df['text'])))
updated_performance = pd.DataFrame(data={'naive_bayes_u' : [accuracy_score(test_df['sentiment'], y_pred_nb), 
                                                            precision_score(test_df['sentiment'], y_pred_nb),
                                                            recall_score(test_df['sentiment'], y_pred_nb)],
                                         'logistic_regression_u' : [accuracy_score(test_df['sentiment'], y_pred_log), 
                                                                    precision_score(test_df['sentiment'], y_pred_log), 
                                                                    recall_score(test_df['sentiment'], y_pred_log)],
                                         'linear_svm_u' : [accuracy_score(test_df['sentiment'], y_pred_svm), 
                                                           precision_score(test_df['sentiment'], y_pred_svm),
                                                           recall_score(test_df['sentiment'], y_pred_svm)]},
                                   index=['accuracy', 'precision', 'recall'])

pd.concat([initial_performance, updated_performance], axis=1)