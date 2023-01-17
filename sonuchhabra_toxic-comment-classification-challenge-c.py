import sys
sys.getrecursionlimit()
sys.setrecursionlimit(5000)
sys.getrecursionlimit()
# Output all code in a chunk
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# importing required libraries and functions
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # regular expression
from nltk import word_tokenize, PorterStemmer # natural language toolkit
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

# performs exactly same as OneVsRestClassifier, using that instead
## from skmultilearn.problem_transform import BinaryRelevance

# not using as removed string.punctuations using re.sub function
## import string

import os
print(os.listdir("../input"))
# download nltk packages
# nltk.download()
# reading data
train = pd.read_csv("../input/train.csv", nrows=40000)
test = pd.read_csv("../input/test.csv")
# verifying data
train.comment_text.head()
test.comment_text.head()
len(train)
len(test)
# creating train-validation split
X_train, X_val, y_train, y_val = train_test_split(train.comment_text, train.iloc[:,2:8], test_size=0.3, random_state=19)
X_test = test.comment_text
# creating function to normalize text
def normalize(text):
    # recognizing new line characters and tab spaces and substituting it with space
    norm_text = re.sub(r'\n|\t', ' ', text)
    # recognizing time values
    norm_text = re.sub(r'[0-9]{1,2}:[0-9][0-9]', 'time_value', norm_text) # example 5:13pm and 05:13pm
    # recognizing date values
    norm_text = re.sub(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', 'date_value', norm_text) # example 2018-03/05 and 04/03-2018
    norm_text = re.sub(r'[0-9]{1,4}[ ,][A-Za-z]{3,10}[ ,][0-9]{1,4}', 'date_value', norm_text) # example 9 june 2009 and 9 June 2009
    # substitute characters not required by nothing, removing unrequired characters
    norm_text = re.sub(r'[^A-Za-z_ ]', ' ', norm_text)
    # removing multiple space values
    norm_text = re.sub(r' +', ' ', norm_text)
    # removing trailing spaces from front and back and converting all text to lowercase
    norm_text = norm_text.strip().lower()
    return norm_text
# creating stemmer object of PorterStemmer function
stemmer = PorterStemmer()

# writing stem_tokens function to perform stemming on tokens
def stem_tokens(tokens, stemmer): # tokens example: ['today', 'is', 'a', 'good', 'day']
    stemmed = [stemmer.stem(word) for word in tokens]
    return stemmed
# processing text as follows
# tokenize words in each comment
# remove stopwords or words upto lenght of 3 characters
# stem words using the stem_tokens function we created above
def text_process(text): # text is a single sentence; for example: 'today is a good day'
    temp_tokens = word_tokenize(text)

    # using alternative to removing stopwords of english
    ## tokens = [word for word in temp_tokens if len(word) > 3]
    
    # removing english stopwords, code was commented to save computation time
    nostop_tokens = [word for word in temp_tokens if word not in stopwords.words('english')]
    
    stems = stem_tokens(nostop_tokens, stemmer)
    return ' '.join(stems)
# lenght of stopword of english
len(stopwords.words('english'))
stopwords.words('english')[:10]
# preparing training text to pass in count vectorizer
corpus = []
for text in X_train:
    text = normalize(text)
    text = text_process(text)
    corpus.append(text)
# build Count Vectorizer, to convert a collection of text documents to a matrix of token counts
count_vect = CountVectorizer(ngram_range=(1,2))
X_train_counts = count_vect.fit_transform(corpus)
# build TFIDF Transformer, to transform a count matrix to a normalized tf or tf-idf representation
# tfidf - term frequency inverse document frequency
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# verifing data
# print(X_train_counts.toarray())
# verifing data
# print(X_train_tfidf.toarray())
# checking how much text is transformed
temp = pd.DataFrame({'Before': X_train, 'After': corpus})
print(temp.sample(10))
# preparing validation text to pass in count vectorizer
X_val_set = []
for text in X_val:
    text = normalize(text)
    text = text_process(text)
    X_val_set.append(text)

# tranforming validation data using count vectorizer followed by tfidf transformer
X_val_counts = count_vect.transform(X_val_set)
X_val_tfidf = tfidf_transformer.transform(X_val_counts)
# preparing test text to pass in count vectorizer
X_test_set = []
for text in X_test:
    text = normalize(text)
    text = text_process(text)
    X_test_set.append(text)

# tranforming validation data using count vectorizer followed by tfidf transformer
X_test_counts = count_vect.transform(X_test_set)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
# creating dictionary to store prediction results
result_test = dict()
result_val = dict()
# Multinomial Naive Bayes Model
MNB_classifier = OneVsRestClassifier(MultinomialNB())
grid_values = {'estimator__alpha': [0.001, 0.01, 0.1, 1.0, 10, 100]}
MNB_model = GridSearchCV(MNB_classifier, param_grid = grid_values, scoring = 'roc_auc')
MNB_model.fit(X_train_tfidf, y_train)
print('Accurary of Multinomial Naive Bayes Classifier on Training Data: {:.3f}' .format(MNB_model.score(X_train_tfidf, y_train)))
print('Accurary of Multinomial Naive Bayes Classifier on Validation Data: {:.3f}' .format(MNB_model.score(X_val_tfidf, y_val)))
print('Grid best parameter (max. accuracy): ', MNB_model.best_params_)
print('Grid best score (accuracy): ', MNB_model.best_score_)
result_test['Multinomial_NB'] = MNB_model.predict_proba(X_test_tfidf)
result_val['Multinomial_NB'] = MNB_model.predict_proba(X_val_tfidf)
result_test['Multinomial_NB'].sum(axis=0)
# Logistic Regression Model, part 1
log_model = OneVsRestClassifier(LogisticRegression())
log_model.get_params().keys()
# Logistic Regression Model, part 2
grid_values = {'estimator__C': [0.3, 1.0, 30.0]}
log_grid = GridSearchCV(log_model, param_grid = grid_values, scoring = 'roc_auc')
log_grid.fit(X_train_tfidf, y_train)
print('Accurary of Logistic Regression Classifier on Training Data: {:.3f}' .format(log_grid.score(X_train_tfidf, y_train)))
print('Accurary of Logistic Regression Classifier on Validation Data: {:.3f}' .format(log_grid.score(X_val_tfidf, y_val)))
print('Grid best parameter (max. accuracy): ', log_grid.best_params_)
print('Grid best score (accuracy): ', log_grid.best_score_)
result_test['Logistic_Regression'] = log_grid.predict_proba(X_test_tfidf)
result_val['Logistic_Regression'] = log_grid.predict_proba(X_val_tfidf)
# storing results of SVM Classifier as our result
y_test = result_test['Logistic_Regression']
type(y_test)
# combining final results with the original test data set
output = pd.DataFrame(y_test, columns = train.columns[2:8], index = test.index)
output = pd.concat([test, output], axis=1)
output.head()
# verifing data
output.sample(20)
# verifing select random case, as per index from above code chunk
output.iloc[5902,:]
output.comment_text[5902]
# quick summary for training, validation and test set respectively
y_train.sum(axis=0)
y_val.sum(axis=0)
output.iloc[:,2:8].sum(axis=0)
#ngrams, is it unigram or bigram or mix?
#alpha parameter for Naive Bayes
#truncatesvd
#precision recall
#visualizations
my_submission = output.drop(['comment_text'], axis = 1, inplace = False)
type(output)
my_submission.to_csv('submission.csv', index=False)

