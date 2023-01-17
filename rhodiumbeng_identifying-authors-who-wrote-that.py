import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
# check the class distribution for the author label in train_df?
train_df['author'].value_counts()
# compute the character length for the rows and record these
train_df['text_length'] = train_df['text'].str.len()
# look at the histogram plot for text length
train_df.hist()
plt.show()
EAP = train_df[train_df['author'] =='EAP']['text_length']
EAP.describe()
EAP.hist()
plt.show()
MWS = train_df[train_df['author'] == 'MWS']['text_length']
MWS.describe()
MWS.hist()
plt.show()
HPL = train_df[train_df['author'] == 'HPL']['text_length']
HPL.describe()
HPL.hist()
plt.show()
# examine the text characters length in test_df and record these
test_df['text_length'] = test_df['text'].str.len()
test_df.hist()
plt.show()
# convert author labels into numerical variables
train_df['author_num'] = train_df.author.map({'EAP':0, 'HPL':1, 'MWS':2})
# Check conversion for first 5 rows
train_df.head()
train_df = train_df.rename(columns={'text':'original_text'})
train_df['text'] = train_df['original_text'].str[:700]
train_df['text_length'] = train_df['text'].str.len()
test_df = test_df.rename(columns={'text':'original_text'})
test_df['text'] = test_df['original_text'].str[:700]
test_df['text_length'] = test_df['text'].str.len()
X = train_df['text']
y = train_df['author_num']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# examine the class distribution in y_train and y_test
print(y_train.value_counts(),'\n', y_test.value_counts())
# import and instantiate CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# vect = CountVectorizer()
# vect = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b')
vect = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b|\,|\.|\;|\:')
# vect = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b|\,|\.|\?|\;|\:|\!|\'')
vect
# learn the vocabulary in the training data, then use it to create a document-term matrix
X_train_dtm = vect.fit_transform(X_train)
# examine the document-term matrix created from X_train
X_train_dtm
# transform the test data using the earlier fitted vocabulary, into a document-term matrix
X_test_dtm = vect.transform(X_test)
# examine the document-term matrix from X_test
X_test_dtm
def add_feature(X, feature_to_add):
    '''
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    '''
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')
from string import punctuation
X_train_chars = X_train.str.len()
X_train_punc = X_train.apply(lambda x: len([c for c in str(x) if c in punctuation]))
X_test_chars = X_test.str.len()
X_test_punc = X_test.apply(lambda x: len([c for c in str(x) if c in punctuation]))
X_train_dtm = add_feature(X_train_dtm, [X_train_chars, X_train_punc])
X_test_dtm = add_feature(X_test_dtm, [X_test_chars, X_test_punc])
X_train_dtm
X_test_dtm
# import and instantiate the Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb
# tune hyperparameter alpha = [0.01, 0.1, 1, 10, 100]
from sklearn.model_selection import GridSearchCV
grid_values = {'alpha':[0.01, 0.1, 1.0, 10.0, 100.0]}
grid_nb = GridSearchCV(nb, param_grid=grid_values, scoring='neg_log_loss')
grid_nb.fit(X_train_dtm, y_train)
grid_nb.best_params_
# set with recommended hyperparameters
nb = MultinomialNB(alpha=1.0)
# train the model using X_train_dtm & y_train
nb.fit(X_train_dtm, y_train)
# make author (class) predictions for X_test_dtm
y_pred_test = nb.predict(X_test_dtm)
# compute the accuracy of the predictions with y_test
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_test)
# compute the accuracy of training data predictions
y_pred_train = nb.predict(X_train_dtm)
metrics.accuracy_score(y_train, y_pred_train)
# look at the confusion matrix for y_test
metrics.confusion_matrix(y_test, y_pred_test)
# calculate predicted probabilities for X_test_dtm
y_pred_prob = nb.predict_proba(X_test_dtm)
y_pred_prob[:10]
# compute the log loss number
metrics.log_loss(y_test, y_pred_prob)
# import and instantiate the Logistic Regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=8)
logreg
# tune hyperparameter
grid_values = {'C':[0.01, 0.1, 1.0, 3.0, 5.0]}
grid_logreg = GridSearchCV(logreg, param_grid=grid_values, scoring='neg_log_loss')
grid_logreg.fit(X_train_dtm, y_train)
grid_logreg.best_params_
# set with recommended parameter
logreg = LogisticRegression(C=1.0, random_state=8)
# train the model using X_train_dtm & y_train
logreg.fit(X_train_dtm, y_train)
# make class predictions for X_test_dtm
y_pred_test = logreg.predict(X_test_dtm)
# compute the accuracy of the predictions
metrics.accuracy_score(y_test, y_pred_test)
# compute the accuracy of predictions with the training data
y_pred_train = logreg.predict(X_train_dtm)
metrics.accuracy_score(y_train, y_pred_train)
# look at the confusion matrix for y_test
metrics.confusion_matrix(y_test, y_pred_test)
# compute the predicted probabilities for X_test_dtm
y_pred_prob = logreg.predict_proba(X_test_dtm)
y_pred_prob[:10]
# compute the log loss number
metrics.log_loss(y_test, y_pred_prob)
# Learn the vocabulary in the entire training data, and create the document-term matrix
X_dtm = vect.fit_transform(X)
# Examine the document-term matrix created from X_train
X_dtm
# Add character counts features
X_chars = X.str.len()
X_punc = X.apply(lambda x: len([c for c in str(x) if c in punctuation]))
X_dtm = add_feature(X_dtm, [X_chars, X_punc])
X_dtm
# Train the Logistic Regression model using X_dtm & y
logreg.fit(X_dtm, y)
# Compute the accuracy of training data predictions
y_pred_train = logreg.predict(X_dtm)
metrics.accuracy_score(y, y_pred_train)
test = test_df['text']
# transform the test data using the earlier fitted vocabulary, into a document-term matrix
test_dtm = vect.transform(test)
# examine the document-term matrix from X_test
test_dtm
# Add character counts features
test_chars = test.str.len()
test_punc = test.str.count(r'\W')
test_dtm = add_feature(test_dtm, [test_chars, test_punc])
test_dtm
# make author (class) predictions for test_dtm
LR_y_pred = logreg.predict(test_dtm)
print(LR_y_pred)
# calculate predicted probabilities for test_dtm
LR_y_pred_prob = logreg.predict_proba(test_dtm)
LR_y_pred_prob[:10]
nb.fit(X_dtm, y)
# compute the accuracy of training data predictions
y_pred_train = nb.predict(X_dtm)
metrics.accuracy_score(y, y_pred_train)
# make author (class) predictions for test_dtm
NB_y_pred = nb.predict(test_dtm)
print(NB_y_pred)
# calculate predicted probablilities for test_dtm
NB_y_pred_prob = nb.predict_proba(test_dtm)
NB_y_pred_prob[:10]
alpha = 0.6
y_pred_prob = ((1-alpha)*LR_y_pred_prob + alpha*NB_y_pred_prob)
y_pred_prob[:10]
result = pd.DataFrame(y_pred_prob, columns=['EAP','HPL','MWS'])
result.insert(0, 'id', test_df['id'])
result.head()
# Generate submission file in csv format
result.to_csv('rhodium_submission_16.csv', index=False, float_format='%.20f')
