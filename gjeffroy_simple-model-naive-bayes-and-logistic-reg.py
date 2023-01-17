import pandas as pd

import numpy as np
train = pd.read_csv('../input/train.csv', header=0, sep=',', quotechar='"', usecols=['target', 'comment_text'])

train['target_bool'] = np.where(train['target']>=0.5, 1, 0)

train[train['target_bool'] ==1].describe()
#select a subset of the train dataset accodirng to the portion (between 0 and 1)

portion = 1

indices = np.random.permutation(train.shape[0])

explore_idx = indices[:np.int(portion * train.shape[0])]

train_explore = train.iloc[explore_idx]

print("New shape is : " + str(train_explore.shape))
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')

import datetime

import string
# train_explore_clean = train_explore

# lemmatizer = nltk.WordNetLemmatizer()

# stop = stopwords.words("english")

# tokenizer = nltk.TreebankWordTokenizer()



# #cleaning

# tic = datetime.datetime.now()

# train_explore_clean['comment_text_clean'] = train_explore_clean['comment_text'].str.replace('[0-9]','') ### remove numbers

# tac = datetime.datetime.now()

# time = tac - tic

# print("remove number time" + str(time))

# tic = datetime.datetime.now()

# train_explore_clean['comment_text_clean']=train_explore_clean['comment_text_clean'].apply(lambda x : x.lower()) ### to lower case

# tac = datetime.datetime.now()

# time = tac - tic

# print("To lower time" + str(time))

# tic = datetime.datetime.now()

# train_explore_clean['comment_text_token'] = train_explore_clean['comment_text_clean'].apply(lambda x : tokenizer.tokenize(x)) ### tokenize

# tac = datetime.datetime.now()

# time = tac - tic

# print("Tokenize time" + str(time))

# tic = datetime.datetime.now()

# train_explore_clean['comment_text_token_no_stop'] = train_explore_clean['comment_text_token'].apply(lambda x: [item for item in x if item not in stop]) ### remove stopr words

# tac = datetime.datetime.now()

# time = tac - tic

# print("Remove stopword time" + str(time))

# tic = datetime.datetime.now()

# train_explore_clean['comment_text_token_no_stop'] = train_explore_clean['comment_text_token_no_stop'].apply(lambda x : " ".join(x))

# tac = datetime.datetime.now()

# time = tac - tic

# print("Join time" + str(time))







# train_explore_clean.head()
train_explore_clean = train_explore

lemmatizer = nltk.WordNetLemmatizer()

stops = stopwords.words("english")

tokenizer = nltk.TreebankWordTokenizer()



def remove_words_in_string(sentence, words):

    sentence = sentence

    remove_list = words

    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space

    sentence = sentence.translate(translator)

    word_list = sentence.split()

    return ' '.join([i for i in word_list if i not in remove_list])



# cleaning

# 

tic = datetime.datetime.now()

train_explore_clean['comment_text_clean'] = train_explore_clean['comment_text'].str.replace('[0-9]','') ### remove numbers

tac = datetime.datetime.now()

time = tac - tic

print("remove number time" + str(time))

tic = datetime.datetime.now()

train_explore_clean['comment_text_clean']=train_explore_clean['comment_text_clean'].apply(lambda x : x.lower()) ### to lower case

tac = datetime.datetime.now()

time = tac - tic

print("To lower time" + str(time))

tic = datetime.datetime.now()

train_explore_clean['comment_no_stop']=train_explore_clean['comment_text_clean'].apply(lambda sentence : remove_words_in_string(sentence, stops)) ### to lower case

tac = datetime.datetime.now()

time = tac - tic

print("remove stop words time : " + str(time))

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

# all parameters not specified are set to their defaults

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score, classification_report

from matplotlib import pyplot

from sklearn.naive_bayes import MultinomialNB
#parameters

max_df = 0.3

min_df = 0.003 

toxicity_threshold = 0.2

n_gram = (1, 3)



#processing

texts = train_explore_clean[train_explore_clean['target'] >toxicity_threshold ]['comment_no_stop']

tfidf = TfidfVectorizer(min_df=np.int(min_df * texts.shape[0]), max_df=max_df, ngram_range=n_gram)

features = tfidf.fit_transform(texts)

vectorizer = pd.DataFrame(

    features.todense(),

    columns=tfidf.get_feature_names()

)

print( "column names are " + str(vectorizer.columns.values), " Number of column is " + str (vectorizer.columns.shape))
# df = pd.DataFrame(train_explore_clean[train_explore_clean['comment_text_token_no_stop'].str.contains(u'could care less')][['target','comment_text']])

X = tfidf.transform(train_explore_clean['comment_no_stop'])

y=  train_explore_clean['target_bool']

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.05, random_state =42)
logisticRegr = LogisticRegression(C=2)

logisticRegr.fit(xTrain, yTrain)

predictions = logisticRegr.predict_proba(xTest)

predictions = pd.DataFrame(predictions)

# cm = metrics.confusion_matrix(yTest, predictions)

# print(cm)

# average_precision = average_precision_score(yTest, predictions)

# print('Average precision-recall score: {0:0.2f}'.format(

#       average_precision))

fpr, tpr, thresholds = roc_curve(yTest, predictions[1])

auc = roc_auc_score(yTest, predictions[1])

print('AUC: %.3f' % auc)

# plot no skill

pyplot.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

pyplot.plot(fpr, tpr, marker='.')

# show the plot

pyplot.show()

classification_report(yTest, predictions[1]>0.5)
Naive = MultinomialNB()

Naive.fit(xTrain,yTrain)



#Predict the response for test dataset

predictions = Naive.predict_proba(xTest)

predictions = pd.DataFrame(predictions)





# cm = metrics.confusion_matrix(yTest, predictions)

# print(cm)

# average_precision = average_precision_score(yTest, predictions)

# print('Average precision-recall score: {0:0.2f}'.format(

#       average_precision))

fpr, tpr, thresholds = roc_curve(yTest, predictions[1])

auc = roc_auc_score(yTest, predictions[1])

print('AUC: %.3f' % auc)

# plot no skill

pyplot.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

pyplot.plot(fpr, tpr, marker='.')

# show the plot

pyplot.show()

classification_report(yTest, predictions[1]>0.5)
test = pd.read_csv('../input/test.csv', header=0, sep=',', quotechar='"')

lemmatizer = nltk.WordNetLemmatizer()

stop = stopwords.words("english")

tokenizer = nltk.TreebankWordTokenizer()



#cleaning

tic = datetime.datetime.now()

test['comment_text_clean'] = test['comment_text'].str.replace('[0-9]','') ### remove numbers

tac = datetime.datetime.now()

time = tac - tic

print("remove number time" + str(time))

tic = datetime.datetime.now()

test['comment_text_clean']=test['comment_text_clean'].apply(lambda x : x.lower()) ### to lower case

tac = datetime.datetime.now()

time = tac - tic

print("To lower time" + str(time))

tic = datetime.datetime.now()

test['comment_no_stop']=test['comment_text_clean'].apply(lambda sentence : remove_words_in_string(sentence, stops)) ### to lower case

tac = datetime.datetime.now()

time = tac - tic

print("remove stop words time : " + str(time))

test = tfidf.transform(test['comment_no_stop'])

submission =  logisticRegr.predict_proba(test)

submission = pd.DataFrame(submission)

my_submission = pd.read_csv('../input/sample_submission.csv', header=0, sep=',', quotechar='"')

my_submission = pd.merge(submission, my_submission, left_index=True, right_index=True)

my_submission=my_submission[['id', 1]].rename(columns= {1 : 'prediction'})

my_submission.to_csv(r'submission.csv', index=False)

my_submission.head(10)