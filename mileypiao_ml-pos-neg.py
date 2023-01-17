#pip install afinn
#pip install vaderSentiment
#pip install shap
#nltk packages

import nltk

from nltk import word_tokenize

from nltk.stem import WordNetLemmatizer 

from nltk.corpus import stopwords

from string import punctuation



#Pandas

import pandas as pd

import numpy as np

#enable display of all columns in notebook

pd.options.display.max_columns = 999 

np.random.seed(12345)



#re

import re



#sklearn

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import r2_score



#string

import string





#sentiment packages

from textblob import TextBlob



#Used in creating ngrams

import math



#xgboost

import shap

import xgboost as xgb



#plt

import matplotlib.pyplot as plt

#enables display of plots in notebook

%matplotlib inline 



#stats

from scipy import stats
#from google.colab import drive

#drive.mount("/content/gdrive")
 ##%cd /content/gdrive/My Drive/ML
##train = pd.read_csv("train.csv")

##test = pd.read_csv("test.csv")

##sample_submission = pd.read_csv("sample_submission.csv")

train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

sample_submission = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
train[train['text'].isna()]
train.drop(314, inplace = True)

train = train.reset_index(drop = True)
train.info()

train.head()
#Prepare column names for xgboost

train.columns = train.columns.str.strip()

test.columns = test.columns.str.strip()
#Define a function to calculate jaccard score

def jaccard(str1,str2):

    a=str1.lower().split(" ")

    b=str2.lower().split(" ")

    c=set(a)&set(b)

    prop=len(c)/(len(a)+len(b)-len(c))

    return(prop)
#To save time, remove neutral from the train set.

# pos_train = train[train['sentiment'] == 'positive']

# neg_train = train[train['sentiment'] == 'negative']

neutral_train = train[train['sentiment'] == 'neutral']

train = train[train['sentiment'] != 'neutral']

neutral_test = test[test['sentiment'] == 'neutral']

test = test[test['sentiment'] != 'neutral']
train.head()
test.head()
# Create jaccard score for texts in train set with neutral sentiment

neutral_train.apply(lambda x: jaccard(x['text'], x['selected_text']), axis = 1).hist()
def remove_url(text):

    url_pattern = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", flags=re.UNICODE)

    return url_pattern.sub(r'', text)
train['text']=train['text'].apply(lambda x:remove_url(x))

test['text']=test['text'].apply(lambda x:remove_url(x))
#Check the distribution of proportion of selected_text versus text_n_words in train set

train['text_n_words'] = train['text'].apply(str).apply(lambda x: len(x.split(" ")))

train['sel_text_n_words'] = train['selected_text'].apply(str).apply(lambda x: len(x.split(" ")))

train['prop_sel_text_len'] = train['sel_text_n_words']/train['text_n_words']

train['prop_sel_text_len'].hist()
#Number of string in text

def find_str(text):

  result = re.split('[!?,.]',text)

  count = len(result)

  return count



#List of prepositions

prep = ['about', 'below', 'excepting', 'off', 'toward', 'above', 'beneath', 'on', 'under', 'across', 'from','onto',

'underneath', 'after','between', 'in', 'out', 'until', 'against', 'beyond' , 'outside', 'up' , 'along', 'but', 'inside','over',

'upon', 'among','by','past', 'around', 'concerning', 'regarding', 'with', 'at', 'despite','into', 'since', 'within',

'down', 'like', 'through','without', 'before', 'during', 'near', 'throughout', 'behind', 'except', 'of', 'to', 'for']



def preposition(sentence):

  words = sentence.split()

  prep_num = 0

  for x in prep:

      prep_num += words.count(x)

  return prep_num
#Create features for train set before running ngrams to save time

train['text_sent_blob'] = train['text'].apply(str).apply(lambda x: TextBlob(x.lower()).sentiment.polarity)#Blog score

##af = Afinn()

##train['text_sent_afinn'] = train['text'].apply(str).apply(lambda x: af.score(x.lower()))#Afinn score

##analyser = SentimentIntensityAnalyzer()

##train['text_sent_varder'] = train['text'].apply(str).apply(lambda x: analyser.polarity_scores(x.lower())["compound"])#Varder score

train['text_n_str'] = train['text'].apply(str).apply(lambda x: find_str(x.lower()))#Number of string in text

train['text_n_uq_words'] = train['text'].apply(str).apply(lambda x: len(np.unique(x.strip().split())))#Number of unique words

train['text_n_uq_chars'] = train['text'].apply(str).apply(lambda x: len(np.unique(list(x.replace(" ", "")))))#Number of unique characters

train['text_n_prepositions'] = train['text'].apply(str).apply(lambda x: preposition(x.lower()))#Number of prepositions
train.head()
#Create features for test set

test['text_n_words'] = test['text'].apply(str).apply(lambda x: len(x.split(" ")))

test['text_sent_blob'] = test['text'].apply(str).apply(lambda x: TextBlob(x.lower()).sentiment.polarity)#Blog score

##af = Afinn()

##test['text_sent_afinn'] = test['text'].apply(str).apply(lambda x: af.score(x.lower()))#Afinn score

##analyser = SentimentIntensityAnalyzer()

##test['text_sent_varder'] = test['text'].apply(str).apply(lambda x: analyser.polarity_scores(x.lower())["compound"])#Varder score

test['text_n_str'] = test['text'].apply(str).apply(lambda x: find_str(x.lower()))#Number of string in text

test['text_n_uq_words'] = test['text'].apply(str).apply(lambda x: len(np.unique(x.strip().split())))#Number of unique words

test['text_n_uq_chars'] = test['text'].apply(str).apply(lambda x: len(np.unique(list(x.replace(" ", "")))))#Number of unique characters

test['text_n_prepositions'] = test['text'].apply(str).apply(lambda x: preposition(x.lower()))#Number of prepositions
test.head()
#Create ngrams for a line

def create_ngrams(line):

  words = line['text'].split()

  # subsets = [words[i:j+1] for i in range(len(words)) for j in range(i,len(words))] #Create subset for whole train set

  subsets = [words[i:j+1] for i in range(len(words)) for j in range(i,int(math.ceil(len(words)/2)))] #Create subset for only the rows with subsetsLen(ngrams)/Len(original text) <= 0.5

  return subsets
%%time

#！！！！！！！！！！It takes 30 mins to run...

#Create ngrams subsets for train set

train_subsets = pd.DataFrame()

train_temp = pd.DataFrame()

for i in range(len(train)):

  ngrams_lines = create_ngrams(train.iloc[i])

  train_temp = pd.DataFrame([train.iloc[i]]*(len(ngrams_lines)))#Create the new lines and 

  train_temp['ngram'] = list(map(lambda x: " ".join(words for words in x),ngrams_lines))#Combine the new lines with their ngrams

  train_subsets = train_subsets.append(train_temp,ignore_index=True)#Append new lines with ngrams to train_subsets
train_temp = train

train_temp['ngram'] = train_temp['text']

train_subsets = train_subsets.append(train_temp,ignore_index=True) #Append original train set to get subsetsLen(ngrams)/Len(original text) = 1
#Update the two columns created before

train_subsets['sel_text_n_words'] = train_subsets['ngram'].apply(str).apply(lambda x: len(x.split(" ")))

train_subsets['prop_sel_text_len'] = train_subsets['sel_text_n_words']/train_subsets['text_n_words']
len(train_subsets)
train_subsets.head()
train_subsets.tail()
%%time

test_subsets = pd.DataFrame()

test_temp = pd.DataFrame()

for i in range(len(test)):

  ngrams_lines = create_ngrams(test.iloc[i])

  test_temp = pd.DataFrame([test.iloc[i]]*(len(ngrams_lines)))#Create the new lines and 

  test_temp['ngram'] = list(map(lambda x: " ".join(words for words in x),ngrams_lines))#Combine the new lines with their ngrams

  test_subsets = test_subsets.append(test_temp,ignore_index=True)#Append new lines with ngrams to test_subsets
test_temp = test

test_temp['ngram'] = test_temp['text']

test_subsets = test_subsets.append(test_temp,ignore_index=True) #Append original test set to get subsetsLen(ngrams)/Len(original text) = 1
#Update the two columns created before

test_subsets['sel_text_n_words'] = test_subsets['ngram'].apply(str).apply(lambda x: len(x.split(" ")))

test_subsets['prop_sel_text_len'] = test_subsets['sel_text_n_words']/test_subsets['text_n_words']
len(test_subsets)
test_subsets.head()
test_subsets.tail()
%%time

#It takes 6 mins



#Jaccard score

train_subsets['jaccard'] = train_subsets.apply(lambda x: jaccard(x['ngram'], x['selected_text']), axis = 1) # Create jaccard score for each row

#Blob score

train_subsets['sel_text_sent_blob'] = train_subsets['ngram'].apply(str).apply(lambda x: TextBlob(x.lower()).sentiment.polarity)

train_subsets['dif_text_sent_blob'] = train_subsets['text_sent_blob'] - train_subsets['sel_text_sent_blob']

#Afinn score

##train_subsets['sel_text_sent_afinn'] = train_subsets['ngram'].apply(str).apply(lambda x: af.score(x.lower()))

##train_subsets['dif_text_sent_afinn'] = train_subsets['text_sent_afinn'] - train_subsets['sel_text_sent_afinn']

#Varder score

##train_subsets['sel_text_sent_varder'] = train_subsets['ngram'].apply(str).apply(lambda x: analyser.polarity_scores(x.lower())["compound"])

##train_subsets['dif_text_sent_varder'] = train_subsets['text_sent_varder'] - train_subsets['sel_text_sent_varder']

#Proportion of number of string of ngrams

train_subsets['sel_text_n_str'] = train_subsets['ngram'].apply(str).apply(lambda x: find_str(x.lower()))

train_subsets['prop_sel_text_n_str'] = train_subsets['sel_text_n_str'] / train_subsets['text_n_str']

#Number of unique words of ngrams

train_subsets['sel_text_n_uq_words'] = train_subsets['ngram'].apply(str).apply(lambda x: len(np.unique(x.strip().split())))

train_subsets['prop_sel_text_n_uq_words'] =  train_subsets['sel_text_n_uq_words']/train_subsets['text_n_uq_words']

#Number of unique characters of ngrams

train_subsets['sel_text_n_uq_chars'] = train_subsets['ngram'].apply(str).apply(lambda x: len(np.unique(list(x.replace(" ", "")))))

train_subsets['prop_sel_text_n_uq_chars'] = train_subsets['sel_text_n_uq_chars']/train_subsets['text_n_uq_chars'] 

#Number of prepositions

train_subsets['sel_text_n_prepositions'] = train_subsets['ngram'].apply(str).apply(lambda x: preposition(x.lower()))

train_subsets['prop_sel_text_n_prepositions'] = train_subsets['sel_text_n_prepositions']/train_subsets['text_n_prepositions']
train_subsets.head()
%%time

#It takes 1 mins



#Blob score

test_subsets['sel_text_sent_blob'] = test_subsets['ngram'].apply(str).apply(lambda x: TextBlob(x.lower()).sentiment.polarity)

test_subsets['dif_text_sent_blob'] = test_subsets['text_sent_blob'] - test_subsets['sel_text_sent_blob']

#Afinn score

##test_subsets['sel_text_sent_afinn'] = test_subsets['ngram'].apply(str).apply(lambda x: af.score(x.lower()))

##test_subsets['dif_text_sent_afinn'] = test_subsets['text_sent_afinn'] - test_subsets['sel_text_sent_afinn']

#Varder score

##test_subsets['sel_text_sent_varder'] = test_subsets['ngram'].apply(str).apply(lambda x: analyser.polarity_scores(x.lower())["compound"])

##test_subsets['dif_text_sent_varder'] = test_subsets['text_sent_varder'] - test_subsets['sel_text_sent_varder']

#Proportion of number of string of ngrams

test_subsets['sel_text_n_str'] = test_subsets['ngram'].apply(str).apply(lambda x: find_str(x.lower()))

test_subsets['prop_sel_text_n_str'] = test_subsets['sel_text_n_str'] / test_subsets['text_n_str']

#Number of unique words of ngrams

test_subsets['sel_text_n_uq_words'] = test_subsets['ngram'].apply(str).apply(lambda x: len(np.unique(x.strip().split())))

test_subsets['prop_sel_text_n_uq_words'] = test_subsets['sel_text_n_uq_words']/test_subsets['text_n_uq_words'] 

#Number of unique characters of ngrams

test_subsets['sel_text_n_uq_chars'] = test_subsets['ngram'].apply(str).apply(lambda x: len(np.unique(list(x.replace(" ", "")))))

test_subsets['prop_sel_text_n_uq_chars'] = test_subsets['sel_text_n_uq_chars']/test_subsets['text_n_uq_chars']

#Number of prepositions

test_subsets['sel_text_n_prepositions'] = test_subsets['ngram'].apply(str).apply(lambda x: preposition(x.lower()))

test_subsets['prop_sel_text_n_prepositions'] = test_subsets['sel_text_n_prepositions']/test_subsets['text_n_prepositions']
test_subsets.head()
train_subsets.to_csv('train_subsets.csv',index = False)

test_subsets.to_csv('test_subsets.csv',index = False)
# %cd C:\Users\77548\Desktop\tweet-sentiment-extraction

# train_subsets = pd.read_csv("train_subsets_without_url.csv")

# test_subsets = pd.read_csv("test_subsets_without_url.csv")
train_subsets_pos =  train_subsets[train_subsets['sentiment'] == 'positive']

test_subsets_pos =  test_subsets[test_subsets['sentiment'] == 'positive']

train_subsets_neg =  train_subsets[train_subsets['sentiment'] == 'negative']

test_subsets_neg =  test_subsets[test_subsets['sentiment'] == 'negative']
train_subsets_pos['ID'] = train_subsets_pos.index + 1

test_subsets_pos['ID'] = test_subsets_pos.index + 1

train_subsets_neg['ID'] = train_subsets_neg.index + 1

test_subsets_neg['ID'] = test_subsets_neg.index + 1
def score_to_numeric(x):

    if x=='negative':

        return 0

    if x=='positive':

        return 1
train_subsets_pos['sentiment'] = train_subsets_pos['sentiment'].apply(score_to_numeric)

test_subsets_pos['sentiment'] = test_subsets_pos['sentiment'].apply(score_to_numeric)

train_subsets_neg['sentiment'] = train_subsets_neg['sentiment'].apply(score_to_numeric)

test_subsets_neg['sentiment'] = test_subsets_neg['sentiment'].apply(score_to_numeric)
y = 'jaccard'

# X = [name for name in train_subsets.columns if name not in [y, 'ID', 'textID','text', 'selected_text','ngram']

X = [name for name in train_subsets.columns if name not in [y, 'ID', 'textID','text', 'selected_text','ngram']]

print('y =', y)

print('X =', X)
train_subsets_pos[X + [y]].describe()
train_subsets_neg[X + [y]].describe()
np.random.seed(12345) # set random seed for reproducibility

split_ratio = 0.7     # 70%/30% train/test split



# execute split_pos

split_pos = np.random.rand(len(train_subsets_pos)) < split_ratio

train_pos = train_subsets_pos[split_pos]

test_pos = train_subsets_pos[~split_pos]



# summarize split_pos

print('Train_pos data rows = %d, columns = %d' % (train_pos.shape[0], train_pos.shape[1]))

print('Test_pos data rows = %d, columns = %d' % (test_pos.shape[0], test_pos.shape[1]))



# execute split_neg

split_neg = np.random.rand(len(train_subsets_neg)) < split_ratio

train_neg = train_subsets_neg[split_neg]

test_neg = train_subsets_neg[~split_neg]



# summarize split_neg

print('Train_neg data rows = %d, columns = %d' % (train_neg.shape[0], train_neg.shape[1]))

print('Test_neg data rows = %d, columns = %d' % (test_neg.shape[0], test_neg.shape[1]))
dtrain_pos = xgb.DMatrix(train_pos[X], train_pos[y])

dtest_pos = xgb.DMatrix(test_pos[X], test_pos[y])

dtrain_neg = xgb.DMatrix(train_neg[X], train_neg[y])

dtest_neg = xgb.DMatrix(test_neg[X], test_neg[y])
#Negative dataset model training

base_y_pos = train_pos[y].mean()



# tuning parameters

params = {

    'objective': 'reg:linear', 

    'bagging_fraction': 0.8768575337571937,

    'colsample_bytree': 0.9933592930641432,

    'feature_fraction': 0.816825176108506,

    'gamma': 0.05587328363633812,

    'learning_rate': 0.19879098664834996,

    'max_depth': 6,

    'min_child_samples': 9,

    'num_leaves': 7,

    'reg_alpha': 0.11806338517600543,

    'reg_lambda': 0.23269341544465222,

    'subsample': 0.6,

    'base_score': base_y_pos,                       # calibrate predictions to mean of y 

    'seed': 12345                               # set random seed for reproducibility

}



# watchlist is used for early stopping

watchlist_pos = [(dtrain_pos, 'train'), (dtest_pos, 'eval')]



# train model

xgb_model_pos = xgb.train(params,                   # set tuning parameters from above                   

                      dtrain_pos,                   # training data

                      1000,                     # maximum of 1000 iterations (trees)

                      evals=watchlist_pos,          # use watchlist for early stopping 

                      early_stopping_rounds=50, # stop after 50 iterations (trees) without increase in rmse 

                      verbose_eval=True)





#Negative dataset model training

base_y_neg = train_neg[y].mean()



# tuning parameters

params = {

    'objective': 'reg:linear', 

    'bagging_fraction': 0.8768575337571937,

    'colsample_bytree': 0.9933592930641432,

    'feature_fraction': 0.816825176108506,

    'gamma': 0.05587328363633812,

    'learning_rate': 0.19879098664834996,

    'max_depth': 6,

    'min_child_samples': 9,

    'num_leaves': 7,

    'reg_alpha': 0.11806338517600543,

    'reg_lambda': 0.23269341544465222,

    'subsample': 0.6,

    'base_score': base_y_neg,                       # calibrate predictions to mean of y 

    'seed': 12345                               # set random seed for reproducibility

}



# watchlist is used for early stopping

watchlist_neg = [(dtrain_neg, 'train'), (dtest_neg, 'eval')]



# train model

xgb_model_neg = xgb.train(params,                   # set tuning parameters from above                   

                      dtrain_neg,                   # training data

                      1000,                     # maximum of 1000 iterations (trees)

                      evals=watchlist_neg,          # use watchlist for early stopping 

                      early_stopping_rounds=50, # stop after 50 iterations (trees) without increase in rmse 

                      verbose_eval=True)

predictions_pos = xgb_model_pos.predict(dtest_pos)

predictions_neg = xgb_model_neg.predict(dtest_neg)
xgb.plot_importance(xgb_model_pos,importance_type='weight')

xgb.plot_importance(xgb_model_neg,importance_type='weight')
dtest_subsets_pos = xgb.DMatrix(test_subsets_pos[X])

dtest_subsets_neg = xgb.DMatrix(test_subsets_neg[X])
prediction_test_pos = xgb_model_pos.predict(dtest_subsets_pos)

prediction_test_neg = xgb_model_neg.predict(dtest_subsets_neg)
test_subsets_pos['Jaccard'] = prediction_test_pos

test_subsets_neg['Jaccard'] = prediction_test_neg
test_subsets_pos.head()
test_subsets_neg.head()
test_submission_pos = test_subsets_pos.sort_values('Jaccard', ascending=False).drop_duplicates(['textID'])

test_submission_neg = test_subsets_neg.sort_values('Jaccard', ascending=False).drop_duplicates(['textID'])
test_submission_pos = test_submission_pos[['textID','ngram']]

test_submission_neg = test_submission_neg[['textID','ngram']]

test_submission_pos = test_submission_pos.rename(columns = {'ngram':'selected_text'})

test_submission_neg = test_submission_neg.rename(columns = {'ngram':'selected_text'})

test_submission = test_submission_pos.append(test_submission_neg)

neutral_submission = neutral_test[['textID','text']]

neutral_submission = neutral_submission.rename(columns = {'text':'selected_text'})

submission = test_submission.append(neutral_submission,ignore_index=True)

sample_submission = sample_submission.drop(columns=['selected_text'])

sample_submission = pd.merge(sample_submission,submission,on='textID',how='left')
sample_submission.to_csv('submission.csv', index = False)