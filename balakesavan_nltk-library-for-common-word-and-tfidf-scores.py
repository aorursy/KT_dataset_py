# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#These are the additional libraries (the list above are included in the standard Kaggle image)

import nltk
#nltk.download() #not needed to run on Kaggle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
#import pickle #not needed to run on Kaggle, but used to store interim objects while working on a local machine
import xgboost as xgb
from sklearn.cross_validation import train_test_split

#reading in the data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
#function to identify common words in question pairs
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

def common_words(row):
    q1words = {}
    q2words = {}
    
    for word in word_tokenize(str(row['question1'])):
        if word not in stops:
            q1words[word] = 1
    for word in word_tokenize(str(row['question2'])):
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 'N/A'
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    return shared_words_in_q1
# function to score common words in question pairs
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

def nltk_word_match_share(row):
    q1words = {}
    q2words = {}
    
    for word in word_tokenize(str(row['question1'])):
        if word not in stops:
            q1words[word] = 1
    
    for word in word_tokenize(str(row['question2'])):
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R
#identifying common words in question pairs
train_common_words_tokenizer = df_train.apply(common_words, axis=1, raw=True)
test_common_words_tokenizer = df_test.apply(common_words, axis=1, raw=True) 
#scoring for common words in question pairs
train_word_match_tokenizer = df_train.apply(nltk_word_match_share, axis=1, raw=True)
test_word_match_tokenizer = df_test.apply(nltk_word_match_share, axis=1, raw=True)
#generating TFIDF score for the train corpus 
tf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = 'english')
corpus = train_qs
tfidf_matrix =  tf.fit_transform(corpus)
feature_names = tf.get_feature_names() 
phrase_scores = tf.idf_

names_scores = pd.DataFrame({'feature_names':feature_names})
names_scores['phrase_scores'] = pd.DataFrame(phrase_scores)

writer = pd.ExcelWriter("/home/Bala/Documents/NLP/names_scores.xlsx")
names_scores.to_excel(writer,'Sheet1')
writer.save()

scores1=[]
for i in range(0,404291):
    row_scores1 = 0
    for word in train_common_words_tokenizer[i]:
            if str.lower(word) in feature_names:
                row_scores1 = row_scores1 + (phrase_scores[feature_names.index(str.lower(word))])
                print(i)
                #print('row_scores', row_scores)
    scores1.append(row_scores1)
scores_tfidf_train_nltk = scores1

#generating TFIDF scores for the test corpus
#IMPORTANT: This code needs optimization/ vectorization
tf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = 'english')
corpus_test = test_qs
tfidf_matrix_test =  tf.fit_transform(corpus_test)
feature_names_test = tf.get_feature_names() 
phrase_scores_test = tf.idf_

names_scores_test = pd.DataFrame({'feature_names':feature_names_test})
names_scores_test['phrase_scores'] = pd.DataFrame(phrase_scores_test)

scores2=[]
for i in range(0,len(df_test)):
    row_scores2 = 0
    for word in test_common_words_tokenizer[i]:
            if str.lower(word) in feature_names_test:
                row_scores2 = row_scores2 + (phrase_scores_test[feature_names_test.index(str.lower(word))])
                print(i)
                #print('row_scores', row_scores)
    scores2.append(row_scores2)
#scores1_500000=scores1
scores_tfidf_test_nltk = pd.DataFrame(scores2)
scores_tfidf_test_nltk = scores2
#creating dataframes for train and test
x_train = pd.DataFrame()
x_test = pd.DataFrame()

x_train['nltk word match'] = train_word_match_tokenizer
x_train['nltk tfidf']= scores_tfidf_train_nltk

y_train = pd.DataFrame(df_train['is_duplicate'].values)

x_test['nltk word match'] = test_word_match_tokenizer
x_test['nltk tfidf']= scores_tfidf_test_nltk
# final data preparation - Split some of the data off for validation

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=3)
#model training
nltk_train = x_train.loc[:,['nltk word match', 'nltk tfidf']]
nltk_valid = x_valid.loc[:,['nltk word match', 'nltk tfidf']]

d_train = xgb.DMatrix(nltk_train, label=y_train)
d_valid = xgb.DMatrix(nltk_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
