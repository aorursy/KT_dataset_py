import re

import spacy

import pickle

import string

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from datetime import datetime

from nltk.stem.snowball import SnowballStemmer

from keras.preprocessing.text import Tokenizer
nlp = spacy.load("en_core_web_sm")

train_file = "/kaggle/input/nlp-getting-started/train.csv"

objects = './objects/'



train_data = pd.read_csv(train_file)



train_data.keyword.fillna('0', inplace=True)

train_data.location.fillna('0', inplace=True)
def stemmer(doc):

    sb_stemmer = SnowballStemmer('english')

    stemmed_doc = ' '.join([sb_stemmer.stem(x) for x in doc.split()])

    return stemmed_doc



def tagger(tweet):

    doc = nlp(tweet)

    return len([x.pos_ for x in doc if x.pos_=='PROPN'])

        

def remove_punct(tweet):

    return ''.join([x for x in tweet if x not in string.punctuation])
# create a mapping for keywords

keyword_analysis = []



for value in train_data.keyword.unique():

    res = train_data[train_data['keyword'] == value].target.value_counts()  

    count = train_data[train_data['keyword'] == value].shape[0]

    try:    ones = res[1]

    except KeyError: ones = 0

    onesPerc = ones / count

    keyword_analysis.append([value, onesPerc])

    

    

keyword_analysis = np.array(keyword_analysis)

keyword_map = {}

for item in keyword_analysis:

    word = item[0]

    val1 = item[1]

    keyword_map[word] = float(val1)

train_data['mapped_keyword'] = train_data.keyword.map(keyword_map)

train_data[['mapped_keyword', 'target']].corr()

propn_counts = [tagger(x) for x in train_data.text]

train_data['propn_counts'] = propn_counts

train_data[['propn_counts', 'target']].corr()
def get_date_time_presence(tweets):

    DT_presence = []

    date_time_re = re.compile(r"(\d+[\/\-\:]+\d+[\/\-\:]?\d*)")

    year_re = re.compile(r"(20)[012][0-9]")

    months = ['january', 'february', 'march', 'april', 'june', 'july', 'august',

              'september', 'october', 'november', 'december', 'jan', 'feb', 'mar',

              'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec']

    for tweet in tweets:

        res1 = len(date_time_re.findall(tweet)) > 0

        res2 = len(year_re.findall(tweet)) > 0

        res4 = len([True for x in tweet.split() if x.lower() in months]) > 0

        DT_presence.append(res1 or res2 or res4)

    return DT_presence

    

DT_presence = get_date_time_presence(train_data.text) 

train_data['presence_date_time'] = DT_presence

train_data[['presence_date_time', 'target']].corr()
num_hashtags = [len(re.findall(r"\#\w*", x)) for x in train_data.text]

train_data['num_hashtags'] = num_hashtags

train_data[['num_hashtags', 'target']].corr()
number_counts = [len(re.findall(r"\d+[,?\d*]*", x)) for x in train_data.text]

train_data['number_counts'] = number_counts

train_data[['number_counts', 'target']].corr()
# [ '/' '#' '@' ':' ] are allowed

# they occur too frequently in urls, hashtags and username tags

def count_special_chars(tweets):

    SP_CH_COUNT = []

    debug_output = []

    allowed_chars = string.ascii_letters+string.digits+ "@?_)'&! "

    for tweet in tweets:

        debug_output.append(''.join([x for x in tweet.strip() if x not in allowed_chars]))

        SP_CH_COUNT.append(len([x for x in tweet.strip() if x not in allowed_chars]))

        

    return SP_CH_COUNT, debug_output

    

SP_CH_COUNT, _debug = count_special_chars(train_data.text)

train_data['special_count'] = SP_CH_COUNT

train_data[['special_count', 'target']].corr()
def clean_tweets(tweets):



    # replace all urls with a URL token

    url_removed = [re.sub(r"http[s]?:\/{2}[^\s]*", ' URL ', x) for x in tweets]

    url_removed = [re.sub(r"pic.twitter.com\/\w*", ' URL ', x) for x in url_removed]

    # replace usernames : replace tokens beginning with '@' with USERNAME token

    username_removed = [re.sub(r"@\w*", ' USERNAME ', x) for x in url_removed]

    # replace hash [ ' # ' ] with ' ' 

    hash_removed = [x.replace('#', ' ') for x in username_removed]

    # replace date/time with DATETIME token

    datetime_removed = [re.sub(r'(\d+[\/\-\:]+\d+[\/\-\:]?\d*)', ' DATETIME ', x) for x in hash_removed]

    datetime_removed = [re.sub(r"(20)[012][0-9]", ' DATETIME ', x) for x in datetime_removed]

    # replace numbers with NUM token

    numbers_replaced = [re.sub(r"\d+[,?\d*]*", ' NUM ', x ) for x in datetime_removed]

    # remove all punctuations

    punct_removed = [remove_punct(x) for x in numbers_replaced]

    # strip whitespaces

    stripped_tweets = [' '.join(x.split()) for x in punct_removed]

    

    stemmed_tweets = [stemmer(x) for x in stripped_tweets]

    # hashtags = [' '.join([x for x in sample]) for sample in _hashtags]

    

    return stemmed_tweets









cleaned_tweets = clean_tweets(train_data.text)
############################################





tokenizer = Tokenizer()

tokenizer.fit_on_texts(cleaned_tweets)

word_cnt = tokenizer.word_counts

words_occurence = {} # counts the number of words that have appeared once, twice, thrice etc.

for i in range(10):

    words_occurence[str(i+1)] = len([1 for v in word_cnt if word_cnt[v] == i+1 ])

words_occurence['10+'] = len([1 for v in word_cnt if word_cnt[v] > 10 ])

for item in words_occurence.items():

    print('Number of tokens appearing {} times: {}'.format(item[0], item[1]))

del(tokenizer)

############################################
VOCAB_SIZE = 3800



tokenizer = Tokenizer(VOCAB_SIZE)

tokenizer.fit_on_texts(cleaned_tweets)

X = tokenizer.texts_to_matrix(cleaned_tweets, mode='tfidf')
# settle on a value for k for PCA



VAR_TO_CAPTURE = 0.95



from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler





sc = StandardScaler()

# scaling tf-df features : important step for PCA

X_std = sc.fit_transform(X)





pca = PCA()

pca.fit(X_std)





# select a k

singular_values = pca.singular_values_

var_captured = 0

total = sum(singular_values)

k = 1

while(var_captured <= VAR_TO_CAPTURE):

    var_captured = sum(singular_values[:k]) / total

    k -=- 1



del(pca)



print('To retain 95% variance from tf-idf matrix, we retain top {} principle components'.format(k))
pca = PCA(n_components=k)

X_reduced = pca.fit_transform(X_std)


import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, classification_report, accuracy_score





X_new = np.hstack((X_reduced, train_data[['mapped_keyword', 'number_counts', 'propn_counts', 'special_count']].values))



x_train, x_val, y_train, y_val = train_test_split(X_new, train_data.target, test_size=0.15)





dtrain = xgb.DMatrix(data=x_train, label=y_train)

dval = xgb.DMatrix(data=x_val)





params ={

    'booster':'gbtree',

    'max_depth': 7,

    'objective': 'multi:softmax',

    'eval_metric' : 'mlogloss',

    'num_class': 2,

    'verbosity':0,

    'n_estimator':500}



bst = xgb.train(params, dtrain)

val_predictions_ = bst.predict(dval)

accuracy = accuracy_score(y_val, val_predictions_)

f1 = f1_score(y_val, val_predictions_)

print('accuracy: {}\tF1: {}'.format(accuracy, f1))

test_file = "/kaggle/input/nlp-getting-started/test.csv"

test_data = pd.read_csv(test_file)



test_data.keyword.fillna('0', inplace=True)

test_data.location.fillna('0', inplace=True)
### create new features for test data
# work with feature transformations



# add keyword mapping feature

test_data['mapped_keyword'] = test_data.keyword.map(keyword_map)



# add num numbers feature

number_counts_test = [len(re.findall(r"\d+[,?\d*]*", x)) for x in test_data.text]

test_data['number_counts'] = number_counts_test



# add proper noun count feature

propn_counts_test = [tagger(x) for x in test_data.text]

test_data['propn_counts'] = propn_counts_test



# add date-time presence feature

DT_presence_test = get_date_time_presence(test_data.text) 

test_data['presence_date_time'] = DT_presence_test



# add num_hashtag feature

num_hashtags_test = [len(re.findall(r"\#\w*", x)) for x in test_data.text]

test_data['num_hashtags'] = num_hashtags_test



SP_CH_COUNT_test, _debug = count_special_chars(test_data.text)

test_data['special_count'] = SP_CH_COUNT_test







# clean tweets and get text features

cleaned_tweets_test = clean_tweets(test_data.text)



X_test = tokenizer.texts_to_matrix(cleaned_tweets_test, mode='tfidf')

X_test_std = sc.transform(X_test)

X_test_reduced = pca.transform(X_test_std)





# put all features together

X_test_new = np.hstack((X_test_reduced, test_data[['mapped_keyword', 'number_counts', 'propn_counts', 'special_count']].values))
dtest = xgb.DMatrix(data=X_test_new)





test_predictions = bst.predict(dtest)

test_predictions = [int(x) for x in test_predictions ]





submission_df = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')



submission_df['target'] = test_predictions

filename = "submission_" + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".csv"

submission_df.to_csv(filename, index=False)
