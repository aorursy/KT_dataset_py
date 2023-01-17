!pip install underthesea

!pip install wordcloud

!pip install lightgbm
import numpy as np

import pandas as pd

import pylab as plt

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import cohen_kappa_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import pickle



from imblearn.over_sampling import SMOTE

from nltk.corpus import stopwords



import io

import requests

import nltk

nltk.download('punkt')

nltk.download('stopwords')



from underthesea import word_tokenize



import wordcloud

import matplotlib.pyplot as plt

import gc



import lightgbm as lgb

%matplotlib inline
def tokenize(text):

    return [word for word in word_tokenize(text.lower()) if word not in stopwords.words('english')]
def choose_vectorizer(option):

    if option == 'generate':

        vectorizer = TfidfVectorizer(tokenizer = tokenize, min_df=5, max_df= 0.8, max_features=10000, sublinear_tf=True)

    elif option == 'load':

        vectorizer = TfidfVectorizer(vocabulary = pickle.load(open('vocabulary.pkl', 'rb')),  min_df=5, max_df= 0.8, max_features=10000, sublinear_tf=True)

    

    return vectorizer
data = pd.read_csv('../input/kpdl-data/train_remove_noise.csv')

print(data.head(2))
X_train, X_test, y_train, y_test = train_test_split(data['Content'], data['Label'], test_size = .3, shuffle = False)
print(len(X_train), len(X_test), len(y_train), len(y_test))
# Wordcloud of training set

cloud = np.array(X_train).flatten()

plt.figure(figsize=(20,10))

word_cloud = wordcloud.WordCloud(

    max_words=200,background_color ="black",

    width=2000,height=1000,mode="RGB"

).generate(str(cloud))

plt.axis("off")

plt.imshow(word_cloud)
# Wordcloud of training set

cloud = np.array(X_test).flatten()

plt.figure(figsize=(20,10))

word_cloud = wordcloud.WordCloud(

    max_words=200,background_color ="black",

    width=2000,height=1000,mode="RGB"

).generate(str(cloud))

plt.axis("off")

plt.imshow(word_cloud)
%%time

options = ['generate', 'load']

# 0 to generate, 1 to load (choose wisely, your life depends on it!)

option = options[0] 

vectorizer = choose_vectorizer(option)



X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

    

if option == 'load':

    pickle.dump(vectorizer.vocabulary_, open('vocabulary.pkl', 'wb'))
label_enc = LabelEncoder()

enc = OneHotEncoder(handle_unknown='ignore')



y_ = label_enc.fit_transform(y_train)

y_train = y_

# y_train = np.zeros((len(y_), y_.max()+1))

# y_train[np.arange(len(y_)), y_] = 1



y_ = label_enc.fit_transform(y_test)

y_test = y_

# y_test = np.zeros((len(y_), y_.max()+1))

# y_test[np.arange(len(y_)), y_] = 1
def lgb_f1_score(y_hat, data):

    y_ = data.get_label()

    y_ = y_.astype(int)

    y_true = np.zeros((len(y_), 23))

    y_true[np.arange(len(y_)), y_] = 1

    y_true = y_true.reshape(-1)

    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities

    return 'f1', f1_score(y_true, y_hat), True
print("Starting LightGBM. Train shape: {}, test shape: {}".format(X_train.shape, X_test.shape))



# Cross validation model

folds = KFold(n_splits=5, shuffle=True, random_state=69)



# Create arrays and dataframes to store results

oof_preds = np.zeros(X_train.shape[0])

sub_preds = np.zeros(X_test.shape[0])



# k-fold

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):

    print("Fold %s" % (n_fold))

    train_x, train_y = X_train[train_idx], y_train[train_idx]

    valid_x, valid_y = X_train[valid_idx], y_train[valid_idx]

    print(train_y.shape)



    # set data structure

    lgb_train = lgb.Dataset(train_x,

                            label=train_y,

                            free_raw_data=False)

    lgb_test = lgb.Dataset(valid_x,

                           label=valid_y,

                           free_raw_data=False)



    params = {

        'objective' :'multiclass',

        'learning_rate' : 0.01,

        'num_leaves' : 75,

        'num_class':23,

        'feature_fraction': 0.64, 

        'bagging_fraction': 0.8, 

        'bagging_freq':1,

        'boosting_type' : 'gbdt',

    }



    reg = lgb.train(

        params,

        lgb_train,

        valid_sets=[lgb_train, lgb_test],

        valid_names=['train', 'valid'],

        num_boost_round=10000,

        verbose_eval=100,

        early_stopping_rounds=100,

        feval=lgb_f1_score

    )



    oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)

    sub_preds += reg.predict(X_test, num_iteration=reg.best_iteration) / folds.n_splits



    del reg, train_x, train_y, valid_x, valid_y

    gc.collect()