# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



import nltk

import xgboost as xgb



from fuzzywuzzy import fuzz



color = sns.color_palette()

%matplotlib inline



eng_stopwords = set(nltk.corpus.stopwords.words('english'))
raw_df = pd.read_csv("../input/train.csv", encoding="utf8")

df = raw_df.set_index("id")

df = df.sample(n=100000, random_state=7)

print (df[df["is_duplicate"]==1].shape)

print (df.shape)
# lower ==> tokenize ==> rm stop words

from nltk.stem.lancaster import LancasterStemmer

st = LancasterStemmer()



def nlp_preprocess(sent):

    try:

        #words = nltk.wordpunct_tokenize(sent.lower())

        words = sent.lower().replace("?", "").split()

        #words = [st.stem(w) for w in words]

    except:

        words = []

    words = [w for w in words if w not in eng_stopwords]

    return words

    

def preprocess(sent):

    q = nlp_preprocess(sent) # q = ["w1", "w2", ..]

    return q



df["q1"] = df["question1"].apply(preprocess)

df["q2"] = df["question2"].apply(preprocess)

df.head()
# find same words



def find_same_words(x):

    return list( set(x["q1"]) & set(x["q2"]) )



df["same_words"] = df.apply(find_same_words, axis=1)

df.head()
# feature 1: unigram

df["f_nSameWords"] = df["same_words"].apply(lambda x: len(x))



def prop_of_same_words_1(x):

    return x["f_nSameWords"]*1.0 / (len(x[1])+1.0)

def prop_of_same_words_2(x):

    return x["f_nSameWords"]*1.0 / max( (len(set(x[1])|set(x[2]))) , 1.0 )



df["f_ratioSameWords"] = df[["f_nSameWords", "q1", "q2"]].apply(prop_of_same_words_2, axis=1)

df.head()
plt.figure(figsize=(12,6))

sns.boxplot(x="is_duplicate", y="f_nSameWords", data=df)

plt.xlabel('Is duplicate', fontsize=12)

plt.ylabel('Common unigram ratio', fontsize=12)

plt.show()



plt.figure(figsize=(12,6))

sns.boxplot(x="is_duplicate", y="f_ratioSameWords", data=df)

plt.xlabel('Is duplicate', fontsize=12)

plt.ylabel('Common unigram ratio', fontsize=12)

plt.show()
# feature 2: 2-gram words

def get_twogram(que):

    return [i for i in nltk.ngrams(que, 2)]



def get_same_twogram(row):

    return list( set(row["twogram1"]) & set(row["twogram2"]) )



def prop_of_same_twogram(row):

    return row["f_nSameTwoGram"]*1.0 / max( (len(set(row["twogram1"]) | set(row["twogram2"]))), 1.0 )



df["twogram1"] = df["q1"].apply(get_twogram)

df["twogram2"] = df["q2"].apply(get_twogram)

df["same_twogram"] = df.apply(get_same_twogram, axis=1)



df["f_nSameTwoGram"] = df.apply(lambda x: len(x["same_twogram"]), axis=1)

df["f_ratioSameTwogram"] = df.apply(prop_of_same_twogram, axis=1)

df.head()
plt.figure(figsize=(12,6))

sns.boxplot(x="is_duplicate", y="f_nSameTwoGram", data=df)

plt.xlabel('Is duplicate', fontsize=12)

plt.ylabel('Common twograms ratio', fontsize=12)

plt.show()



plt.figure(figsize=(12,6))

sns.boxplot(x="is_duplicate", y="f_ratioSameTwogram", data=df)

plt.xlabel('Is duplicate', fontsize=12)

plt.ylabel('Common twograms ratio', fontsize=12)

plt.show()
# reduce the # of positive samples in training data

def resample(df):

    df_train_neg = df[df["is_duplicate"]==0]

    df_train_pos = df[df["is_duplicate"]==1]

    ratio_inc = (df_train_pos.shape[0]/0.17 - df.shape[0])*1.0 / df_train_neg.shape[0]

    if ratio_inc > 1:

        df_inc = df_train_neg.sample(frac=(ratio_inc-1.0))

        df_train_neg_inc = pd.concat([df_train_neg, df_inc])

    else:

        df_train_neg_inc = df_train_neg.sample(frac=ratio_inc)

    df = pd.concat([df_train_neg, df_train_neg_inc, df_train_pos]).sample(frac=1.0) # concat and shuffle

    return df
#=============================================#

#               train a model                 #

#=============================================#



n_valid = 20000

features = ["f_nSameWords", "f_ratioSameWords"]

features.extend(["f_nSameTwoGram", "f_ratioSameTwogram"])



df_need = df[features+["is_duplicate"]]

df_valid = df_need.iloc[0:n_valid, ]

df_train = df_need.iloc[n_valid:, ]

df_valid = resample(df_valid)

df_train = resample(df_train)

print (df_valid.shape)

print (df_train.shape)



X_valid = df_valid[features].iloc[0:n_valid, ]

X_train = df_train[features].iloc[n_valid:, ]



y_valid = df_valid["is_duplicate"].iloc[0:n_valid, ]

y_train = df_train["is_duplicate"].iloc[n_valid:, ]



dtrain = xgb.DMatrix( X_train, label=y_train)

dvalid = xgb.DMatrix( X_valid, label=y_valid)



evallist  = [(dtrain,'train'), (dvalid,'test')]



param = {'max_depth':4, 'eta':0.05, 'silent':1, 'objective':'binary:logistic' }

param['nthread'] = 12

param['eval_metric'] = 'logloss'



param["min_child_weight"] = 1

param["colsample_bytree"] = 0.7

param["subsample"] = 0.7

param["seed"] = 71



num_round = 300

bst = xgb.train( param.items(), dtrain, num_round, evallist, early_stopping_rounds=100, verbose_eval=10 )



xgb.plot_importance(bst)
# sampling from wrong predicted samples

pred = bst.predict(dvalid)
# do predict

#df_test = pd.read_csv("../input/test.csv", encoding="utf8")
#df_test["q1"] = df_test["question1"].apply(preprocess)

#df_test["q2"] = df_test["question2"].apply(preprocess)

#df_test["same_words"] = df_test.apply(find_same_words, axis=1)

#df_test["f_nSameWords"] = df_test["same_words"].apply(lambda x: len(x))

#df_test["f_ratioSameWords"] = df_test[["f_nSameWords", "q1", "q2"]].apply(prop_of_same_words_2, axis=1)
#df_test["twogram1"] = df_test["q1"].apply(get_twogram)

#df_test["twogram2"] = df_test["q2"].apply(get_twogram)

#df_test["same_twogram"] = df_test.apply(get_same_twogram, axis=1)

#df_test["f_nSameTwoGram"] = df_test.apply(lambda x: len(x["same_twogram"]), axis=1)

#df_test["f_ratioSameTwogram"] = df_test.apply(prop_of_same_twogram, axis=1)
#=============================================#

#                    predict                  #

#=============================================#



#X_test = df_test[features]

#dtest = xgb.DMatrix( X_test )



#ypred = bst.predict(dtest)



#df_test["is_duplicate"] = ypred

#df_test[["test_id", "is_duplicate"]].to_csv("xgb_starter.csv", index=False)
import random



que1 = df["question1"].iloc[0:n_valid, ]

que2 = df["question2"].iloc[0:n_valid, ]

q1 = df["q1"].iloc[0:n_valid, ]

q2 = df["q2"].iloc[0:n_valid, ]



r = random.sample(range(n_valid), 100)

for i in r:

    print (y_valid.iloc[i], pred[i], q1.iloc[i], q2.iloc[i], que1.iloc[i], que2.iloc[i])