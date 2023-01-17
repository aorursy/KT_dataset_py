#Import the important packages. 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.corpus import stopwords

from nltk import word_tokenize, ngrams

eng_stopwords = set(stopwords.words('english'))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from nltk.corpus import wordnet as wn

from nltk.corpus import wordnet

from nltk.corpus import stopwords

from nltk import word_tokenize, ngrams
train_data = pd.read_csv("../input/q_quora.csv")

print(train_data.shape)

train_data.head()
test_data = pd.read_csv("../input/test.csv")

print(test_data.shape)

test_data.head()
is_dup = train_data['is_duplicate'].value_counts()

sns.barplot(is_dup.index, is_dup.values)
train_q1 = train_data['question1']

train_q2 = train_data['question2']

train_q1_length = [len(i) for i in train_q1]

sns.histplot(train_q1_length)
from nltk.corpus import wordnet as wn

from nltk.corpus import wordnet

from nltk.corpus import stopwords

from nltk import word_tokenize, ngrams

import os

import numpy as np 

import pandas as pd 

import csv

import re

from collections import Counter



eng_stopwords = set(stopwords.words('english'))

nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}



def getWords(text):

    return re.compile('\w+').findall(text)



def getWords_0(text):

    temp = re.compile('\w+').findall(str(text))

    temp = " ".join(temp).lower()

    return(temp)



trans_Q1_AllWords = pd.Series(train_data['question1'].tolist()).astype(str)

df_train_Q1 = train_data['question1'].apply(lambda row: getWords_0(row))

words_1 = (" ".join(df_train_Q1)).split()

counts_Q1 = Counter(words_1)





trans_Q2_AllWords = pd.Series(train_data['question2'].tolist()).astype(str)

df_train_Q2 = train_data['question2'].apply(lambda row: getWords_0(row))

words_2 = (" ".join(df_train_Q2)).split()

counts_Q2 = Counter(words_2)



 

def feature_extraction(text):

    

    que1 = str(text['question1']).lower()

    que2 = str(text['question2']).lower()

    que1 = getWords(que1)

    que2 = getWords(que2)

    

    feature = []

    feature.extend([len(que1),len(que2)])

    

    Simplified_que1 = [word for word in que1 if word not in eng_stopwords]

    Simplified_que2 = [word for word in que2 if word not in eng_stopwords]

    Length_que1 = len(Simplified_que1)

    Length_que2 = len(Simplified_que2)

    feature.extend([Length_que1,Length_que2])



    Unique_que1 = [word for word in Simplified_que1 if word not in Simplified_que2]

    Unique_que2 = [word for word in Simplified_que2 if word not in Simplified_que1]

    feature.extend([len(Unique_que1),len(Unique_que2)])

    

    Unique_que1_Nouns = [word for word in Unique_que1 if word in nouns]

    Unique_que2_Nouns = [word for word in Unique_que2 if word in nouns]

    feature.extend([len(Unique_que1_Nouns), len(Unique_que2_Nouns)])

    

    # Tfdif

    df_Q1_1 = 0

    df_Q1_2 = 0

    df_Q2_1 = 0

    df_Q2_2 = 0

    for i in Unique_que1:

        df_Q1_1 = df_Q1_1 + counts_Q1[i] / len(words_1)

        df_Q1_2 = df_Q1_2 + counts_Q2[i] / len(words_2)

    for i in Unique_que2:

        df_Q2_1 = df_Q2_1 + counts_Q1[i] / len(words_1)

        df_Q2_2 = df_Q2_2 + counts_Q2[i] / len(words_2)

        

    feature.extend([df_Q1_1, df_Q1_2, df_Q2_1, df_Q2_2])



    return(feature)



df_train_Questions = train_data[['question1','question2']]

train_X = np.vstack( np.array(df_train_Questions.apply(lambda row: feature_extraction(row), axis=1)) )



train_Y = train_data['is_duplicate']



pos_train = train_X[train_Y == 1]

neg_train = train_X[train_Y == 0]

index = np.random.choice(len(pos_train), int(0.17 * 255027), replace = False)

pos_train = pos_train[index]

train_X = np.concatenate([pos_train, neg_train])

train_Y = np.concatenate([np.zeros(len(pos_train)) + 1, np.zeros(len(neg_train))])
train_X
# Here one simple example will show the basic way to find the best model. The RandomForest algorithm will be used here

from sklearn.ensemble import RandomForestClassifier

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from sklearn.model_selection import train_test_split, cross_val_score



# Here the train_X is the features which are generated from the raw data. Train_Y is the results

X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.2, random_state=1234)



def objective(space):

    clf = RandomForestClassifier(**space)

    logloss = cross_val_score(clf,train_X,train_Y).mean()

    print ("SCORE:", logloss)

    return{'loss':logloss, 'status': STATUS_OK }



space4rf = {

    'max_depth': hp.choice('max_depth', range(1,20)),

    'max_features': hp.choice('max_features', range(1,5)),

    'n_estimators': hp.choice('n_estimators', range(1,20)),

    'criterion': hp.choice('criterion', ["gini", "entropy"]),

    'scale': hp.choice('scale', [0, 1]),

    'normalize': hp.choice('normalize', [0, 1])

}



trials = Trials()

best = fmin(fn=objective,

            space=space4rf,

            algo=tpe.suggest,

            max_evals=10,

            trials=trials)



print (best)
# The code for the stacking. 



clfs = [RandomForestClassifier(n_estimators=100, criterion='gini'),

        RandomForestClassifier(n_estimators=100, criterion='entropy'),

        RandomForestClassifier(n_estimators=10, criterion='gini'),

        RandomForestClassifier(n_estimators=10, criterion='entropy'),

        ExtraTreesClassifier(n_estimators=100, criterion='gini'),

        ExtraTreesClassifier(n_estimators=100, criterion='entropy'),

        ExtraTreesClassifier(n_estimators=10, criterion='gini'),

        ExtraTreesClassifier(n_estimators=10, criterion='entropy'),

        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),

        KNeighborsClassifier(n_neighbors=5),

        KNeighborsClassifier(n_neighbors=10),

        GaussianNB(),

        LogisticRegression()]



n_folds = 5

skf = list(StratifiedKFold(train_Y, n_folds))



dataset_blend_train = np.zeros((train_X.shape[0], len(clfs)))

dataset_blend_test = np.zeros((test_X.shape[0], len(clfs)))

    

for j, clf in enumerate(clfs):

    print (j, clf)

    dataset_blend_test_j = np.zeros((test_X.shape[0], len(skf)))

    for i, (train, test) in enumerate(skf):

        print ("Fold", i)

        X_train = train_X[train]

        y_train = train_Y[train]

        X_test = train_X[test]

        y_test = train_Y[test]

        clf.fit(X_train, y_train)

        y_submission = clf.predict_proba(X_test)[:, 1]

        dataset_blend_train[test, j] = y_submission

        dataset_blend_test_j[:, i] = clf.predict_proba(test_X)[:, 1]

    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
kf = KFold(n_splits=5, shuffle=True, random_state=2016)



for dev_index, val_index in kf.split(range(dataset_blend_train_L2.shape[0])):

    

    params = {}

    params["objective"] = "binary:logistic"

    params['eval_metric'] = 'logloss'

    params["eta"] = 0.3

    params["subsample"] = 0.7

    params["min_child_weight"] = 2

    params["colsample_bytree"] = 0.7

    params["max_depth"] = 5

    params["silent"] = 1

    dev_X, val_X = dataset_blend_train_L2[dev_index,:], dataset_blend_train_L2[val_index,:]

    dev_y, val_y = train_Y[dev_index], train_Y[val_index]

    d_train = xgb.DMatrix(dev_X, label = dev_y)

    d_test = xgb.DMatrix(val_X, label = val_y)

    watchlist = [ (d_train,'train'), (d_test, 'test') ]

    bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

    break



# Prediction

dataset_blend_test = xgb.DMatrix(dataset_blend_test_L2)

data_XGB = xgb.DMatrix(test_X)

predict_y_XGB = bst.predict(data_XGB)

data_prediction = predict_y_XGB



def Prediction(training_data, testing_data, clfs):

    for i in clfs:

        temp_model = i.fit(training_data, train_Y)

        temp_prediction = temp_model.predict(testing_data)

        data_prediction = np.vstack([data_prediction,temp_prediction])

        print(i)

        

for i in clfs_prediction:

    temp_model = i.fit(dataset_blend_train_L2, train_Y)

    temp_prediction = temp_model.predict(dataset_blend_test_L2)

    data_prediction = np.vstack([data_prediction,temp_prediction])

    print(i)



sub = pd.DataFrame()

sub['test_id'] = df_test['test_id']

sub['is_duplicate'] = data_prediction

sub.to_csv("submission.csv", index=False)
#Import the important packages. 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.corpus import stopwords

from nltk import word_tokenize, ngrams

eng_stopwords = set(stopwords.words('english'))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from nltk.corpus import wordnet as wn

from nltk.corpus import wordnet

from nltk.corpus import stopwords

from nltk import word_tokenize, ngrams
train_data = pd.read_csv("../input/train.csv")

print(train_data.shape)

train_data.head()
test_data = pd.read_csv("../input/test.csv")

print(test_data.shape)

test_data.head()
is_dup = train_data['is_duplicate'].value_counts()

sns.barplot(is_dup.index, is_dup.values)
train_q1 = train_data['question1']

train_q2 = train_data['question2']

train_q1_length = [len(i) for i in train_q1]

sns.distplot(train_q1_length)
from nltk.corpus import wordnet as wn

from nltk.corpus import wordnet

from nltk.corpus import stopwords

from nltk import word_tokenize, ngrams

import os

import numpy as np 

import pandas as pd 

import csv

import re

from collections import Counter



eng_stopwords = set(stopwords.words('english'))

nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}



def getWords(text):

    return re.compile('\w+').findall(text)



def getWords_0(text):

    temp = re.compile('\w+').findall(str(text))

    temp = " ".join(temp).lower()

    return(temp)



trans_Q1_AllWords = pd.Series(train_data['question1'].tolist()).astype(str)

df_train_Q1 = train_data['question1'].apply(lambda row: getWords_0(row))

words_1 = (" ".join(df_train_Q1)).split()

counts_Q1 = Counter(words_1)





trans_Q2_AllWords = pd.Series(train_data['question2'].tolist()).astype(str)

df_train_Q2 = train_data['question2'].apply(lambda row: getWords_0(row))

words_2 = (" ".join(df_train_Q2)).split()

counts_Q2 = Counter(words_2)



 

def feature_extraction(text):

    

    que1 = str(text['question1']).lower()

    que2 = str(text['question2']).lower()

    que1 = getWords(que1)

    que2 = getWords(que2)

    

    feature = []

    feature.extend([len(que1),len(que2)])

    

    Simplified_que1 = [word for word in que1 if word not in eng_stopwords]

    Simplified_que2 = [word for word in que2 if word not in eng_stopwords]

    Length_que1 = len(Simplified_que1)

    Length_que2 = len(Simplified_que2)

    feature.extend([Length_que1,Length_que2])



    Unique_que1 = [word for word in Simplified_que1 if word not in Simplified_que2]

    Unique_que2 = [word for word in Simplified_que2 if word not in Simplified_que1]

    feature.extend([len(Unique_que1),len(Unique_que2)])

    

    Unique_que1_Nouns = [word for word in Unique_que1 if word in nouns]

    Unique_que2_Nouns = [word for word in Unique_que2 if word in nouns]

    feature.extend([len(Unique_que1_Nouns), len(Unique_que2_Nouns)])

    

    # Tfdif

    df_Q1_1 = 0

    df_Q1_2 = 0

    df_Q2_1 = 0

    df_Q2_2 = 0

    for i in Unique_que1:

        df_Q1_1 = df_Q1_1 + counts_Q1[i] / len(words_1)

        df_Q1_2 = df_Q1_2 + counts_Q2[i] / len(words_2)

    for i in Unique_que2:

        df_Q2_1 = df_Q2_1 + counts_Q1[i] / len(words_1)

        df_Q2_2 = df_Q2_2 + counts_Q2[i] / len(words_2)

        

    feature.extend([df_Q1_1, df_Q1_2, df_Q2_1, df_Q2_2])



    return(feature)



df_train_Questions = train_data[['question1','question2']]

train_X = np.vstack( np.array(df_train_Questions.apply(lambda row: feature_extraction(row), axis=1)) )



train_Y = train_data['is_duplicate']



pos_train = train_X[train_Y == 1]

neg_train = train_X[train_Y == 0]

index = np.random.choice(len(pos_train), int(0.17 * 255027), replace = False)

pos_train = pos_train[index]

train_X = np.concatenate([pos_train, neg_train])

train_Y = np.concatenate([np.zeros(len(pos_train)) + 1, np.zeros(len(neg_train))])
train_X
# Here one simple example will show the basic way to find the best model. The RandomForest algorithm will be used here

from sklearn.ensemble import RandomForestClassifier

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from sklearn.model_selection import train_test_split, cross_val_score



# Here the train_X is the features which are generated from the raw data. Train_Y is the results

X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.2, random_state=1234)



def objective(space):

    clf = RandomForestClassifier(**space)

    logloss = cross_val_score(clf,train_X,train_Y).mean()

    print ("SCORE:", logloss)

    return{'loss':logloss, 'status': STATUS_OK }



space4rf = {

    'max_depth': hp.choice('max_depth', range(1,20)),

    'max_features': hp.choice('max_features', range(1,5)),

    'n_estimators': hp.choice('n_estimators', range(1,20)),

    'criterion': hp.choice('criterion', ["gini", "entropy"])

}



trials = Trials()

best = fmin(fn=objective,

            space=space4rf,

            algo=tpe.suggest,

            max_evals=10,

            trials=trials)



print (best)
# The code for the stacking. It will take long time to train the model, but it works.  

import numpy as np

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor

from sklearn.model_selection import cross_val_predict, GridSearchCV, train_test_split

from sklearn.cross_validation import StratifiedKFold

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Ridge



clfs = [RandomForestClassifier(n_estimators=100, criterion='gini'),

        RandomForestClassifier(n_estimators=100, criterion='entropy'),

        RandomForestClassifier(n_estimators=10, criterion='gini'),

        RandomForestClassifier(n_estimators=10, criterion='entropy'),

        ExtraTreesClassifier(n_estimators=100, criterion='gini'),

        ExtraTreesClassifier(n_estimators=100, criterion='entropy'),

        ExtraTreesClassifier(n_estimators=10, criterion='gini'),

        ExtraTreesClassifier(n_estimators=10, criterion='entropy'),

        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),

        KNeighborsClassifier(n_neighbors=5),

        KNeighborsClassifier(n_neighbors=10),

        GaussianNB(),

        LogisticRegression()]



test_X = np.vstack( np.array(test_data.apply(lambda row: feature_extraction(row), axis=1)) )

X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y, test_size = 0.3)





n_folds = 5

skf = list(StratifiedKFold(train_Y, n_folds))



dataset_blend_train = np.zeros((train_X.shape[0], len(clfs)))

dataset_blend_test = np.zeros((test_X.shape[0], len(clfs)))

    

for j, clf in enumerate(clfs):

    print (j, clf)

    dataset_blend_test_j = np.zeros((test_X.shape[0], len(skf)))

    for i, (train, test) in enumerate(skf):

        print ("Fold", i)

        X_train = train_X[train]

        y_train = train_Y[train]

        X_test = train_X[test]

        y_test = train_Y[test]

        clf.fit(X_train, y_train)

        y_submission = clf.predict_proba(X_test)[:, 1]

        dataset_blend_train[test, j] = y_submission

        dataset_blend_test_j[:, i] = clf.predict_proba(test_X)[:, 1]

    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
import xgboost as xgb

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=2016)

for dev_index, val_index in kf.split(range(dataset_blend_train.shape[0])):

    

    params = {}

    params["objective"] = "binary:logistic"

    params['eval_metric'] = 'logloss'

    params["eta"] = 0.3

    params["subsample"] = 0.7

    params["min_child_weight"] = 2

    params["colsample_bytree"] = 0.7

    params["max_depth"] = 5

    params["silent"] = 1

    dev_X, val_X = dataset_blend_train[dev_index,:], dataset_blend_train[val_index,:]

    dev_y, val_y = train_Y[dev_index], train_Y[val_index]

    d_train = xgb.DMatrix(dev_X, label = dev_y)

    d_test = xgb.DMatrix(val_X, label = val_y)

    watchlist = [ (d_train,'train'), (d_test, 'test') ]

    bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

    break



# Prediction

dataset_blend_test = xgb.DMatrix(dataset_blend_test)

predict_y_XGB = bst.predict(dataset_blend_test)

data_prediction = predict_y_XGB



sub = pd.DataFrame()

sub['test_id'] = df_test['test_id']

sub['is_duplicate'] = data_prediction

sub.to_csv("submission.csv", index=False)