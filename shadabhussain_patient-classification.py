# importing packages

import pandas as pd

import nltk

import numpy as np

import matplotlib.pyplot as plt

import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

from string import punctuation

from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier

import xgboost as xgb

from sklearn.naive_bayes import GaussianNB
# importing data and dropping row with null value in TRANS_CONV_TEXT column

train_df = pd.read_csv("../input/train.csv", encoding="latin") 

train_df = train_df[~ train_df["TRANS_CONV_TEXT"].isnull()].copy()

X = train_df.copy()

X = X.drop("Patient_Tag",axis=1)

X.head()
# Replacing null values in Title with empty string and concatenating Title and TRANS_CONV_TEXT

X["Title"] = X["Title"].fillna("")

X["combined"] = X["Title"]+" "+X["TRANS_CONV_TEXT"]

X["combined"] = X["combined"].apply(lambda x:x.strip())
# Removing Punctuations

new_text = []

for i in X["combined"]:

    string=""

    for j in i:

        if j in punctuation:

            pass

        else:

            string+=j.lower()

    new_text.append(string)

X["new_text"] = new_text

X["new_text"].head()
# Removing Stopwords

remove_sw = []

for i in X["new_text"]:

    string=""

    for j in i.split():

        if j in set(stopwords.words('english')) or j in ["i","im"]:

            pass

        else:

            string=string + " " +j

    remove_sw.append(string.strip())

X["remove_sw"] = remove_sw

X["remove_sw"].head()
# Finding meaningful root words

lemma = WordNetLemmatizer()

lemma_list = []

for i in X["remove_sw"]:

    string=""

    for j in i.split():

        string=string + " " +lemma.lemmatize(j,"v")

    lemma_list.append(string.strip())

X["lemma_list"] = lemma_list

X["lemma_list"].head()
X.shape
# Finding TFIDF

y = train_df["Patient_Tag"]

q = X["lemma_list"]

tf = TfidfVectorizer()

q = tf.fit_transform(q)

q.shape
# Storing TFIDF in vectorized column

X["vectorized"] = list(q.toarray())

X["vectorized"].head()
# Splitting data into train and test

train_X, test_X, train_y, test_y = train_test_split(X["vectorized"],y,test_size=0.3, random_state=42)

train_X.shape, train_y.shape, test_X.shape, test_y.shape
# Creating a Random Forest to test the Accuracy

rf = RandomForestClassifier()

rf.fit(list(train_X),train_y)

pred = rf.predict(list(test_X))

s = confusion_matrix(test_y,pred)

(s[0][0]+s[1][1])*100/(s[0][0]+s[0][1]+s[1][0]+s[1][1])
# Creating a naive bayes model

gb= GradientBoostingClassifier()

gb.fit(list(train_X),train_y)

pred = gb.predict(list(test_X))

s = confusion_matrix(test_y,pred)

(s[0][0]+s[1][1])*100/(s[0][0]+s[0][1]+s[1][0]+s[1][1])
# Creating XGBoost Classfier model

lr=0.2699999999999

a=34

md=19

ne=34

xg = xgb.XGBClassifier(learning_rate=lr,alpha=a,max_depth=12,n_estimators=ne)

xg.fit(np.array(list(train_X)),np.array(train_y))

pred = xg.predict(list(test_X))

s = confusion_matrix(test_y,pred)

print(s)

print((s[0][0]+s[1][1])*100/(s[0][0]+s[0][1]+s[1][0]+s[1][1]))

print(lr,a,md,ne)

df1 = pd.DataFrame()

df1["Patient_Tag"] = pred

df1.to_csv("submission.csv")

print("-----------------------------------------")
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import GridSearchCV 



xgb_model = XGBClassifier()



test_params = {

 'max_depth':[12, 19, 23]

}



model = GridSearchCV(estimator = xgb_model,param_grid = test_params)

model.fit(np.array(list(train_X)),np.array(train_y))

model.best_params_
# from scipy import stats

# from sklearn.model_selection import RandomizedSearchCV, KFold

# clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic')

# param_dist = {'n_estimators': stats.randint(150, 500),

#               'learning_rate': stats.uniform(0.01, 0.07),

#               'subsample': stats.uniform(0.3, 0.7),

#               'max_depth': [3, 4, 5, 6, 7, 8, 9],

#               'colsample_bytree': stats.uniform(0.5, 0.45),

#               'min_child_weight': [1, 2, 3]

#              }

# clf = RandomizedSearchCV(clf_xgb, param_distributions = param_dist, n_iter = 25, scoring = 'f1', error_score = 0, 

#                          verbose = 3, n_jobs = -1)



# numFolds = 5

# folds = KFold(shuffle = True, n_splits = numFolds)



# estimators = []

# results = np.zeros(len(X))

# score = 0.0

# for train_index, test_index in folds:

#     train_X, test_X = X[train_index], X[test_index]

#     train_y, test_y = y[train_index], y[test_index]

#     clf.fit(train_X, y_train)



#     estimators.append(clf.best_estimator_)

#     results[test_index] = clf.predict(test_X)

#     score += f1_score(test_y, results[test_index])

# score /= numFolds
# Cross validation for XGBoost

xg_train = xgb.DMatrix(list(train_X), label=train_y);

n_folds = 5

early_stopping = 10

params = {'eta': 0.27, 'max_depth': 5, 'subsample': 0.7, 'colsample_bytree': 0.7, 'objective': 'binary:logistic', 

          'seed': 99, 'silent': 1, 'eval_metric':'auc', 'nthread':4}

cv = xgb.cv(params, xg_train, 5000, nfold=n_folds, early_stopping_rounds=early_stopping, verbose_eval=1)
# Exporting file for test data

test_df = pd.read_csv("../input/test.csv", encoding="latin") 

# test_df = test_df[~ test_df["TRANS_CONV_TEXT"].isnull()].copy()

X = test_df.copy()

# X = X.drop("Patient_Tag",axis=1)

X["Title"] = X["Title"].fillna("")

X["combined"] = X["Title"]+" "+X["TRANS_CONV_TEXT"]

X["combined"] = X["combined"].apply(lambda x:x.strip())

new_text = []

for i in X["combined"]:

    string=""

    for j in i:

        if j in punctuation:

            pass

        else:

            string+=j

    new_text.append(string)

X["new_text"] = new_text

X["new_text"].head()



y = train_df["Patient_Tag"]

q = X["combined"]

# tf = TfidfVectorizer()

q = tf.transform(q)

X["vectorized"] = list(q.toarray())

test_df["Patient_Tag"] = rf.predict(list(q.toarray()))

test_df[["Index","Patient_Tag"]].to_csv("submission_rf.csv", index = False)