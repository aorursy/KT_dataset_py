# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

from sklearn import preprocessing

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score 

from sklearn.ensemble import RandomForestClassifier

import xgboost

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

import warnings

np.random.seed(123)

warnings.filterwarnings('ignore')

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
train_dataset = pd.read_csv("/kaggle/input/steam-reviews/train.csv", delimiter=",")

train_dataset
test_dataset = pd.read_csv("/kaggle/input/steam-reviews-test-dataset/test.csv", delimiter=",")

test_dataset['user_suggestion'] = None

test_dataset
dataset = pd.concat([train_dataset, test_dataset], axis = 0)

dataset.reset_index(drop = True, inplace = True)

dataset
from matplotlib import pyplot as plt

import seaborn as sns
# Visualizing the variable - 'year'

plt.figure(figsize = (10,5))

plt.xticks(rotation=90)

sns.countplot(train_dataset['year'])
import re

def decontracted(phrase):

    # specific

    phrase = re.sub(r"won\'t", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"early access review", "early access review ", phrase)

    phrase = re.sub(r"\+", " + ", phrase) 

    phrase = re.sub(r"\-", " - ", phrase)     

    phrase = re.sub(r"/10", "/10 ", phrase)     

    phrase = re.sub(r"10/", " 10/", phrase)         

    return phrase





trial = "Hey I'm Yann, how're you and how's it going ? That's interesting: I'd love to hear more about it+info"

decontracted(trial)
from textblob import TextBlob

# Define function to lemmatize each word with its POS tag

def lemmatize_with_postag(sentence):

    sent = TextBlob(sentence)

    tag_dict = {"J": 'a', 

                "N": 'n', 

                "V": 'v', 

                "R": 'r'}

    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    

    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]

    return " ".join(lemmatized_list)



# Lemmatize

trial = "The striped bats are hanging on their feet for best"

lemmatize_with_postag(trial)
import re

def clean_reviews(lst):

    # remove URL links (httpxxx)

    lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")

    # remove special characters, numbers, punctuations (except for #)

    lst = np.core.defchararray.replace(lst, "[^a-zA-Z]", " ")

    # remove amp with and

    lst = np.vectorize(replace_pattern)(lst, "amp", "and")  

    # remove hashtags

    lst = np.vectorize(remove_pattern)(lst, "#[A-Za-z0-9]+")

    lst = np.vectorize(remove_pattern)(lst, "#[\w]*")    

    return lst

def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)        

    return input_txt

def replace_pattern(input_txt, pattern, replace_text):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, replace_text, input_txt)        

    return input_txt
# Applying pre-processing to user reviews

text2 = clean_reviews(list(dataset['user_review'].astype('str')))

text3 = [ta.lower() for ta in text2]

text4 = [''.join([i if ord(i) < 128 else ' ' for i in t]) for t in text3]

text5 = [decontracted(u) for u in text4]

text6 = [lemmatize_with_postag(u) for u in text5]

text6
dataset.loc[4, 'user_review']
text6[4]
# Word Level Count Vectorizer 

count_vect = CountVectorizer(analyzer='word', max_features = 1500, stop_words = "english")

countdf_user_review= count_vect.fit_transform(text6)

print("All tags are:")

print(count_vect.get_feature_names())

print("Matrix looks like")

print(countdf_user_review.shape)

print(countdf_user_review.toarray())
countdf_user_review_df = pd.DataFrame(data = countdf_user_review.toarray(), index = dataset.index)

countdf_user_review_df.columns = count_vect.get_feature_names()

countdf_user_review_df.head()
# Count Vectorizer for N-grams 

count_vect2 = CountVectorizer(analyzer='word', max_features = 1500, ngram_range=(2,3), stop_words = "english")

countdf_user_review2= count_vect2.fit_transform(text6)

print("All tags are:")

print(count_vect2.get_feature_names())

print("Matrix looks like")

print(countdf_user_review2.shape)

print(countdf_user_review2.toarray())
countdf_user_review_df2 = pd.DataFrame(data = countdf_user_review2.toarray(), index = dataset.index)

countdf_user_review_df2.columns = count_vect2.get_feature_names()

countdf_user_review_df2.head()
# Word level Tf-Idf

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer='word', max_features=1500, stop_words = "english")

tfidf_user_review = tfidf_vect.fit_transform(text6)

print("All tags are:")

print(tfidf_vect.get_feature_names())

print("Matrix looks like")

print(tfidf_user_review.shape)

print(tfidf_user_review.toarray())
tfidf_user_review_df = pd.DataFrame(data = tfidf_user_review.toarray(), index = dataset.index)

tfidf_user_review_df.columns = tfidf_vect.get_feature_names()

tfidf_user_review_df.head()
# Tf-Idf for N-grams

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect2 = TfidfVectorizer(analyzer='word', max_features=1500, ngram_range=(2,3), stop_words = "english")

tfidf_user_review2 = tfidf_vect2.fit_transform(text6)

print("All tags are:")

print(tfidf_vect2.get_feature_names())

print("Matrix looks like")

print(tfidf_user_review2.shape)

print(tfidf_user_review2.toarray())
tfidf_user_review_df2 = pd.DataFrame(data = tfidf_user_review2.toarray(), index = dataset.index)

tfidf_user_review_df2.columns = tfidf_vect2.get_feature_names()

tfidf_user_review_df2.head()
target = dataset['user_suggestion']
dataset2 = pd.concat([tfidf_user_review_df, tfidf_user_review_df2], axis=1)

dataset2
dataset2.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in dataset2.columns]
cols=pd.Series(dataset2.columns)



for dup in cols[cols.duplicated()].unique(): 

    cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]



# rename the columns with the cols list.

dataset2.columns=cols
target[~target.isnull()]
X = dataset2[~target.isnull()]

Y = target[~target.isnull()].astype(int)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,test_size=0.2)
import lightgbm as lgb



train_data = lgb.Dataset(X_Train, label = Y_Train)



#setting parameters for lightgbm

param = {'num_leaves':50, 'objective':'binary','max_depth':4,'learning_rate':.1,'max_bin':100}

param['metric'] = ['auc', 'binary_logloss']



lgbmodel = lgb.train(param, train_data, 500, verbose_eval=True)
from sklearn.metrics import roc_curve, auc



Y_Pred_Prob = lgbmodel.predict(X_Test)



Y_Pred = Y_Pred_Prob.copy()



Y_Pred[Y_Pred >= 0.5] = 1

Y_Pred[Y_Pred < 0.5] = 0



fpr, tpr, thresholds = roc_curve(np.array(Y_Test), np.array(Y_Pred_Prob))

roc_auc = auc(fpr, tpr)

print("ROC AUC of LightGBM  is {}".format(roc_auc))



from sklearn.metrics import f1_score

f1_score2 = f1_score(Y_Test, Y_Pred, average = "weighted")

print("F1 Score of LightGBM  is {}".format(f1_score2))



from sklearn.metrics import accuracy_score

acc = accuracy_score(Y_Test, Y_Pred)

print("Accuracy Score of LightGBM  is {}".format(acc))
dataset2 = pd.concat([countdf_user_review_df, countdf_user_review_df2], axis=1)

dataset2



dataset2.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in dataset2.columns]



cols=pd.Series(dataset2.columns)



for dup in cols[cols.duplicated()].unique(): 

    cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]



# rename the columns with the cols list.

dataset2.columns=cols



target[~target.isnull()]



X = dataset2[~target.isnull()]

Y = target[~target.isnull()].astype(int)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,test_size=0.2)
import lightgbm as lgb



train_data = lgb.Dataset(X_Train, label = Y_Train)



#setting parameters for lightgbm

param = {'num_leaves':50, 'objective':'binary','max_depth':4,'learning_rate':.1,'max_bin':100}

param['metric'] = ['auc', 'binary_logloss']



lgbmodel = lgb.train(param, train_data, 500, verbose_eval=True)
from sklearn.metrics import roc_curve, auc



Y_Pred_Prob = lgbmodel.predict(X_Test)



Y_Pred = Y_Pred_Prob.copy()



Y_Pred[Y_Pred >= 0.5] = 1

Y_Pred[Y_Pred < 0.5] = 0



fpr, tpr, thresholds = roc_curve(np.array(Y_Test), np.array(Y_Pred_Prob))

roc_auc = auc(fpr, tpr)

print("ROC AUC of LightGBM  is {}".format(roc_auc))



from sklearn.metrics import f1_score

f1_score2 = f1_score(Y_Test, Y_Pred, average = "weighted")

print("F1 Score of LightGBM  is {}".format(f1_score2))



from sklearn.metrics import accuracy_score

acc = accuracy_score(Y_Test, Y_Pred)

print("Accuracy Score of LightGBM  is {}".format(acc))
from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1048)

predictions = np.zeros((len(X_Test), ))

feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_Train.values, Y_Train.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(X_Train.iloc[trn_idx,:], label=Y_Train.iloc[trn_idx])

    val_data = lgb.Dataset(X_Train.iloc[val_idx,:], label=Y_Train.iloc[val_idx])



    num_round = 1000000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1000)

    temp = clf.predict(X_Train.iloc[val_idx,:], num_iteration=clf.best_iteration)

    

    predictions_val = temp.copy()

    

    predictions_val[predictions_val >= 0.5] = 1

    predictions_val[predictions_val < 0.5] = 0

    

    print("CV score (Accuracy): {:<8.5f}".format(accuracy_score(Y_Train.iloc[val_idx], predictions_val)))

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = X_Train.columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(X_Test, num_iteration=clf.best_iteration) / folds.n_splits
Y_Pred = predictions.copy()



Y_Pred[Y_Pred >= 0.5] = 1

Y_Pred[Y_Pred < 0.5] = 0



fpr, tpr, thresholds = roc_curve(np.array(Y_Test), np.array(predictions))

roc_auc = auc(fpr, tpr)

print("ROC AUC of LightGBM  is {}".format(roc_auc))



from sklearn.metrics import f1_score

f1_score2 = f1_score(Y_Test, Y_Pred, average = "weighted")

print("F1 Score of LightGBM  is {}".format(f1_score2))



from sklearn.metrics import accuracy_score

acc = accuracy_score(Y_Test, Y_Pred)

print("Accuracy of LightGBM  is {}".format(acc))