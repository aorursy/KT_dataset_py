# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, recall_score, precision_score,make_scorer
from sklearn.preprocessing import StandardScaler
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
import nltk
from nltk import word_tokenize, ngrams
import warnings
import string
import time
stop_words = list(set(stopwords.words('english')))
warnings.filterwarnings('ignore')
punctuation = string.punctuation
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from scipy import stats
from scipy.stats import norm, skew #for some statistics
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
np.random.seed(25)
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['font.size'] = 15
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
resources = pd.read_csv('../input/resources.csv')
train.head()
train.isnull().sum(axis=0)
top_states = train['school_state'].value_counts().head(10)
plt.figure(figsize=(12,8))
sns.barplot(top_states.index, top_states.values)
plt.xlabel("State", fontsize=15)
plt.ylabel("Number of Projects", fontsize=15)
plt.show()
top_states = train['project_grade_category'].value_counts().head()
plt.figure(figsize=(12,8))
sns.barplot(top_states.index, top_states.values)
plt.xlabel("State", fontsize=15)
plt.ylabel("Number of Projects", fontsize=15)
plt.show()
plt.figure(figsize=(12,8))
top_categories = train['project_subject_categories'].value_counts()
ax = top_categories.iloc[:10].plot(kind="barh")
ax.invert_yaxis()
plt.xlabel("State", fontsize=15)
plt.ylabel("Number of Projects", fontsize=15)
plt.show()
plt.figure(figsize=(12,8))
top_categories = train['project_subject_subcategories'].value_counts()
ax = top_categories.iloc[:10].plot(kind="barh")
ax.invert_yaxis()
plt.xlabel("State", fontsize=15)
plt.ylabel("Number of Projects", fontsize=15)
plt.show()
plt.figure(figsize=(12,8))
plt.scatter(range(len(train['teacher_number_of_previously_posted_projects'])),np.sort(train['teacher_number_of_previously_posted_projects'].values))
plt.xlabel("index", fontsize=15)
plt.ylabel("Number of previous projects", fontsize=15)
plt.show()
plt.figure(figsize=(12,8))
sns.distplot(train.teacher_number_of_previously_posted_projects.values,bins=50)
plt.xlabel("Previous Projects",fontsize=15)
plt.show()
zero_count = train[train['teacher_number_of_previously_posted_projects'] == 0]
zero_project_percentage = (float(zero_count.shape[0]) / train.shape[0]) * 100
print("Percentage of teachers with their first project: " + str(zero_project_percentage))
plt.figure(figsize=(12,8))
train['project_is_approved'].value_counts().plot.bar()
plt.xlabel("is accepted", fontsize=15)  ## 1 means 'accepted' and 0 means 'rejected'
plt.ylabel("Number of Projects", fontsize=15)
plt.show()
train['teacher_prefix'].value_counts()
plt.figure(figsize=(12,8))
train['teacher_prefix'].value_counts().plot.bar()
plt.xlabel("Teacher Prefix", fontsize=15)
plt.ylabel("Number of Projects", fontsize=15)
plt.show()
# li = ['project_title','project_essay_1','project_essay_2','project_essay_3','project_essay_4','project_resource_summary']
# for i in li:
#     words = train[i][~pd.isnull(train[i])]
#     wordcloud = WordCloud(max_font_size=50, width=600, height=300,max_words=2000).generate(' '.join(words))
#     plt.figure(figsize=(12,12))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.title(i)
#     plt.axis("off")
## merge features of resources with train and test data

resources['total_price'] = resources.quantity * resources.price
mean_total_price = pd.DataFrame(resources.groupby('id').total_price.mean()) 
sum_total_price = pd.DataFrame(resources.groupby('id').total_price.sum()) 
count_total_price = pd.DataFrame(resources.groupby('id').total_price.count())
mean_total_price['id'] = mean_total_price.index
sum_total_price['id'] = mean_total_price.index
count_total_price['id'] = mean_total_price.index

def create_features(df):
    df = pd.merge(df, mean_total_price, on='id')
    df = pd.merge(df, sum_total_price, on='id')
    df = pd.merge(df, count_total_price, on='id')
    return df

train = create_features(train)
test = create_features(test)
# gender mapping using prefixes
gender_mapping = {"Mrs.": "Female", "Ms.":"Female", "Mr.":"Male", "Teacher":"Unknown", "Dr.":"Unknown", np.nan:"Unknown"}
train["gender"] = train.teacher_prefix.map(gender_mapping)
test["gender"] = test.teacher_prefix.map(gender_mapping)
## create month and year from datetime
train["project_submitted_datetime"] = pd.to_datetime(train["project_submitted_datetime"])
train["month"] = train["project_submitted_datetime"].dt.month
train["year"] = train["project_submitted_datetime"].dt.year

test["project_submitted_datetime"] = pd.to_datetime(test["project_submitted_datetime"])
test["month"] = test["project_submitted_datetime"].dt.month
test["year"] = test["project_submitted_datetime"].dt.year
## First combine project essay 1 and project essay 2
train['full_essay'] = train['project_essay_1'] + ' ' + train['project_essay_2']
test['full_essay'] = test['project_essay_1'] + ' ' + test['project_essay_2']
start_time=time.time()

train['num_of_words'] = train["full_essay"].apply(lambda x: len(str(x).split()))
train['num_of_chars'] = train["full_essay"].apply(len)
train["num_unique_words"] = train["full_essay"].apply(lambda x: len(set(str(x).split())))
train["num_stopwords"] = train["full_essay"].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))
train["num_punctuations"] =train['full_essay'].apply(lambda x: len([c for c in str(x) if c in punctuation]))

test['num_of_words'] = test["full_essay"].apply(lambda x: len(str(x).split()))
test['num_of_chars'] = test["full_essay"].apply(len)
test["num_unique_words"] = test["full_essay"].apply(lambda x: len(set(str(x).split())))
test["num_stopwords"] = test["full_essay"].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))
test["num_punctuations"] =test['full_essay'].apply(lambda x: len([c for c in str(x) if c in punctuation]))

end_time=time.time()
print("total time consumed: ",end_time-start_time,"s")
start_time=time.time()
# One-hot encoding
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
train["teacher_id"] = lb_make.fit_transform(train["teacher_id"].astype(str))
train["teacher_prefix"] = lb_make.fit_transform(train["teacher_prefix"].astype(str))
train["school_state"] = lb_make.fit_transform(train["school_state"].astype(str))
train["project_grade_category"] = lb_make.fit_transform(train["project_grade_category"].astype(str))
train["project_subject_categories"] = lb_make.fit_transform(train["project_subject_categories"].astype(str))
train["project_subject_subcategories"] = lb_make.fit_transform(train["project_subject_subcategories"].astype(str))
train["gender"] = lb_make.fit_transform(train["gender"].astype(str))

test["teacher_id"] = lb_make.fit_transform(test["teacher_id"].astype(str))
test["teacher_prefix"] = lb_make.fit_transform(test["teacher_prefix"].astype(str))
test["school_state"] = lb_make.fit_transform(test["school_state"].astype(str))
test["project_grade_category"] = lb_make.fit_transform(test["project_grade_category"].astype(str))
test["project_subject_categories"] = lb_make.fit_transform(test["project_subject_categories"].astype(str))
test["project_subject_subcategories"] = lb_make.fit_transform(test["project_subject_subcategories"].astype(str))
test["gender"] = lb_make.fit_transform(test["gender"].astype(str))

end_time=time.time()
print("total time consumed: ",end_time-start_time,"s")
#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(corrmat, vmax=.8, square=True);
train.columns
feature_names = [x for x in train.columns if x not in ['id',
       'project_submitted_datetime',
       'project_title', 'project_essay_1', 'project_essay_2',
       'project_essay_3', 'project_essay_4', 'project_resource_summary',
       'project_is_approved','description','full_essay']]
target = train['project_is_approved']
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train[feature_names], target, test_size = 0.2, random_state = 42)
vote_est = [
    
   ('lgb', lgb.LGBMClassifier(lambda_l2=1.0,feature_fraction=0.6,num_boost_round=1200,num_leaves=9)),
    ('logit', LogisticRegression()),
    ('xgb', xgb.XGBClassifier(max_depth=0,eta=0.01)),
    ('gb',GradientBoostingClassifier(max_features='auto',n_estimators=1000))
]

model = VotingClassifier(estimators = vote_est , voting = 'soft')
## model training and prediction
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("roc_auc_score: " + str(roc_auc_score(pred, y_test.values)))
vote_est = [
    
   ('lgb', lgb.LGBMClassifier(lambda_l2=1.0,feature_fraction=0.6,num_boost_round=1200,num_leaves=9)),
    ('logit', LogisticRegression()),
    ('xgb', xgb.XGBClassifier(max_depth=0,eta=0.01)),
    ('gb',GradientBoostingClassifier(max_features='auto',n_estimators=1000))
]

model = VotingClassifier(estimators = vote_est , voting = 'soft')
model.fit(train[feature_names], target)
pred = model.predict_proba(test[feature_names])[:, 1]
pred[:5]
## make submission
sub = pd.DataFrame()
sub['id'] = test['id']
sub['project_is_approved'] = pred
sub.to_csv('result.csv', index=False)