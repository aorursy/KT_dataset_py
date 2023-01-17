import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import seaborn as sns  # visualization tool
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from nltk.corpus import stopwords
from textblob import TextBlob
import datetime as dt
import warnings
import string
import time
# stop_words = []
stop_words = list(set(stopwords.words('english')))
warnings.filterwarnings('ignore')
punctuation = string.punctuation


import tensorflow as tf
from tensorflow.python.data import Dataset

from scipy import stats
from scipy.sparse import hstack, csr_matrix
from mlxtend.preprocessing import minmax_scaling

from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import punkt
stop_words = stopwords.words('english')
from textblob import TextBlob

import xgboost as xgb
import lightgbm as lgb

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

from tqdm import tqdm
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import random
from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import punkt
stop_words = stopwords.words('english')

import string
import re
import gc
train = pd.read_csv("../input/train.csv", dtype={"project_essay_3": object, "project_essay_4": object})

target = train['project_is_approved']
train = train.drop('project_is_approved', axis=1)

test = pd.read_csv("../input/test.csv", dtype={"project_essay_3": object, "project_essay_4": object})
resources = pd.read_csv("../input/resources.csv")

train.fillna(('unk'), inplace=True)
test.fillna(('unk'), inplace=True)
print("Size of training data : ",train.shape)
print("Size of test data : ",test.shape)
print("Size of resource data : ",resources.shape)
train.head()
train.dtypes
train.info()
train.describe()
train.describe(include=["O"])
for label in ['teacher_prefix', 'project_grade_category']:
    print(train[label].value_counts())
    
# Teacher Prefix     
idx = np.where(train['teacher_prefix'].isna() == True)
print(idx)
print(train.iloc[idx])

# States    
states = np.where(train['school_state'].unique())
print(train['school_state'].iloc[states])
resources.head()
resources.dtypes
resources.info()
zero_count = train[train['teacher_number_of_previously_posted_projects'] == 0]
zero_project_percentage = (float(zero_count.shape[0]) / train.shape[0]) * 100
print("Percentage of teachers with their first project: " + str(zero_project_percentage))

one_count = train[train['teacher_number_of_previously_posted_projects'] == 1]
one_count_percentage = (float(one_count.shape[0]) / train.shape[0]) * 100
print("Percentage of teachers with only one project: " + str(one_count_percentage))

more_than_one = train[train['teacher_number_of_previously_posted_projects'] > 1]
more_than_one_percentage = (float(more_than_one.shape[0]) / train.shape[0]) * 100
print("Percentage of teachers with more than one project: " + str(more_than_one_percentage))

plt.figure(figsize = (12, 8))

sns.distplot(train['teacher_number_of_previously_posted_projects'])
plt.title('Histogram of number of previously posted applications by the submitting teacher')
plt.xlabel('Number of previously posted applications by the submitting teacher', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

plt.figure(figsize = (12, 8))
plt.hist(train['teacher_number_of_previously_posted_projects'], bins=[0, 10, 150, 450])
plt.title('Histogram Counting # of Teachers that Previously Posted Projects)')
plt.xlabel('Projects')
plt.ylabel('Count')
plt.show()

# Check to see if all teacher_id are present in the data set
# ['teacher_id'][0]
# ['teacher_id'][4]
print(len(np.where(train['teacher_id'] == train['teacher_id'][0])[0]))
print(len(np.where(train['teacher_id'] == train['teacher_id'][4])[0]))

project_approved = target.value_counts()
labels = project_approved.index
sizes = (project_approved / project_approved.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='Status of Project Proposal')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
train['teacher_prefix'].value_counts()
temp = train["teacher_prefix"].value_counts()
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(target[train["teacher_prefix"]==val] == 1))
    temp_y0.append(np.sum(target[train["teacher_prefix"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Popular Teacher prefixes in terms of project acceptance rate and project rejection rate",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
top_states = train["school_state"].value_counts().head(10)
plt.figure(figsize=(12,8))
sns.barplot(top_states.index, top_states.values)
plt.xlabel("State", fontsize=15)
plt.ylabel("Number of Projects", fontsize=15)
plt.show()

temp = train["school_state"].value_counts()
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(target[train["school_state"]==val] == 1))
    temp_y0.append(np.sum(target[train["school_state"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Popular School states in terms of project acceptance rate and project rejection rate",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = train["project_grade_category"].value_counts()
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(target[train["project_grade_category"]==val] == 1))
    temp_y0.append(np.sum(target[train["project_grade_category"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Popular school grade levels in terms of project acceptance rate and project rejection rate",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
plt.figure(figsize=(12,8))
subject_categories = train['project_subject_categories'].value_counts()
ax = subject_categories.iloc[:15].plot(kind="barh")
ax.invert_yaxis()
plt.xlabel("Project Subject Category", fontsize=15)
plt.ylabel("Number of Project", fontsize=15)
plt.show()
plt.figure(figsize=(12,8))
project_subject_subcategories = train['project_subject_subcategories'].value_counts()
ax = project_subject_subcategories.iloc[:15].plot(kind="barh")
ax.invert_yaxis()
plt.xlabel("Project Subject Subcategories", fontsize=15)
plt.ylabel("Number of Projects", fontsize=15)
plt.show()
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head()
train.head()
from sklearn import preprocessing
from tqdm import tqdm
import gc

features = [
    'teacher_id', 
    'teacher_prefix', 
    'school_state', 
    'project_grade_category',
    'project_subject_categories', 
    'project_subject_subcategories']

df_all = pd.concat([train, test], axis=0)
    
for c in tqdm(features):
    le = LabelEncoder()
    le.fit(df_all[c].astype(str))
    train[c] = le.transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))
train.head()
# Feature engineering

# Date and time
train['project_submitted_datetime'] = pd.to_datetime(train['project_submitted_datetime'])
test['project_submitted_datetime'] = pd.to_datetime(test['project_submitted_datetime'])

# Date as int may contain some ordinal value
train['datetime_int'] = train['project_submitted_datetime'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['datetime_int'] = test['project_submitted_datetime'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

# Date parts
train["year"] = train["project_submitted_datetime"].dt.year
train["month"] = train["project_submitted_datetime"].dt.month
#train['weekday'] = train['project_submitted_datetime'].dt.weekday
train["hour"] = train["project_submitted_datetime"].dt.hour
train["month_Day"] = train['project_submitted_datetime'].dt.day
#train["year_Day"] = train['project_submitted_datetime'].dt.dayofyear
train['datetime_dow'] = train['project_submitted_datetime'].dt.dayofweek
train = train.drop('project_submitted_datetime', axis=1)


# ****** Test data *********
test["year"] = test["project_submitted_datetime"].dt.year
test["month"] = test["project_submitted_datetime"].dt.month
#test['weekday'] = test['project_submitted_datetime'].dt.weekday
test["hour"] = test["project_submitted_datetime"].dt.hour
test["month_Day"] = test['project_submitted_datetime'].dt.day
#test["year_Day"] = test['project_submitted_datetime'].dt.dayofyear
test['datetime_dow'] = test['project_submitted_datetime'].dt.dayofweek
test = test.drop('project_submitted_datetime', axis=1)

# Essay length
train['e1_length'] = train['project_essay_1'].apply(len)
test['e1_length'] = train['project_essay_1'].apply(len)

train['e2_length'] = train['project_essay_2'].apply(len)
test['e2_length'] = train['project_essay_2'].apply(len)

# Title length
train['project_title_len'] = train['project_title'].apply(lambda x: len(str(x)))
test['project_title_len'] = test['project_title'].apply(lambda x: len(str(x)))

# Project resource summary length
train['project_resource_summary_len'] = train['project_resource_summary'].apply(lambda x: len(str(x)))
test['project_resource_summary_len'] = test['project_resource_summary'].apply(lambda x: len(str(x)))

# Has more than 2 essays?
train['has_gt2_essays'] = train['project_essay_3'].apply(lambda x: 0 if x == 'unk' else 1)
test['has_gt2_essays'] = test['project_essay_3'].apply(lambda x: 0 if x == 'unk' else 1)
resources['resources_total'] = resources['quantity'] * resources['price']

dfr = resources.groupby(['id'], as_index=False)[['resources_total']].sum()
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

dfr = resources.groupby(['id'], as_index=False)[['resources_total']].mean()
dfr = dfr.rename(columns={'resources_total':'resources_total_mean'})
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

dfr = resources.groupby(['id'], as_index=False)[['quantity']].count()
dfr = dfr.rename(columns={'quantity':'resources_quantity_count'})
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

dfr = resources.groupby(['id'], as_index=False)[['quantity']].sum()
dfr = dfr.rename(columns={'quantity':'resources_quantity_sum'})
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

# We're done with IDs for now
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)
train['project_essay'] = train.apply(lambda row: ' '.join([
    str(row['project_title']),
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']),
    str(row['project_essay_4']),
    str(row['project_resource_summary'])]), axis=1)
test['project_essay'] = test.apply(lambda row: ' '.join([
    str(row['project_title']),
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']),
    str(row['project_essay_4']),
    str(row['project_resource_summary'])]), axis=1)

train = train.drop([
    'project_title',
    'project_essay_1', 
    'project_essay_2', 
    'project_essay_3', 
    'project_essay_4',
    'project_resource_summary'], axis=1)
test = test.drop([
    'project_title',
    'project_essay_1', 
    'project_essay_2', 
    'project_essay_3', 
    'project_essay_4',
    'project_resource_summary'], axis=1)
train.head()
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import punkt
import re

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def prep_text(text):
    q = "[\'\’\´\ʻ]"
    text = text.strip().lower()
    text = re.sub('\W+',' ', text)
    text = re.sub(r'(\")', ' ', text)
    text = re.sub(r"\\r|\\n", " ", text)
    text = re.sub(re.compile("won%st" % q), "will not", text)
    text = re.sub(re.compile("can%st" % q), "can not", text)
    text = re.sub(re.compile("n%st" % q), " not", text)
    text = re.sub(re.compile("%sre" % q), " are", text)
    text = re.sub(re.compile("%ss" % q), " is", text)
    text = re.sub(re.compile("%sd" % q), " would", text)
    text = re.sub(re.compile("%sll" % q), " will", text)
    text = re.sub(re.compile("%st" % q), " not", text)
    text = re.sub(re.compile("%sve" % q), " have", text)
    text = re.sub(re.compile("%sm" % q), " am", text)
    text = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
    return text

train['project_essay'] = train['project_essay'].apply(lambda x: prep_text(x))
test['project_essay'] = test['project_essay'].apply(lambda x: prep_text(x))
tfv = TfidfVectorizer(norm='l2', min_df=0,  max_features=8000, 
            strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
            ngram_range=(1,2), use_idf=True, smooth_idf=False, sublinear_tf=True,
            stop_words = 'english')
train_text = train['project_essay'].apply(lambda x: ' '.join(x))
test_text = test['project_essay'].apply(lambda x: ' '.join(x))

# Fitting tfidf on train + test might be leaky
tfv.fit(list(train_text.values) + list(test_text.values))
train_tfv = tfv.transform(train_text)
test_tfv = tfv.transform(test_text)
feat_train = train.drop('project_essay', axis=1)
feat_test = test.drop('project_essay', axis=1)

feat_train = csr_matrix(feat_train.values)
feat_test = csr_matrix(feat_test.values)

X_train_stack = hstack([feat_train, train_tfv[0:feat_train.shape[0]]])
X_test_stack = hstack([feat_test, test_tfv[0:feat_test.shape[0]]])

print('Train shape: ', X_train_stack.shape, '\n\nTest Shape: ', X_test_stack.shape)
print("Building model using Light GBM and finding AUC(Area Under Curve)")

cnt = 0
p_buf = []
n_splits = 5
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=28)
auc_buf = []  

for train_index, valid_index in kf.split(X_train_stack):
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_stack, target, test_size=0.20, random_state=random.seed(28))
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 7,
        'num_leaves': 32,
        'learning_rate': 0.02,
        'feature_fraction': 0.80,
        'bagging_fraction': 0.80,
        'bagging_freq': 5,
        'verbose': 0,
        'lambda_l2': 1,
    }  

    model = lgb.train(
        params,
        lgb.Dataset(X_train, y_train),
        num_boost_round=10000,
        valid_sets=[lgb.Dataset(X_valid, y_valid)],
        early_stopping_rounds=50,
        verbose_eval=100
        )

    p = model.predict(X_valid, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_valid, p)

    print('{} AUC: {}'.format(cnt, auc))

    p = model.predict(X_test_stack, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p)
    else:
        p_buf += np.array(p)
    auc_buf.append(auc)

    cnt += 1
    
auc_mean = np.mean(auc_buf)
auc_std = np.std(auc_buf)
print('AUC = {:.6f} +/- {:.6f}'.format(auc_mean, auc_std))

lgb_preds = p_buf/cnt
l_preds = pd.DataFrame(lgb_preds)
l_preds.columns = ['project_is_approved']
l_preds.head()

subid = sub['id']
lsub = pd.concat([submid, l_preds], axis=1)
print("Building model using XGBoost and finding AUC(Area Under Curve)")

kf = KFold(n_splits = 5, random_state = 28, shuffle = True)

cv_scores = []
xgb_preds = []

for train_index, test_index in kf.split(X_train_stack):
    
    # Split out a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_stack, target, test_size=0.20, random_state=random.seed(28))
    
    # params are tuned with kaggle kernels in mind
    xgb_params = {'eta': 0.15, 
                  'max_depth': 7, 
                  'subsample': 0.80, 
                  'colsample_bytree': 0.80, 
                  'objective': 'binary:logistic', 
                  'eval_metric': 'auc', 
                  'seed': 28
                 }
    
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    d_test = xgb.DMatrix(X_test_stack)
    
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(xgb_params, d_train, 2000, watchlist, verbose_eval=50, early_stopping_rounds=30)
    cv_scores.append(float(model.attributes()['best_score']))
    xgb_pred = model.predict(d_test)
    xgb_preds.append(list(xgb_pred))
x_preds = pd.DataFrame(x_preds)
x_preds.columns = ['project_is_approved']

submid = sub['id']
xsub = pd.concat([submid, x_preds], axis=1)
