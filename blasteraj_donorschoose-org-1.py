import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import datetime
import lightgbm as lgb
import pandas_profiling as pp
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
stop = set(stopwords.words('english'))

import xgboost as xgb
import lightgbm as lgb
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import re
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
Train = pd.read_csv('../input/train.csv')
Test = pd.read_csv('../input/test.csv')
Resources = pd.read_csv('../input/resources.csv')
Submission = pd.read_csv('../input/sample_submission.csv')
temp = Train[Train['teacher_id']=='484aaf11257089a66cfedc9461c6bd0a']
temp = temp.sort_values(by=['project_submitted_datetime'])
temp[:2]
Resources.head()
train_len = len(Train)
y = Train['project_is_approved']
#Train.drop('project_is_approved',axis=1,inplace=True)
Train.head()
print('Values Counts of projects each \'teacher_id\'')
print(Train['teacher_id'].value_counts().reset_index()[:5])
print('Values Counts of projects each \'teacher_prefix\'')
print(Train['teacher_prefix'].value_counts().reset_index())
print('Values Counts of projects each \'school_state\'')
print(Train['school_state'].value_counts().reset_index()[:10])
print('Values Counts of projects each \'project_grade_category\'')
print(Train['project_grade_category'].value_counts().reset_index()[:10])
print('Values Counts of projects each \'project_is_approved\'')
print(Train['project_is_approved'].value_counts().reset_index()[:10])
Train.columns
unique_val_cols = ['teacher_prefix', 'school_state', 'project_grade_category','project_subject_categories', 'project_subject_subcategories']
for cols in unique_val_cols:
    print('Unique values in '+cols)
    print("Train",len(Train[str(cols)].unique()))
    print("Test",len(Test[str(cols)].unique()))
    print('\n')
# Merged_Data = Train.append(Test)
# Train_len =  len(Train)

# from sklearn.preprocessing import LabelEncoder
# for x in unique_val_cols:
#     lbl = LabelEncoder()
#     Merged_Data[x]=lbl.fit_transform(Merged_Data[x].astype(str))

# Merged_Data.head()

# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# # function to clean data

# stops = set(stopwords.words("english"))
# def cleanData(text, lowercase = False, remove_stops = False, stemming = False):
#     txt = str(text)
#     txt = re.sub(r'[^A-Za-z0-9\s]',r'',txt)
#     txt = re.sub(r'\n',r' ',txt)
    
#     if lowercase:
#         txt = " ".join([w.lower() for w in txt.split()])
        
#     if remove_stops:
#         txt = " ".join([w for w in txt.split() if w not in stops])
    
#     if stemming:
#         st = PorterStemmer()
#         txt = " ".join([st.stem(w) for w in txt.split()])

#     return txt
#Let's use pandas profiling library to generate report which is very fast method of analysis  -
pp.ProfileReport(Resources[['quantity', 'price']])
pp.ProfileReport(Train[['project_grade_category','project_is_approved']])
pp.ProfileReport(Train[['teacher_id', 'teacher_prefix', 'school_state', 'project_grade_category', 'teacher_number_of_previously_posted_projects', 'project_is_approved']])
Train.columns
Train['project_submitted_datetime']=pd.to_datetime(Train['project_submitted_datetime'])
print('Projects before May 17 2016 are', np.sum(Train['project_submitted_datetime']<datetime.date(2016,5,7)))
print('Projects after May 17 2016 are', np.sum(Train['project_submitted_datetime']>datetime.date(2016,5,7)))
Train.loc[Train['project_submitted_datetime']<datetime.date(2016,5,7),'project_essay_1']=Train.loc[Train['project_submitted_datetime']<datetime.date(2016,5,7),'project_essay_1']+Train.loc[Train['project_submitted_datetime']<datetime.date(2016,5,7),'project_essay_2'] 
Train.loc[Train['project_submitted_datetime']<datetime.date(2016,5,7),'project_essay_2']= Train.loc[Train['project_submitted_datetime']<datetime.date(2016,5,7),'project_essay_3']+Train.loc[Train['project_submitted_datetime']<datetime.date(2016,5,7),'project_essay_4']
Train.drop(['project_essay_3','project_essay_4'],axis=1,inplace=True)
Train['project_submitted_datetime']
# function to clean data
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


stops = set(stopwords.words("english"))
def cleanData(text, lowercase = False, remove_stops = False, stemming = False):
    txt = str(text)
    txt = re.sub(r'[^A-Za-z0-9\s]',r'',txt)
    txt = re.sub(r'\n',r' ',txt)
    
    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])
        
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stops])
    
    if stemming:
        st = PorterStemmer()
        txt = " ".join([st.stem(w) for w in txt.split()])

    return txt
Train['project_essay_1'] = Train['project_essay_1'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True))
Train['project_essay_2'] = Train['project_essay_2'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True))
text = ' '.join(Train['project_essay_1'].values)
wordcloud = WordCloud(max_font_size = None, stopwords = stop, background_color = 'white', width = 1200, height = 1000).generate(text)
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.title('Top words from project_essay_1')
plt.axis('off')
plt.show()
text = ' '.join(Train['project_essay_2'].values)
wordcloud = WordCloud(max_font_size = None, stopwords = stop, background_color = 'white', width = 1200, height = 1000).generate(text)
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.title('Top words from project_essay_1')
plt.axis('off')
plt.show()
Train['project_resource_summary'] = Train['project_resource_summary'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))
text = ' '.join(Train.loc[Train['project_is_approved'] == 1, 'project_resource_summary'].values)
wordcloud = WordCloud(max_font_size = None, stopwords = stop, background_color = 'white', width = 1200, height = 1000).generate(text)
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.title('Top words from project_resource_simmary where projects_are_approved')
plt.axis('off')
plt.show()
text = ' '.join(Train.loc[Train['project_is_approved'] == 0, 'project_resource_summary'].values)
wordcloud = WordCloud(max_font_size = None, stopwords = stop, background_color = 'white', width = 1200, height = 1000).generate(text)
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.title('Top words from project_resource_simmary where projects_are_not_approved')
plt.axis('off')
plt.show()
Train['project_title'] = Train['project_title'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))
text = ' '.join(Train['project_title'].values)
text = [i for i in ngrams(text.split(), 3)]
print('Common trigrams.')
Counter(text).most_common(20)
Train.columns
print('Common titles')
Train['project_title'].value_counts()
print('Summary of project : Flexible seating')
for i in Train.loc[Train['project_title']=='Flexible Seating','project_resource_summary'][:3]:
    print(i)
    
Train['teacher_prefix'].value_counts(dropna=False)
pd.crosstab(Train.teacher_prefix,Train.project_is_approved,dropna='False',normalize='index')
Train['teacher_prefix'].fillna('Mrs.', inplace=True)
pd.crosstab(Train.school_state,Train.project_is_approved,dropna='False',normalize='index')
Train.groupby('school_state').agg({'project_is_approved':['mean','count']}).reset_index().sort_values([('project_is_approved','mean')],ascending=False,).reset_index(drop=True)
Train['date']= Train['project_submitted_datetime'].dt.date
Train['weekday']= Train['project_submitted_datetime'].dt.weekday
Train['day']= Train['project_submitted_datetime'].dt.day

count_by_date = Train.groupby('date')['project_is_approved'].count()
mean_by_date = Train.groupby('date')['project_is_approved'].mean()
Train.head()
fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Trends of approval rates and number of projects")
#count_by_date.rolling(window=12,center=False).mean().plot(ax=ax1, legend=False)
count_by_date.plot(ax=ax1, legend=False)
ax1.set_ylabel('Projects count', color='b')
plt.legend(['Projects count'])
ax2 = ax1.twinx()
#mean_by_date.rolling(window=12,center=False).mean().plot(ax=ax2, color='g', legend=False)
mean_by_date.plot(ax=ax2, color='g', legend=False)
ax2.set_ylabel('Approval rate', color='g')
plt.legend(['Approval rate'], loc=(0.875, 0.9))
plt.grid(False)



fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Project count and approval rate by day of week.")
sns.countplot(x='weekday', data=Train, ax=ax1)
ax1.set_ylabel('Projects count', color='b')
plt.legend(['Projects count'])
ax2 = ax1.twinx()
sns.pointplot(x="weekday", y="project_is_approved", data=Train, ci=99, ax=ax2, color='black')
ax2.set_ylabel('Approval rate', color='g')
plt.legend(['Approval rate'], loc=(0.875, 0.9))
plt.grid(False)
pd.crosstab(Train.project_grade_category, Train.project_is_approved, dropna=False, normalize='index')
Train['project_subject_subcategories'].values
psc = [i.split(', ') for i in Train.project_subject_subcategories.values]
psc
psc = [i for j in psc for i in j]
print('Most common subcategories')
Counter(psc).most_common()
Train.groupby('teacher_number_of_previously_posted_projects')['project_is_approved'].mean().plot()
print("No of unique id's in resources column : ",len(Resources['id'].unique()))
print("Totla length of resources column :",len(Resources))
Resources['cost'] = Resources['quantity'] * Resources['price']
resources_aggregated = Resources.groupby('id').agg({'description': ['nunique'], 'quantity': ['sum'], 'cost': ['mean', 'sum']})
resources_aggregated.columns = ['unique_items', 'total_quantity', 'mean_cost', 'total_cost']
resources_aggregated.reset_index(inplace=True)
resources_aggregated.head()
print('99 percentile is {0}.'.format(np.percentile(resources_aggregated.mean_cost, 99)))
plt.boxplot(resources_aggregated.mean_cost);
Train = pd.merge(Train, resources_aggregated, how='left', on='id')
Test = pd.merge(Test, resources_aggregated, how='left', on='id')
Test['project_submitted_datetime'] = pd.to_datetime(Test['project_submitted_datetime'])
Test.loc[Test['project_submitted_datetime']<datetime.date(2016,5,7),'project_essay_1']=Test.loc[Test['project_submitted_datetime']<datetime.date(2016,5,7),'project_essay_1']+Test.loc[Test['project_submitted_datetime']<datetime.date(2016,5,7),'project_essay_2'] 
Test.loc[Test['project_submitted_datetime']<datetime.date(2016,5,7),'project_essay_2']= Test.loc[Test['project_submitted_datetime']<datetime.date(2016,5,7),'project_essay_3']+Test.loc[Test['project_submitted_datetime']<datetime.date(2016,5,7),'project_essay_4']
Test.drop(['project_essay_3','project_essay_4'],axis=1,inplace=True)
Test['project_essay_1'] = Test['project_essay_1'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True))
Test['project_essay_2'] = Test['project_essay_2'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True))
Train.drop('date', axis=1, inplace=True)
Train.columns
Test.columns
Test['teacher_prefix'].fillna('Mrs.', inplace=True)

Test['weekday'] = Test.project_submitted_datetime.dt.weekday
Test['day'] = Test.project_submitted_datetime.dt.day
Train = pd.concat([Train,
                   pd.get_dummies(Train['teacher_prefix'], drop_first=True),
                   pd.get_dummies(Train['project_grade_category'], drop_first=True),
                   pd.get_dummies(Train['weekday'], drop_first=True)], axis=1)
Train.drop(['teacher_prefix', 'project_grade_category', 'weekday'], axis=1, inplace=True)

Test = pd.concat([Test,
                   pd.get_dummies(Test['teacher_prefix'], drop_first=True),
                   pd.get_dummies(Test['project_grade_category'], drop_first=True),
                   pd.get_dummies(Test['weekday'], drop_first=True)], axis=1)
Test.drop(['teacher_prefix', 'project_grade_category', 'weekday'], axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
Train['school_state'] = lbl.fit_transform(Train['school_state'])
Test['school_state'] = lbl.fit_transform(Test['school_state'])
Train['school_state'].head(10)
Train['len_project_subject_categories'] = Train['project_subject_categories'].apply(lambda x: len(x))
Train['words_project_subject_categories'] = Train['project_subject_categories'].apply(lambda x: len(x.split()))
Train['len_project_subject_subcategories'] = Train['project_subject_subcategories'].apply(lambda x: len(x))
Train['words_project_subsubject_categories'] = Train['project_subject_subcategories'].apply(lambda x: len(x.split()))
Train['len_project_title'] = Train['project_title'].apply(lambda x: len(x))
Train['words_project_title'] = Train['project_title'].apply(lambda x: len(x.split()))
Train['len_project_resource_summary'] = Train['project_resource_summary'].apply(lambda x: len(x))
Train['words_project_resource_summary'] = Train['project_resource_summary'].apply(lambda x: len(x.split()))
Train['len_project_essay_1'] = Train['project_essay_1'].apply(lambda x: len(x))
Train['words_project_essay_1'] = Train['project_essay_1'].apply(lambda x: len(x.split()))
Train['len_project_essay_2'] = Train['project_essay_2'].apply(lambda x: len(x))
Train['words_project_essay_2'] = Train['project_essay_2'].apply(lambda x: len(x.split()))

Test['len_project_subject_categories'] = Test['project_subject_categories'].apply(lambda x: len(x))
Test['words_project_subject_categories'] = Test['project_subject_categories'].apply(lambda x: len(x.split()))
Test['len_project_subject_subcategories'] = Test['project_subject_subcategories'].apply(lambda x: len(x))
Test['words_project_subsubject_categories'] = Test['project_subject_subcategories'].apply(lambda x: len(x.split()))
Test['len_project_title'] = Test['project_title'].apply(lambda x: len(x))
Test['words_project_title'] = Test['project_title'].apply(lambda x: len(x.split()))
Test['len_project_resource_summary'] = Test['project_resource_summary'].apply(lambda x: len(x))
Test['words_project_resource_summary'] = Test['project_resource_summary'].apply(lambda x: len(x.split()))
Test['len_project_essay_1'] = Test['project_essay_1'].apply(lambda x: len(x))
Test['words_project_essay_1'] = Test['project_essay_1'].apply(lambda x: len(x.split()))
Test['len_project_essay_2'] = Test['project_essay_2'].apply(lambda x: len(x))
Test['words_project_essay_2'] = Test['project_essay_2'].apply(lambda x: len(x.split()))
vectorizer=TfidfVectorizer(stop_words=stop)
vectorizer.fit(Train['project_subject_categories'])
train_project_subject_categories = vectorizer.transform(Train['project_subject_categories'])
test_project_subject_categories = vectorizer.transform(Test['project_subject_categories'])

vectorizer.fit(Train['project_subject_subcategories'])
train_project_subject_subcategories = vectorizer.transform(Train['project_subject_subcategories'])
test_project_subject_subcategories = vectorizer.transform(Test['project_subject_subcategories'])
vectorizer=TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), max_df=0.9, min_df=5, max_features=2000)
vectorizer.fit(Train['project_title'])
train_project_title = vectorizer.transform(Train['project_title'])
test_project_title = vectorizer.transform(Test['project_title'])

vectorizer.fit(Train['project_resource_summary'])
train_project_resource_summary = vectorizer.transform(Train['project_resource_summary'])
test_project_resource_summary = vectorizer.transform(Test['project_resource_summary'])
vectorizer=TfidfVectorizer(stop_words=stop, ngram_range=(1, 3), max_df=0.9, min_df=5, max_features=2000)
vectorizer.fit(Train['project_essay_1'])
train_project_essay_1 = vectorizer.transform(Train['project_essay_1'])
test_project_essay_1 = vectorizer.transform(Test['project_essay_1'])

vectorizer.fit(Train['project_essay_2'])
train_project_essay_2 = vectorizer.transform(Train['project_essay_2'])
test_project_essay_2 = vectorizer.transform(Test['project_essay_2'])
cols_to_normalize = ['teacher_number_of_previously_posted_projects', 'len_project_subject_categories', 'words_project_subject_categories', 'len_project_subject_subcategories',
                     'words_project_subsubject_categories', 'len_project_title', 'words_project_title', 'len_project_resource_summary', 'words_project_resource_summary',
                     'len_project_essay_1', 'words_project_essay_1', 'len_project_essay_2', 'words_project_essay_2']
scaler = StandardScaler()
for col in cols_to_normalize:
    #print(col)
    scaler.fit(Train[col].values.reshape(-1, 1))
    Train[col] = scaler.transform(Train[col].values.reshape(-1, 1))
    Test[col] = scaler.transform(Test[col].values.reshape(-1, 1))
to_drop = ['teacher_id', 'school_state', 'project_submitted_datetime', 'project_subject_categories', 'project_subject_subcategories', 'project_title', 'project_essay_1', 'project_essay_2', 'project_resource_summary']
for col in to_drop:
    Train.drop([col], axis=1, inplace=True)
    Test.drop([col], axis=1, inplace=True)
y = Train['project_is_approved']
X = Train.drop(['id', 'project_is_approved'], axis=1)
X_test = Test.drop('id', axis=1)
X_full = csr_matrix(hstack([X.values, train_project_subject_categories, train_project_subject_subcategories, train_project_resource_summary, train_project_essay_1, train_project_essay_2]))
X_test_full = csr_matrix(hstack([X_test.values, test_project_subject_categories, test_project_subject_subcategories, test_project_resource_summary, test_project_essay_1, test_project_essay_2]))

X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, test_size=0.20, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, test_size=0.20, random_state=42)
# Delete unnecessary data to free memory.
del train_project_subject_categories
del train_project_subject_subcategories
del train_project_resource_summary
del train_project_essay_1
del train_project_essay_2
del test_project_subject_categories
del test_project_subject_subcategories
del test_project_resource_summary
del test_project_essay_1
del test_project_essay_2
del X_full
params = {'eta': 0.05, 'max_depth': 15, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 42, 'silent': True, 'colsample':0.9}
watchlist = [(xgb.DMatrix(X_train, y_train), 'train'), (xgb.DMatrix(X_valid, y_valid), 'valid')]
model = xgb.train(params, xgb.DMatrix(X_train, y_train), 1000,  watchlist, verbose_eval=10, early_stopping_rounds=20)
submission = pd.read_csv('../input/sample_submission.csv')
submission['project_is_approved'] = model.predict(xgb.DMatrix(X_test_full), ntree_limit=model.best_ntree_limit)
submission.to_csv('xgb_lgb.csv', index=False)
