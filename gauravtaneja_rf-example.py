# Basic Data manipulation and math functions
import pandas as pd
import numpy as np
import random
import scipy

# File listing, creating directory paths etc. and memory management
import os
import gc

# Garphing
from matplotlib import pyplot as plt
import seaborn as sns

# NLP specific and string functionalities
import re, string
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer

# Iporting fucntions from the popular sklearn ML module
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
# Constants
run_gridsearch__=False
nrows__ = 1000
fit_sample_size__=1.1
rf_n_estimators__=1000
train_inloc_ = '../input/train.csv'
test_inloc_ = '../input/test.csv'
labels_ = []
en_stop_ = get_stop_words('en')
p_stemmer = PorterStemmer()
rsrc = pd.read_csv("../input/resources.csv")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
rsrc.head()
print("Train:", train.shape)
print("Test:", test.shape)
print("Resource:", rsrc.shape)

train.isnull().sum()[train.isnull().sum()>0]
test.isnull().sum()[test.isnull().sum()>0]
rsrc.isnull().sum()[rsrc.isnull().sum()>0]
train = train.fillna('')
test = test.fillna('')
rsrc = rsrc.fillna('')
# Combining the project essay 1, 2, 3, 4 
for df in [train, test]:
    df["essays"] = df["project_essay_1"] + df["project_essay_2"] + df["project_essay_3"] + df["project_essay_4"]
    df.drop(['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4'], axis=1, inplace=True)

def data_preview_str(df, name):
    tot_feats = 0
    if name=='train':
        for col in [i for i in df.columns if (df[i].dtypes=='O') & (i.find('_id')<0)]:
            tot_feats = tot_feats + len(df[col].unique())-1
            print(" %s "% col, '# unique: ',len(df[col].unique()))
            xdf = df.groupby(col).agg({'id':'count','project_is_approved':'sum'})
            xdf.columns = ['#Records','Response_Rate']
            xdf['Response_Rate'] = xdf['Response_Rate']/sum(xdf['Response_Rate'])
            xdf['%Records'] = xdf['#Records']*100/sum(xdf['#Records'])
            xdf.sort_values('#Records', ascending=[0], inplace=True)
            print(xdf.loc[:,['#Records','%Records','Response_Rate']].head(10))
            print("%d of %d"%(min(10,xdf.shape[0]), xdf.shape[0]))
            del xdf
            print("-"*50)        
    else:
        for col in [i for i in df.columns if (df[i].dtypes=='O') or (len(df[i].unique())<=15)]:
            tot_feats = tot_feats + len(df[col].unique())-1
            print(" %s "% col, '# unique: ',len(df[col].unique()))
            xdf = pd.DataFrame(df[col].value_counts())
            xdf['%Records'] = xdf[col]*100/sum(xdf[col])
            xdf.columns = ['#Records','%Records']
            print(xdf.head())
            del xdf
            print("-"*50)
    ll=[i for i in df.columns if (df[i].dtypes!='O') and (len(df[i].unique())>15)]
    tot_feats= tot_feats + len(ll)
    print("Creating dummies will lead to a total of %d features"% tot_feats)
    return 1
data_preview_str(train, 'train')
data_preview_str(test, 'test')
# Combine Train and Test ( for easier submissions)
train['train_flag'] = 1
test['train_flag'] = 0
full_data = pd.concat([train,test], sort=False)
full_data.groupby('train_flag')['id','project_is_approved'].count()
# Dummy or Flag Features
full_data = pd.get_dummies(full_data, columns =['teacher_prefix','school_state','project_grade_category'])
# full_data.drop(['teacher_prefix','school_state','project_grade_category'], axis=1, inplace=True)
full_data.head()
# Date features
full_data['project_submitted_datetime'] = pd.to_datetime(full_data['project_submitted_datetime'])
full_data['day'] = full_data['project_submitted_datetime'].dt.day
full_data['dayofweek'] = full_data['project_submitted_datetime'].dt.dayofweek
full_data['month'] = full_data['project_submitted_datetime'].dt.month
full_data['year'] = full_data['project_submitted_datetime'].dt.year
full_data.drop('project_submitted_datetime', axis=1, inplace=True)
full_data[['day','dayofweek','month','year']].head()
# Project Title attributes : First impression is a Last-ing one
full_data['pt_caps'] = full_data['project_title'].str.findall(r'[A-Z]').str.len()/full_data['project_title'].str.len()
full_data['pt_special_chars'] = full_data['project_title'].str.findall(r'[^A-Za-z0-9]').str.len()/full_data['project_title'].str.len()
full_data['pt_len'] = full_data['project_title'].str.len()
full_data['pt_words'] = full_data['project_title'].str.findall(r'[\s]').str.len()/full_data['project_title'].str.len()

# project_resource_summary attributes
full_data['prs_caps'] = full_data['project_resource_summary'].str.findall(r'[A-Z]').str.len()/full_data['project_title'].str.len()
full_data['prs_special_chars'] = full_data['project_resource_summary'].str.findall(r'[^A-Za-z0-9]').str.len()/full_data['project_title'].str.len()
full_data['prs_len'] = full_data['project_resource_summary'].str.len()
full_data['prs_words'] = full_data['project_resource_summary'].str.findall(r'[\s]').str.len()/full_data['project_title'].str.len()

# essays
full_data['ess_caps'] = full_data['essays'].str.findall(r'[A-Z]').str.len()/full_data['project_title'].str.len()
full_data['ess_special_chars'] = full_data['essays'].str.findall(r'[^A-Za-z0-9]').str.len()/full_data['project_title'].str.len()
full_data['ess_len'] = full_data['essays'].str.len()
full_data['ess_words'] = full_data['essays'].str.findall(r'[\s]').str.len()/full_data['project_title'].str.len()
# aggregating it based on the project id

rsrc["description"].fillna("", inplace = True)

rsrc_grp = pd.DataFrame(rsrc.groupby("id").agg({"description" : lambda x : "".join(x),
                   "quantity": ["sum", "mean"],
                   "price" : ["sum", "mean"]}))

rsrc_grp.reset_index(inplace = True)
rsrc_grp.columns.droplevel(0)
rsrc_grp.columns= ["id", "description", "quantity_sum",
                "quantity_mean", "price_sum", "price_mean"]
# Merge the aggregated data with the combined Data frame
print("Before merge:",full_data.shape)
full_data = pd.merge(full_data, rsrc_grp, on = "id", how = "left")
print("After merge:",full_data.shape)
del rsrc_grp
text_cols = [
    'project_title', 
    'essays', 
    'project_resource_summary',
    'description'
]
nfeats=5

print(full_data[text_cols].head(1).T)
from tqdm import tqdm

for c in text_cols:
    tfidf = TfidfVectorizer(
        max_features=nfeats,
        norm='l2',
        sublinear_tf = True,
        stop_words = 'english',
        analyzer = 'word',
        min_df = 5,
        max_df = .9,
        smooth_idf = False)
    
    print("*** %s ***"% c)
    print(tfidf)
    
    tfidf.fit(full_data[c])
    tfidf_train = np.array(tfidf.transform(full_data[c]).toarray(), dtype=np.float16)
    for i in range(nfeats):
        full_data[c + '_tfidf_' + str(i)] = tfidf_train[:, i]
    del tfidf, tfidf_train
    gc.collect()
    
print('Done.')

remcols = [
    'id',
    'teacher_id',
    'project_title', 
    'essays', 
    'project_resource_summary',
    'description',
    'project_subject_categories', 'project_subject_subcategories', 'project_is_approved'
]


train_idcol = full_data.loc[full_data['train_flag']==1,'id']
train_teacher_idcol = full_data.loc[full_data['train_flag']==1,:'teacher_id']

test_idcol = full_data.loc[full_data['train_flag']==1,'id']
test_teacher_idcol = full_data.loc[full_data['train_flag']==1,:'teacher_id']


full_data.drop(remcols, axis = 1, inplace = True)

X = full_data.loc[full_data['train_flag']==1, :].reset_index()
y = train["project_is_approved"].reset_index()

print(X.shape, y.shape)

X_test = full_data.loc[full_data['train_flag']==0, :].reset_index()

del train, test

#  Data Splitting for validation

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3)
yTrain.drop('index', axis=1, inplace=True)
yTest.drop('index', axis=1, inplace=True)
xTrain.columns
first_rf = RandomForestClassifier(n_estimators=1,random_state=1)
first_rf.fit(xTrain, yTrain.values.ravel())
importance = first_rf.feature_importances_
importance = pd.DataFrame(importance, index=xTrain.columns, 
                          columns=["Importance"])

importance["Std"] = np.std([tree.feature_importances_
                            for tree in first_rf.estimators_], axis=0)

importance.sort_values('Importance', ascending=[0], inplace=True)

x = importance.index
y = importance['Importance']
yerr = importance['Std']
plt.figure(figsize=(50,5))
plt.bar(x, y, yerr=yerr, align="center")
plt.show()
score_dict = {'AUC': 'roc_auc'}
# clf = RandomForestClassifier(max_depth=2, random_state=0)
clf = GridSearchCV(
    RandomForestClassifier(random_state=0),
    param_grid={
        'n_estimators':[1],
        'max_depth':[2, 4, 8, 10, 12, 14, 16, 18, 20, 22, 24]},
    scoring = score_dict,
    cv=5,refit='AUC')

clfs = clf.fit(xTrain,yTrain.values.ravel())

results = clfs.cv_results_


pd.DataFrame(results['mean_test_AUC']).rename(columns= {0:'AUC'}).plot()

