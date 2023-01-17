import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Models Packages
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing, model_selection, metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 

import re
import string
import os
import gc
from datetime import date

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge

# Viz
import seaborn as sns
import matplotlib.pyplot as plt

color = sns.color_palette()
%matplotlib inline

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
## Import necessary files
train_df = pd.read_csv("../input/train.csv", parse_dates=["activation_date"])
test_df = pd.read_csv("../input/test.csv", parse_dates=["activation_date"])
print("Train file rows and columns are : ", train_df.shape)
print("Test file rows and columns are : ", test_df.shape)

testdex = test_df.index
traindex = train_df.index
# New variables #
train_df["activation_weekday"] = train_df["activation_date"].dt.weekday
test_df["activation_weekday"] = test_df["activation_date"].dt.weekday

train_df["activation_month"] = train_df["activation_date"].dt.month
test_df["activation_month"] = test_df["activation_date"].dt.month

train_df["title_nwords"] = train_df["title"].apply(lambda x: len(x.split()))
test_df["title_nwords"] = test_df["title"].apply(lambda x: len(x.split()))

train_df["description"].fillna("NA", inplace=True)
test_df["description"].fillna("NA", inplace=True)
train_df["desc_nwords"] = train_df["description"].apply(lambda x: len(x.split()))
test_df["desc_nwords"] = test_df["description"].apply(lambda x: len(x.split()))

train_df['param123'] = train_df['param_1'].fillna('') + " " + train_df['param_2'].fillna('') + " " + train_df['param_3'].fillna('') 
test_df['param123'] = test_df['param_1'].fillna('') + " " + test_df['param_2'].fillna('') + " " + test_df['param_3'].fillna('') 

#Impute image_top_1
enc = train_df.groupby('category_name')['image_top_1'].agg(lambda x:x.value_counts().index[0]).astype(np.float32).reset_index()
enc.columns = ['category_name' ,'image_top_1_impute']
#Cross Check values
#enc = train_df.loc[train_df['category_name'] == 'Аквариум'].groupby('image_top_1').agg('count')
#enc.sort_values(['item_id'], ascending=False).head(2)

train_df = pd.merge(train_df, enc, how='left', on='category_name')
test_df = pd.merge(test_df, enc, how='left', on='category_name')

del enc
gc.collect()

train_df['image_top_1'].fillna(train_df['image_top_1_impute'], inplace=True)
test_df['image_top_1'].fillna(test_df['image_top_1_impute'], inplace=True)

#Impute Days diff
#enc = train_df.groupby('category_name')['days'].agg('median').astype(np.float32).reset_index()
#enc.columns = ['category_name' ,'days_impute']
#Cross Check values
#enc = train_df.loc[train_df['category_name'] == 'Аквариум'].groupby('image_top_1').agg('count')
#enc.sort_values(['item_id'], ascending=False).head(2)

#train_df = pd.merge(train_df, enc, how='left', on='category_name')
#test_df = pd.merge(test_df, enc, how='left', on='category_name')

#train_df['days'].fillna(train_df['days_impute'], inplace=True)
#test_df['days'].fillna(test_df['days_impute'], inplace=True)


#Create image flag 
test_df['image'] = test_df['image'].map(lambda x: 1 if len(str(x)) >0 else 0)
train_df['image'] = train_df['image'].map(lambda x: 1 if len(str(x)) >0 else 0)

# City names are duplicated across region, HT: Branden Murray 
#https://www.kaggle.com/c/avito-demand-prediction/discussion/55630#321751
train_df['city'] = train_df['city'] + "_" + train_df['region']
test_df['city'] = test_df['city'] + "_" + test_df['region']

train_df['price'].fillna(0, inplace=True)
test_df['price'].fillna(0, inplace=True)
train_df['price'] = np.log1p(train_df['price'])
test_df['price'] = np.log1p(test_df['price'])

price_mean = train_df['price'].mean()
price_std = train_df['price'].std()
train_df['price'] = (train_df['price'] - price_mean) / price_std
test_df['price'] = (test_df['price'] - price_mean) / price_std
cat_cols = ['category_name', 'image_top_1']
num_cols = ['price', 'deal_probability']

for c in cat_cols:
    for c2 in num_cols:
        enc = train_df.groupby(c)[c2].agg(['median']).astype(np.float32).reset_index()
        enc.columns = ['_'.join([str(c), str(c2), str(c3)]) if c3 != c else c for c3 in enc.columns]
        train_df = pd.merge(train_df, enc, how='left', on=c)
        test_df = pd.merge(test_df, enc, how='left', on=c)
        
del cat_cols, num_cols, c, c2, enc
gc.collect()
def cleanName(text):
    try:
        textProc = text.lower()
        # textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        #regex = re.compile(u'[^[:alpha:]]')
        #textProc = regex.sub(" ", textProc)
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except: 
        return "name error"
# Meta Text Features
textfeats = ["description", "title"]
train_df['desc_punc'] = train_df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
test_df['desc_punc'] = test_df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

train_df['title'] = train_df['title'].apply(lambda x: cleanName(x))
train_df["description"]   = train_df["description"].apply(lambda x: cleanName(x))

test_df['title'] = test_df['title'].apply(lambda x: cleanName(x))
test_df["description"]   = test_df["description"].apply(lambda x: cleanName(x))

for cols in textfeats:
    train_df[cols] = train_df[cols].astype(str) 
    train_df[cols] = train_df[cols].astype(str).fillna('missing') # FILL NA
    train_df[cols] = train_df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    train_df[cols + '_num_words'] = train_df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    train_df[cols + '_num_unique_words'] = train_df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    train_df[cols + '_words_vs_unique'] = train_df[cols+'_num_unique_words'] / train_df[cols+'_num_words'] * 100 # Count Unique Words
    train_df[cols + '_num_letters'] = train_df[cols].apply(lambda comment: len(comment)) # Count number of Letters

    test_df[cols] = test_df[cols].astype(str) 
    test_df[cols] = test_df[cols].astype(str).fillna('missing') # FILL NA
    test_df[cols] = test_df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    test_df[cols + '_num_words'] = test_df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    test_df[cols + '_num_unique_words'] = test_df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    test_df[cols + '_words_vs_unique'] = test_df[cols+'_num_unique_words'] / test_df[cols+'_num_words'] * 100 # Count Unique Words
    test_df[cols + '_num_letters'] = test_df[cols].apply(lambda comment: len(comment)) # Count number of Letters
    
# Extra Feature Engineering
train_df['title_desc_len_ratio'] = train_df['title_num_letters']/train_df['description_num_letters']
test_df['title_desc_len_ratio'] = test_df['title_num_letters']/test_df['description_num_letters']

print(train_df.head())
print(test_df.head())
# Label encode the categorical variables #
cat_vars = ["region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param123"]
for col in cat_vars:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

cols_to_drop = ["item_id", "user_id", "activation_date", "image", "param_2", "param_3"
                , "image_top_1_impute"] #,"days_impute"]
train_X = train_df.drop(cols_to_drop + ["deal_probability"], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

train_y = train_df["deal_probability"].values
test_id = test_df["item_id"].values

del train_df, test_df
gc.collect()
russian_stop = set(stopwords.words('russian'))

tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    #"min_df":5,
    #"max_df":.9,
    "smooth_idf":False
}
def get_col(col_name): return lambda x: x[col_name]
##I added to the max_features of the description. It did not change my score much but it may be worth investigating
vectorizer = FeatureUnion([
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=17000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('title',CountVectorizer(
            ngram_range=(1, 2),
            stop_words = russian_stop,
            #max_features=7000,
            preprocessor=get_col('title')))
    ])

#Fit my vectorizer on the entire dataset instead of the training rows
#Score improved by .0001
df = pd.concat([train_X,test_X],axis=0)
vectorizer.fit(df.to_dict('records'))
ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
df.drop(textfeats, axis=1,inplace=True)
sparse_m = csr_matrix(df.values)
complete = hstack([sparse_m,ready_df])
tfvocab = df.columns.tolist() + tfvocab
complete = complete.tocsr()

print(complete.shape)
print(ready_df.shape)

del sparse_m, ready_df, df
gc.collect()
train_X = complete[0:traindex.shape[0],]
test_X = complete[traindex.shape[0]:,]

del complete 
gc.collect()
train_X.shape
#split the train into development and validation sample. Take the last 100K rows as validation sample.
# Splitting the data for model training#
dev_X = train_X[:-200000,:]
val_X = train_X[-200000:,:]
dev_y = train_y[:-200000]
val_y = train_y[-200000:]
print(dev_X.shape, val_X.shape, test_X.shape)

del train_X, train_y
gc.collect()
#custom function to build the LightGBM model.
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 300,
        "learning_rate" : 0.01,
        "bagging_fraction" : 0.75,
        "feature_fraction" : 0.5,
        "bagging_freq" : 2,
        "bagging_seed" : 2018,
        "verbosity" : -1,
        "max_depth": 18,
        "min_child_samples":100
       # ,"boosting":"rf"
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y, feature_name=tfvocab)
    lgval = lgb.Dataset(val_X, label=val_y, feature_name=tfvocab)
    
    del train_X, val_X; gc.collect()
    
    evals_result = {}
    model = lgb.train(params, lgtrain, 10000, valid_sets=[lgval], early_stopping_rounds=50, verbose_eval=200, evals_result=evals_result)
    
    #model = lgb.cv(params, lgtrain, 1000, early_stopping_rounds=20, verbose_eval=20, stratified=False )
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    del lgtrain, lgval; gc.collect()
    return pred_test_y, model, evals_result
# Training the model #
import lightgbm as lgb
pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
# Plot importance
lgb.plot_importance(model, importance_type="split", title="split",max_num_features=50, figsize=(7,10))
plt.show()

lgb.plot_importance(model, importance_type="gain", title='gain', max_num_features=50, figsize=(7,10))
plt.show()

# Importance values are also available in:
print(model.feature_importance("split"))
print(model.feature_importance("gain"))
# Making a submission file #
pred_test[pred_test>1] = 1
pred_test[pred_test<0] = 0
sub_df = pd.DataFrame({"item_id":test_id})
sub_df["deal_probability"] = pred_test
sub_df.to_csv("baseline_lgb.csv", index=False)