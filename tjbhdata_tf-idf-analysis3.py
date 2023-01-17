from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import time
notebookstart= time.time()
from io import StringIO


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import random
random.seed(2018)
print("Data:\n",os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 

# Viz
#import seaborn as sns
import matplotlib.pyplot as plt
import re
import string

#from gby_functions. import *

import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff


init_notebook_mode(connected=True) #do not miss this line

NFOLDS = 5
SEED = 2
VALID = False

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
        
def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
def cleanName(text):
    try:
        textProc = text.lower()
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except: 
        return "name error"
    
    
def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))
training = pd.read_csv('../input/avito-demand-prediction/train.csv', index_col = "item_id", parse_dates = ["activation_date"])
traindex = training.index
testing = pd.read_csv('../input/avito-demand-prediction/test.csv', index_col = "item_id", parse_dates = ["activation_date"])
testdex = testing.index
print("\n Finished Data Load Stage")

training.columns
training.head(5)
testing.head(5)
training.reset_index(inplace = True,drop = False)
testing.reset_index(inplace = True,drop = False)
#### Quick data exploration

from io import StringIO

russian_en1 = StringIO(u"""\
parent_category_name,parent_category_name_en
Личные вещи,Personal belongings
Для дома и дачи,For the home and garden
Бытовая электроника,Consumer electronics
Недвижимость,Real estate
Хобби и отдых,Hobbies & leisure
Транспорт,Transport
Услуги,Services
Животные,Animals
Для бизнеса,For business
""")

russian_en1_df = pd.read_csv(russian_en1)
training = pd.merge(training, russian_en1_df, on="parent_category_name", how="left")
plt.figure(figsize=(15,5))
sns.distplot(training["deal_probability"].values, bins=120, color="#ff002e")
plt.xlabel('Deal Probability', fontsize=14);
plt.title("Distribution of Deal Probability", fontsize=14);
plt.style.use('ggplot')
plt.show();
plt.figure(figsize=(12,8))
sns.boxplot(y="parent_category_name_en", x="deal_probability", data=training)
plt.xlabel('Deal probability', fontsize=12)
plt.ylabel('Parent Category', fontsize=12)
plt.title("Deal probability by Parent Category")
plt.xticks(rotation='vertical')
plt.show()

training.drop('parent_category_name_en',axis = 1,inplace = True)
training.describe()

plt.figure(figsize=(15,5))
plt.style.use('ggplot')

plt.scatter(training.loc[(training['param_1'] == 'Samsung')&(training['price']<1e5),'price'],\
            training.loc[(training['param_1'] == 'Samsung')&(training['price']<1e5),'deal_probability'])
plt.xlabel('Price', fontsize=14);
plt.ylabel('Deal Probability', fontsize=14);
plt.title("Deal Probability sensitivity with Price for Samsung Phones", fontsize=14);
#plt.style.use('ggplot')
plt.show();
colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]

fig = ff.create_2d_density(
    training.loc[(training['param_1'] == 'Samsung')&(training['price']<1e5)&\
                 (training['deal_probability']>0),'price'], 
    training.loc[(training['param_1'] == 'Samsung')&(training['price']<1e5)&\
                 (training['deal_probability']>0),'deal_probability'], colorscale=colorscale,
    hist_color='rgb(255, 237, 222)', point_size=2
)
fig.layout.update({'title': 'Deal Probability sensitivity with Price for Samsung Phones'})
fig['layout']['yaxis1'].update(title='Deal Probability')
fig['layout']['xaxis1'].update(title='Price of Phone')


py.offline.iplot(fig, filename='histogram_subplots')
training['dayofweek'] = training['activation_date'].dt.dayofweek
dayofweek_count = training['dayofweek'].value_counts()
trace = go.Bar(
    x=dayofweek_count.index,
    y=dayofweek_count.values,
    orientation = 'v',
    marker=dict(
        color=dayofweek_count.values,
        colorscale = 'Jet',
        reversescale = True
    ),
)

layout = dict(
    title='Ad Counts by day of week',
    yaxis=dict(
        title='Counts'
    ),
    xaxis=dict(
        title='Day of Week'
    ),
    height=400
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename="category name")
print("Combine Train and Test")
ntrain = training.shape[0]
ntest = testing.shape[0]

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)

y = training.deal_probability.copy()
training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))


df = pd.concat([training,testing],axis=0)
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))
def do_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    #df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )


# In[ ]:


def do_cumcount( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Cumulative count by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    #df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )


# In[ ]:


def do_countuniq( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Counting unqiue ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    #df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )


# In[ ]:


def do_mean( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating mean of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    #df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )


# In[ ]:


def do_median( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating mean of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].median().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    #df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )


def do_var( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating variance of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )
df["price"] = np.log(df["price"]+0.001)
df["Weekday"] = df['activation_date'].dt.weekday
df["Day of Month"] = df['activation_date'].dt.day
df["price"].fillna(-999,inplace=True)
df["image_top_1"].fillna(-999,inplace=True)


df = do_count( df, ['region', 'param_1'], 'region_param_1', show_max=True ); gc.collect()
df = do_count( df, ['region', 'Day of Month','param_1'], 'region_day_param_1', show_max=True ); gc.collect()
df = do_count( df, ['city', 'category_name'], 'city_category_name', show_max=True ); gc.collect()
df = do_count( df, ['param_1' ], 'param_1_count', show_max=True ); gc.collect()

df = do_countuniq( df, ['city'], 'param_1', 'X2', 'uint8', show_max=True ); gc.collect()
df = do_countuniq( df, ['city'], 'param_2', 'X3', 'uint8', show_max=True ); gc.collect()



df = do_mean( df, [ 'param_1'],'price', 'param_1_mean_price', show_max=True ); gc.collect()
df = do_mean( df, [ 'param_2'],'price', 'param_2_mean_price', show_max=True ); gc.collect()



df = do_var( df, [ 'param_1'],'price', 'param_1_var_price', show_max=True ); gc.collect()
df = do_var( df, [ 'param_2'],'price', 'param_2_var_price', show_max=True ); gc.collect()


df = do_median( df, [ 'param_1'],'item_seq_number', 'param_1_median_seq', show_max=True ); gc.collect()
df = do_median( df, [ 'category_name'],'item_seq_number', 'category_median_seq', show_max=True ); gc.collect()
df = do_median( df, [ 'param_2'],'item_seq_number', 'param_2_median_seq', show_max=True ); gc.collect()

df["param_1_mean_price"].fillna(-999,inplace=True)
df["param_2_mean_price"].fillna(-999,inplace=True)
df["region_param_1"].fillna(-999,inplace=True)
df["region_day_param_1"].fillna(-999,inplace=True)
df["param_1_count"].fillna(-999,inplace=True)


df['price_minus_mean'] = df['price'] - df['param_1_mean_price']
df['price2_minus_mean'] = df['price'] - df['param_2_mean_price']

df.head(5)

# Meta Text Features
textfeats = ["description", "title"]
df['desc_punc'] = df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

df['title'] = df['title'].apply(lambda x: cleanName(x))
df["description"]   = df["description"].apply(lambda x: cleanName(x))

for cols in textfeats:
    df[cols] = df[cols].astype(str) 
    df[cols] = df[cols].astype(str).fillna('missing') # FILL NA
    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words
    df[cols + '_num_letters'] = df[cols].apply(lambda comment: len(comment)) # Count number of Letters
    df[cols + '_num_alphabets'] = df[cols].apply(lambda comment: (comment.count(r'[a-zA-Z]'))) # Count number of Alphabets
    df[cols + '_num_alphanumeric'] = df[cols].apply(lambda comment: (comment.count(r'[A-Za-z0-9]'))) # Count number of AlphaNumeric
    df[cols + '_num_digits'] = df[cols].apply(lambda comment: (comment.count('[0-9]'))) # Count number of Digits
    
# Extra Feature Engineering
df['title_desc_len_ratio'] = df['title_num_letters']/df['description_num_letters']
training_index = df.loc[df.activation_date<=pd.to_datetime('2017-04-07')].index
validation_index = df.loc[df.activation_date>=pd.to_datetime('2017-04-08')].index
df.drop(["activation_date","image"],axis=1,inplace=True)

print("\nEncode Variables")
categorical = ["user_id","region","city","parent_category_name","category_name","user_type","image_top_1","param_1","param_2","param_3"]
print("Encoding :",categorical)

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col].fillna('Unknown')
    df[col] = lbl.fit_transform(df[col].astype(str))
    
df.set_index('item_id',inplace = True)
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
            max_features=1000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('title',CountVectorizer(
            ngram_range=(1, 2),
            stop_words = russian_stop,
            max_features=100,
            preprocessor=get_col('title')))
    ])
    
start_vect=time.time()

#Fit my vectorizer on the entire dataset instead of the training rows
#Score improved by .0001
vectorizer.fit(df.to_dict('records'))

ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))

# Drop Text Cols
textfeats = ["description", "title",'Day of Month']
df.drop(textfeats, axis=1,inplace=True)


from sklearn.metrics import mean_squared_error
from math import sqrt

ridge_params = {'alpha':30.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}

#Ridge oof method from Faron's kernel
#I was using this to analyze my vectorization, but figured it would be interesting to add the results back into the dataset
#It doesn't really add much to the score, but it does help lightgbm converge faster
ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:ntrain], y, ready_df[ntrain:])

rms = sqrt(mean_squared_error(y, ridge_oof_train))
print('Ridge OOF RMSE: {}'.format(rms))

ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])

df['ridge_preds'] = ridge_preds

# Combine Dense Features with Sparse Text Bag of Words Features
X = hstack([csr_matrix(df.loc[traindex,:].values),ready_df[0:traindex.shape[0]]]) # Sparse Matrix
testing = hstack([csr_matrix(df.loc[testdex,:].values),ready_df[traindex.shape[0]:]])
tfvocab = df.columns.tolist() + tfvocab
for shape in [X,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))
del df
gc.collect();



del ridge_preds,vectorizer,ready_df
gc.collect();
    
print("Light Gradient Boosting Regressor")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    # 'max_depth': 15,
    'num_leaves': 30,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.75,
    'bagging_freq': 2,
    'learning_rate': 0.3,
    'verbose': 0
}  



# LGBM Dataset Formatting 
import datetime
print(datetime.datetime.now())
lgtrain_cv = lgb.Dataset(X, y,
                    feature_name=tfvocab,
                    categorical_feature = categorical)
    
   
cv_results = lgb.cv(
            lgbm_params,
            lgtrain_cv,
            num_boost_round=600,
            nfold=5,
            early_stopping_rounds=50,
            verbose_eval=100,
            stratified=False
            )

print("LGB CV modeling complete Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))
print(datetime.datetime.now())

print("done")
