import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords 



import pandas as pd

import numpy as np # linear algebra



from scipy.sparse import hstack, csr_matrix

import seaborn as sns

# https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lb-0-2241



from sklearn.model_selection import train_test_split

import lightgbm as lgb



SEED = 23
stopWords = set(stopwords.words('english'))

stopWords
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

traindex = train.index



test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

testdex = test.index
train.head(5)
test_id = test.id.values
import gc

y = train.target.values

train.drop(['target'],axis=1, inplace=True)

df = pd.concat([train,test],axis=0)



to_drop = ['id', 'location']

df.drop(to_drop, axis=1, inplace=True)

df.reset_index(drop=True, inplace=True)

del train, test

gc.collect()
from sklearn.model_selection import KFold



kf = KFold(n_splits=5, shuffle=False)

cols = ['keyword']

train_new = df.loc[traindex,:].copy()

train_new['target'] = y



for train_index, test_index in kf.split(train_new):

    X_tr, X_val = train_new.iloc[train_index], train_new.iloc[test_index]

    for col in cols:

        item_id_target_mean = X_tr.groupby(col).target.mean()

        X_val[col + 'mean_enc'] = X_val[col].map(item_id_target_mean)

#         X_tr[col + 'mean_enc'] = X_tr[col].map(item_id_target_mean)

    train_new.iloc[test_index] = X_val

#     train_new.iloc[train_index] = X_tr



prior = y.mean()

train_new.fillna(prior, inplace=True)
# Calculate a mapping: {item_id: target_mean}

train_new = df.loc[traindex,:].copy()

train_new['target'] = y

keyword_target_mean = train_new.groupby('keyword').target.mean()



# In our non-regularized case we just *map* the computed means to the `item_id`'s

df.loc[traindex,'keyword_target_enc'] = df.loc[testdex,'keyword'].map(keyword_target_mean)



# Fill NaNs

df['keyword_target_enc'].fillna(y.mean(), inplace=True) 

df.drop(['keyword'],axis=1, inplace=True)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), dpi=100)

df_train = df.loc[traindex,:].copy()

df_test = df.loc[testdex,:].copy()

DISASTER_TWEETS = y == 1



for i, feature in enumerate(['keyword_target_enc']):

    sns.distplot(df_train.loc[~DISASTER_TWEETS][feature], label='Not Disaster', ax=axes[0], color='green')

    sns.distplot(df_train.loc[DISASTER_TWEETS][feature], label='Disaster', ax=axes[0], color='red')

# plt.legend()

    sns.distplot(df_train[feature], label='Training', ax=axes[1])

    sns.distplot(df_test[feature], label='Test', ax=axes[1])

    axes[0].legend()

    axes[1].legend()
# Meta Text Features

textfeats = ["text"]

for cols in textfeats:

    df[cols] = df[cols].astype(str) 

    df[cols] = df[cols].astype(str).fillna('nicapotato') # FILL NA

    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently

    df[cols + '_num_chars'] = df[cols].apply(len) # Count number of Characters

    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words

    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))

    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words



print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")

stopWords = set(stopwords.words('english'))
import numpy as np

tfidf_para = {

    "stop_words": stopWords,

    "analyzer": 'word',

    "token_pattern": r'\w{1,}',

    "sublinear_tf": True,

    "dtype": np.float32,

    "norm": 'l2',

    #"min_df":5,

    #"max_df":.9,

    "smooth_idf":False

}
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(ngram_range=(1,2), max_features=17000, **tfidf_para)

vect.fit(df.loc[traindex,:].text.values)

ready_df = vect.transform(df.text.values)
df.head(4)
ready_df.shape
# X = hstack([csr_matrix(df.drop(['id', 'keyword', 'location', 'text'],axis=1).values),ready_df]) # Sparse Matrix

X = hstack([csr_matrix(df.iloc[traindex,1:].values),ready_df[0:traindex.shape[0]]])
X_train, X_valid, y_train, y_valid = train_test_split(

    X, y, test_size=0.10, random_state=SEED)
lgbm_params =  {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': ['auc', 'f1'],

    'max_depth': 16,

    'num_leaves': 37,

    'feature_fraction': 0.6,

    'bagging_fraction': 0.8,

    # 'bagging_freq': 5,

    'learning_rate': 0.019,

    'verbose': 0

}  



# LGBM Dataset Formatting 

lgtrain = lgb.Dataset(X_train, y_train)#,

#                 feature_name=tfvocab,

#                 categorical_feature = categorical)

lgvalid = lgb.Dataset(X_valid, y_valid)#,

#                 feature_name=tfvocab,

#                 categorical_feature = categorical)
from sklearn.metrics import f1_score



def lgb_f1_score(y_hat, data):

    y_true = data.get_label()

    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities

    return 'f1', f1_score(y_true, y_hat), True



evals_result = {}



lgb_clf = lgb.train(

    lgbm_params,

    lgtrain,

    num_boost_round=16000,

    valid_sets=[lgtrain, lgvalid],

    valid_names=['train','valid'],

    early_stopping_rounds=200,

    verbose_eval=200,

    feval=lgb_f1_score, evals_result=evals_result)



lgb.plot_metric(evals_result, metric='f1')
testing = hstack([csr_matrix(df.iloc[testdex,1:].values),\

                  ready_df[traindex.shape[0]:]])



lgpred = lgb_clf.predict(testing)

lgsub = pd.DataFrame(lgpred,columns=["target"],index=testdex)

lgsub['target'].clip(0.0, 1.0, inplace=True) # Between 0 and 1

lgsub['id'] = test_id

lgsub.to_csv("submission.csv",index=False,header=True)
import seaborn as sns

sns.distplot(lgsub.target.values)
lgpred = lgb_clf.predict(X)

train_new = df.loc[traindex,:].copy()

train_new['target'] = y

train_new['pred'] = lgpred
sns.distplot(train_new.target)
train_new.sample(20)