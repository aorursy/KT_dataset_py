import warnings
warnings.filterwarnings('ignore')
# Import libraries 
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
import math
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
# Read the training and test data sets
train_df = pd.read_csv('train_sessions.csv', index_col='session_id')
test_df = pd.read_csv('test_sessions.csv', index_col='session_id')
# Switch time1, ..., time10 columns to datetime type
times = ['time%s' % i for i in range(1, 11)]
sites = ['site%s' % i for i in range(1, 11)]

train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

#Alice
Alice_df = train_df[train_df['target']==1]

# Our target variable
y_train = train_df['target']

# United dataframe of the initial data 
full_df = pd.concat([train_df.drop('target', axis=1), test_df])

# Sort the data by time
train_df = train_df.sort_values(by='time1')
# Fill sites NA-values with zeros
train_df[sites] = train_df[sites].fillna(0).astype('int')
test_df[sites] = test_df[sites].fillna(0).astype('int')

# Load websites dictionary
with open(r"site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

    
# Create dataframe for the dictionary
sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])
print(u'Websites total:', sites_dict.shape[0])
def get_auc_lr_valid(X, y, C=1.0, seed=17, ratio = 0.7, n_jobs = -1):
    # Split the data into the training and validation sets
    idx = int(round(X.shape[0] * ratio))
    X_train = X[:idx, :]
    X_valid = X[idx:, :]
    y_train = y[:idx]
    y_valid = y[idx:]
    # Classifier training
    lr = LogisticRegression(C=C, random_state=seed).fit(X_train, y_train)
    #penalty='l2'
    
    # Prediction for validation set
    y_pred = lr.predict_proba(X_valid)[:, 1]
    # Calculate the quality
    score = roc_auc_score(y_valid, y_pred)
    
    return score

# Function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)
def calc_auc(X_train_sparse, y_train):
    result = []
    for C in np.logspace(-3, 2, 50):
        result.append( [C, get_auc_lr_valid(X_train_sparse, y_train, C=C)])   
    print (result)

def find_C(result):
    max_C = 0
    for i in range (len(result)):
        if result[i][1] > max_C:
            max_C = result[i][1]
            C = result[i][0]
    return max_C,C
train_to_text = train_df[sites].apply(
    lambda x: " ".join([str(a) for a in x.values if a != 0]), axis=1)\
               .values.reshape(len(train_df[sites]), 1)
test_to_text = test_df[sites].apply(
    lambda x: " ".join([str(a) for a in x.values if a != 0]), axis=1)\
               .values.reshape(len(test_df[sites]), 1)
pipeline = Pipeline([
    ("vectorize", CountVectorizer()),
    ("tfidf", TfidfTransformer())
])
pipeline.fit(train_to_text.ravel())

X_train_sparse = pipeline.transform(train_to_text.ravel())
X_test_sparse = pipeline.transform(test_to_text.ravel())

X_train_sparse.shape, X_test_sparse.shape
data = pd.DataFrame(index=train_df.index)
data = train_df[sites]
data['list_of_words'] = train_df['site1'].apply(str)
data['list_of_words'] += ','
for i in range(2, 10):
    data['list_of_words'] += train_df['site%s'% i].apply(str)
    data['list_of_words'] += ','
train_df['list_of_words'] = data['list_of_words'].apply(lambda x: x.split(','))
data = pd.DataFrame(index=test_df.index)
data = test_df[sites]
data['list_of_words'] = test_df['site1'].apply(str)
data['list_of_words'] += ','
for i in range(2, 10):
    data['list_of_words'] += test_df['site%s'% i].apply(str)
    data['list_of_words'] += ','
test_df['list_of_words'] = data['list_of_words'].apply(lambda x: x.split(','))
from gensim.models import word2vec
test_df['target'] = -1
data = pd.concat([train_df,test_df],axis=0)
text_model = word2vec.Word2Vec(data['list_of_words'], size=500, window=5, workers=-1)
w2v = dict(zip(text_model.wv.index2word, text_model.wv.syn0))
class sense_vectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(w2v.values())))

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[word] for word in words if word in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
    def fit(self, X):
        return self 
data_sentence=average_vectorizer(w2v).fit(train_df['list_of_words']).transform(train_df['list_of_words'])
data_sentence.shape
y = train_df['target']
get_auc_lr_valid(data_average, y)
find_C(calc_auc(data_average, y))
domains = ['domain%s' % i for i in range(1, 11)]
new_sites_dict = {y:x for x,y in site_dict.items()}
domain_array = ['0','30','40','50','60','be','ca','ch','cn','com','edu','fr','net','org','ru','tt','tv','uk','us']
new_set = pd.DataFrame(index=train_df.index)
new_set = train_df[sites]
for i in range(1, 11):
    new_set['domain%s'% i] = new_set['site%s'% i].map(new_sites_dict).str.split('.').str[-1]
feat_train['unique_domains'] = new_set[domains].apply(lambda x: len(set(a for a in x.values if a != 0)), 1)

new_set = pd.DataFrame(index=train_df.index)
new_set = test_df[sites]
for i in range(1, 11):
    new_set['domain%s'% i] = new_set['site%s'% i].map(new_sites_dict).str.split('.').str[-1]
feat_test['unique_domains'] = new_set[domains].apply(lambda x: len(set(a for a in x.values if a != 0)), 0)
new_subset = ("unique_domains",)
to_pp(feat_train['unique_domains'], feat_test['unique_domains'])
X_train_sparse_new, X_test_sparse_new = add_features(new_subset)
get_auc_lr_valid(X_train_sparse_new, y_train, C=0.5)
def fit_domain_feature(lmdb, feature, reason, axis = 0):
    feat_train[feature] = domains_df_train[reason].apply(lmbd, axis)
    feat_test[feature] = domains_df_test[reason].apply(lmbd, axis)
    if axis != 0:
        feat_train[feature].values.reshape(len(domains_df_train[reason]), 1)
        feat_test[feature].values.reshape(len(domains_df_test[reason]), 1)
    scaler = StandardScaler()
domains_df_train = pd.DataFrame()
for i in range(1, 11):
    domains_df_train['domain%s'% i] = train_df['site%s'% i].map(new_sites_dict).str.split('.').str[-1]

top_domains = pd.Series(alice_domain_set.fillna(0).values.flatten()
                     ).value_counts().sort_values(ascending=False).head(20)

domains_df_test = pd.DataFrame()
for i in range(1, 11):
    domains_df_test['domain%s'% i] = test_df['site%s'% i].map(new_sites_dict).str.split('.').str[-1]
for domain in domain_array:
    domain_feature = "in_domain_" + domain

    lmbd = lambda x: 1 if (domain in x.values) == True else 0
    axis=1
    feature = domain_feature
    reason = domains
    fit_domain_feature(lmbd,feature,reason,axis)
array = domain_array

auc2 = {}
for element in array:
        new_subset = subset + ("in_domain_"+element,)
        X_train_sparse_new, X_test_sparse_new = add_features(new_subset)
        r = get_auc_lr_valid(X_train_sparse_new, y_train, C=0.5)
        auc2[new_subset] = r
        print(new_subset, r)
t = sorted(auc2.items(), key=lambda x:-x[1])[:20]
for x in t:
    print(x)
predomain_array = ['0','196','197','acsta','annotathon','audienceinsights','baidu','bing','facebook','geotrust','ggpht',
                   'google','googleapis','googlevideo','info-jeunes','leboncoin','live','melty','microsoft','nih','nocookie',
                   'oracle','phylogeny','twitter','univ-bpclermont','vk','wikimedia','yahoo','youtube','ytimg']
def fit_predomain_feature(lmdb, feature, reason, axis = 0):
    feat_train[feature] = predomains_df_train[reason].apply(lmbd, axis)
    feat_test[feature] = predomains_df_test[reason].apply(lmbd, axis)
    if axis != 0:
        feat_train[feature].values.reshape(len(predomains_df_train[reason]), 1)
        feat_test[feature].values.reshape(len(predomains_df_test[reason]), 1)
    scaler = StandardScaler()
predomains_df_train = pd.DataFrame()
for i in range(1, 11):
    predomains_df_train['domain%s'% i] = train_df['site%s'% i].map(new_sites_dict).str.split('.').str[-2]

predomains_df_test = pd.DataFrame()
for i in range(1, 11):
    predomains_df_test['domain%s'% i] = test_df['site%s'% i].map(new_sites_dict).str.split('.').str[-2]
for predomain in predomain_array:
    predomain_feature = "in_predomain_" + predomain

    lmbd = lambda x: 1 if (predomain in x.values) == True else 0
    axis=1
    feature = predomain_feature
    reason = domains
    fit_predomain_feature(lmbd,feature,reason,axis)
array = predomain_array

auc3 = {}
for element in array:
        new_subset = subset + ("in_predomain_"+element,)
        X_train_sparse_new, X_test_sparse_new = add_features(new_subset)
        r = get_auc_lr_valid(X_train_sparse_new, y_train, C=0.5)
        auc3[new_subset] = r
        print(new_subset, r)
t = sorted(auc3.items(), key=lambda x:-x[1])[:20]
for x in t:
    print(x)