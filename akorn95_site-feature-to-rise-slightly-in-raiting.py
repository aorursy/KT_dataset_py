import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import pickle
from sklearn.pipeline import Pipeline
train_df = pd.read_csv('../input/train_sessions.csv', index_col='session_id')
test_df = pd.read_csv('../input/test_sessions.csv', index_col='session_id')
site=["site%s" % i for i in range(1, 11)]
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)
y = train_df['target']
full_df = pd.concat([train_df.drop('target', axis=1), test_df])
idx=train_df.shape[0]
full_df[site]=full_df[site].fillna(float(0))
full_df['str0']=full_df[site].apply(lambda x: str( " ".join([str(a) for a in x.values if a != 0])), axis=1)
def get_auc_lr_valid(X, y, C=1.0,  ratio = 0.7):
    # Split the data into the training and validation sets
    idx1 = int(round(X.shape[0] * ratio))
    # Classifier training
    lr =  LogisticRegression(C=C,  n_jobs=-1)
    lr.fit(X[:idx1], y[:idx1])
    # Prediction for validation set
    y_pred = lr.predict_proba(X[idx1:])[:, 1]
    # Calculate the quality
    score = roc_auc_score(y[idx1:], y_pred,  )    
    return score
pipeline0 = Pipeline([("vectorize", TfidfVectorizer(ngram_range=(1, 3), max_features=100000)), ("tfidf", TfidfTransformer())])
pipeline0.fit(full_df['str0'][:idx].ravel(),y)

X_train0 = pipeline0.transform(full_df['str0'][:idx].ravel())
X_test0 = pipeline0.transform(full_df['str0'][idx:].ravel())
for C in range(3,6):
    print(C, get_auc_lr_valid(X_train0, y, C=C))
with open('../input/site_dic.pkl', "rb") as inp_file:
    site_dic = pickle.load(inp_file)

inv_site_dic = {v: k for k, v in site_dic.items()}
inv_site_dic.update({0: ''})
full_df['str']=full_df[site].apply(lambda x: " ".join( [inv_site_dic[a] for a in x.values if a != 0]), axis=1)
full_df['str']=full_df['str'].apply(lambda x: x.replace('.', ' '))
pipeline = Pipeline([("vectorize", TfidfVectorizer(ngram_range=(1, 3), max_features=100000)), ("tfidf", TfidfTransformer())])
pipeline.fit(full_df['str'][:idx].ravel(),y)

X_train = pipeline.transform(full_df['str'][:idx].ravel())
X_test = pipeline.transform(full_df['str'][idx:].ravel())
for C in range(3,6):
    print(C, get_auc_lr_valid(X_train, y, C=C))
LR=LogisticRegression(C=5,  n_jobs=-1)
LR.fit(X_train, y)
test_pred=LR.predict_proba(X_test)
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)
write_to_submission_file(test_pred[:, 1], 'site_feature.csv')