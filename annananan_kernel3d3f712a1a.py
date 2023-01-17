# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.utils import column_or_1d
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv", index_col = "id")
test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv", index_col = "id")
train_test = pd.concat([train, test], sort=False)
train_size = len(train)
train.head()
for col in filter(lambda s: s.startswith("nom"), train_test.columns):
    print(col, ":", len(set(train[col])))
def encode_bin(dataframe):
    label_enc = LabelEncoder()
    encoded_df = pd.DataFrame()
    for cat in filter(lambda s: s.startswith("bin"), dataframe.columns):
        encoded_df[cat] = label_enc.fit_transform(dataframe[cat].astype(str))
    return encoded_df
        
def encode_nom(dataframe):
    encoded_df = pd.get_dummies(dataframe[['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5']])
    return encoded_df

def encode_date(dataframe):
    # TODO: maybe handle cyclic features using sin and cos
    encoded_df = pd.get_dummies(dataframe[['day', 'month']])
    return encoded_df
class LexicoGraphicalLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique().sort()
        return self
def encode_ord(dataframe):
    ord_1_sorted = ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster']
    ord_2_sorted = ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']
    ord_features_map = {'ord_1': {k: v for v, k in enumerate(ord_1_sorted)}, 
                        'ord_2': {k: v for v, k in enumerate(ord_2_sorted)}}
    le = LexicoGraphicalLabelEncoder()

    df = pd.DataFrame()
    for cat in filter(lambda s: s.startswith("ord"), train.columns):
        if cat == 'ord_0':
            df[cat] = dataframe[cat]
        elif cat in ord_features_map:
            df[cat] = dataframe[cat].map(ord_features_map[cat])
        else:
            df[cat] = le.fit(dataframe[cat]).transform(dataframe[cat])
    return df
def preproccessing(dataframe):
    return pd.concat([encode_bin(train_test), 
                      encode_nom(train_test), 
                      encode_ord(train_test),
                      encode_date(train_test)], axis=1)
preproccessed = preproccessing(train.iloc[:,:-1])
X, X_test = preproccessed.iloc[:len(train),:], preproccessed.iloc[len(train):,:]
y = train.iloc[:,-1]
scaler = StandardScaler()
X_scaled = scaler.fit(X).transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
logreg = LogisticRegression(C=0.095, class_weight={0: 1, 1: 1.5}, tol = 0.00001, 
                            solver='liblinear', penalty='l2')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_val)
y_pred_train = logreg.predict(X_train)
print("train auc score:", roc_auc_score(y_train, y_pred_train))
print("val auc score:", roc_auc_score(y_val, y_pred))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
# max depth is small to avoid overfitting
rf = RandomForestClassifier(n_estimators = 100, verbose = 2, max_depth = 10)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_val)[:,1]
rf_probs_train = rf.predict_proba(X_train)[:,1]
roc_value_train = roc_auc_score(y_train, rf_probs_train)
roc_value_val = roc_auc_score(y_val, rf_probs)
print("train auc score:", roc_value_train)
print("val auc score:", roc_value_val)
rf_probs_test = rf.predict_proba(X_test)[:,1]
submission = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv', index_col='id')
submission['target'] = rf_probs_test
submission.to_csv('result.csv')
submission.head()
import os
os.chdir('/kaggle/working')
from IPython.display import FileLink
FileLink('result.csv')
parameters = {
    'n_estimators'      : [150],
    'max_depth'         : [8, 9, 10, 11, 12],
    'random_state'      : [0],
    'max_features': ['auto'],
    'criterion' : ['gini']
}
rf = GridSearchCV(RandomForestClassifier(), parameters, cv=10, n_jobs=-1)
best_rf = clf.fit(X_train, y_train)
parameters = {
    "C": np.logspace(-3, 3, 7), 
    "penalty":["l1","l2"]}
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,cv = 10)
best_logreg = logreg_cv.fit(x_train,y_train)