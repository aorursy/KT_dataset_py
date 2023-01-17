import pandas as pd
import numpy as np 
from sklearn.svm import SVC
import dask
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
fnc = pd.read_csv('../input/trends-assessment-prediction/fnc.csv')
loading = pd.read_csv('../input/trends-assessment-prediction/loading.csv')
labels = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv')
sites = pd.read_csv('../input/trends-assessment-prediction/reveal_ID_site2.csv')
loading = loading.drop(['IC_20'], axis=1)
fnc_features, loading_features = list(fnc.columns[1:]), list(loading.columns[1:])
df = fnc.merge(loading, on="Id")
sites = np.array(sites).reshape(sites.shape[0])
site1 = df[~df['Id'].isin(set(sites))]
site2 = df[df['Id'].isin(set(sites))]
site1['Label'] = 0
site2['Label'] = 1
FNC_SCALE = 1/600

site1[fnc_features] *= FNC_SCALE
site2[fnc_features] *= FNC_SCALE
df = pd.concat([site1, site2], axis=0)
X = df.iloc[:, :-1]
y = df.loc[:, 'Label']

# one hot
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

model = SVC(decision_function_shape='ovo', class_weight="balanced")

skf = StratifiedKFold(n_splits=4)
skf.get_n_splits(X, y)
features = loading_features + fnc_features
for train_ind, val_ind in tqdm_notebook(skf.split(X, y)):
    train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]
    model.fit(train_df[features], train_df["Label"])
    
    y_scores = model.predict(val_df[features])
    print(roc_auc_score(val_df["Label"], y_scores))

