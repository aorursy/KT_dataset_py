import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
data=pd.read_csv('precleaned-datasets_2/dataset_1.csv')
data.head()
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['target'], axis=1),
    data['target'],
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape
X_train_original = X_train.copy()
X_test_original = X_test.copy()
X_train.shape, X_test.shape
constant_feat=[feat for feat in X_train.columns if X_train[feat].std()==0]
X_train.drop(labels=constant_feat,inplace=True,axis=1)
X_test.drop(labels=constant_feat,inplace=True,axis=1)
X_train.shape, X_test.shape
quasi_constant_feat=[]
for feature in X_train.columns:
    predominant=(X_train[feature].value_counts()/np.float(len(X_train))).sort_values(ascending=False).values[0]
    if predominant>0.998:
        quasi_constant_feat.append(feature)
len(quasi_constant_feat)
sel = VarianceThreshold(
    threshold=0.01)  # 0.1 indicates 99% of observations approximately

sel.fit(X_train)  # fit finds the features with low variance

sum(sel.get_support())

features_to_keep=X_train.columns[sel.get_support()]
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

X_train.shape, X_test.shape
X_train= pd.DataFrame(X_train)
X_train.columns = features_to_keep

X_test= pd.DataFrame(X_test)
X_test.columns = features_to_keep

def correlation(dataset, threshold):
    col_corr = set()
    
    corr_matrix = dataset.corr()
    
    for i in range(len(corr_matrix.columns)):
    
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: 
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.8)
print('correlated features: ', len(set(corr_features)) )
X_train.drop(labels=corr_features, axis=1, inplace=True)
X_test.drop(labels=corr_features, axis=1, inplace=True)

X_train.shape, X_test.shape
rf=RandomForestClassifier(n_estimators=200,random_state=39,max_depth=4)
rf.fit(X_train,y_train)
print('Train set')
pred = rf.predict_proba(X_train)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = rf.predict_proba(X_test)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
