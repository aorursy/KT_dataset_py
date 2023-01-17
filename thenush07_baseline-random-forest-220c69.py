import os
print((os.listdir('../input/')))
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
df_train = pd.read_csv('../input/web-club-recruitment-2018/train.csv')
df_test = pd.read_csv('../input/web-club-recruitment-2018/test.csv')

df_train.head()
train_X = df_train.loc[:, 'X1':'X23']
train_y = df_train.loc[:, 'Y']
rf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100,
                                   subsample=0.9, criterion='friedman_mse', min_samples_split=5,
                                   min_samples_leaf=49, min_weight_fraction_leaf=0.0, max_depth=3,
                                   min_impurity_decrease=0.6, min_impurity_split=None, init=None,
                                   random_state=70, max_features=0.4, verbose=0, 
                                   max_leaf_nodes=None, warm_start=True, presort='auto')
rf.fit(train_X, train_y)
df_test = df_test.loc[:, 'X1':'X23']
pred = rf.predict_proba(df_test)
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier

num_folds = 10
num_instances = len(train_X)
KFold = cross_validation.KFold(n=num_instances, n_folds=num_folds)
model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100,
                                   subsample=0.9, criterion='friedman_mse', min_samples_split=5,
                                   min_samples_leaf=49, min_weight_fraction_leaf=0.0, max_depth=3,
                                   min_impurity_decrease=0.6, min_impurity_split=None, init=None,
                                   random_state=70, max_features=0.4, verbose=0, 
                                   max_leaf_nodes=None, warm_start=True, presort='auto')
results = cross_validation.cross_val_score(model,train_X,train_y,cv=KFold)
print(results)
print(results.mean()*100)
result = pd.DataFrame(pred[:,1])
result.index.name = 'id'
result.columns = ['predicted_val']
result.to_csv('output.csv', index=True)
