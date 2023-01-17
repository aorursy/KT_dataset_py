import numpy as np # linear algebra

import pandas as pd
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import manifold

import os
df = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample = pd.read_csv("../input/sample_submission.csv")
total = df.isnull().sum().sort_values(ascending = False)
total[total>0]
df['rez_esc'] = df['rez_esc'].fillna(0)
test['rez_esc'] = test['rez_esc'].fillna(0)
df[df['v18q1'].isnull()]['v18q'].describe()
def replace_v18q1(x):
    if(x['v18q'] == 0):
        return 0
    else:
        return x['v18q1']
df['v18q1'] = df.apply(lambda x: replace_v18q1(x), axis=1)
test['v18q1'] = test.apply(lambda x: replace_v18q1(x), axis=1)
df[df['v2a1'].isnull()]['tipovivi3'].describe()
df['v2a1'] = df['v2a1'].fillna(0)
test['v2a1'] = test['v2a1'].fillna(0)
df['meaneduc'] = df['meaneduc'].fillna(0)
test['SQBmeaned'] = test['SQBmeaned'].fillna(0)
df['edjefe'] = df['edjefe'].replace({'no': 0, 'yes':1}).astype(float)
test['edjefe'] = test['edjefe'].replace({'no': 0, 'yes':1}).astype(float)
df['edjefa'] = df['edjefa'].replace({'no': 0, 'yes':1}).astype(float)
test['edjefa'] = test['edjefa'].replace({'no': 0, 'yes':1}).astype(float)
df['dependency']=np.sqrt(df['SQBdependency'])
test['dependency']=np.sqrt(test['SQBdependency'])
df['adult_num'] = df['hogar_adul'] - df['hogar_mayor']
df['adult_rate'] = df['adult_num'] / df['hogar_total']
df['dependency_num'] = df['hogar_nin'] + df['hogar_mayor']
df['dependency_rate'] = df['dependency_num'] / df['hogar_total']
df['adult_dependency_rate'] = df['adult_num'] / (df['dependency_num']+0.1)
df['children_rate'] = df['hogar_nin'] / df['hogar_total']
df['elder_rate'] = df['hogar_mayor'] / df['hogar_total']

df['rent_per_person'] = df['v2a1'] / df['hogar_total']
df['rent_per_adult'] = df['v2a1'] / (df['adult_num']+0.1)

df['head_is_adult'] = (df['adult_num'] > 0).astype(int)

df['bedroom_per_person'] = df['bedrooms'] / df['hogar_total']
df['bedroom_per_adult'] = df['bedrooms'] / (df['adult_num']+0.1)

df['rent_per_room'] = df['v2a1'] / df['rooms']
test['adult_num'] = test['hogar_adul'] - test['hogar_mayor']
test['adult_rate'] = test['adult_num'] / test['hogar_total']
test['dependency_num'] = test['hogar_nin'] + test['hogar_mayor']
test['dependency_rate'] = test['dependency_num'] / test['hogar_total']
test['adult_dependency_rate'] = test['adult_num'] / (test['dependency_num']+0.1)
test['children_rate'] = test['hogar_nin'] / test['hogar_total']
test['elder_rate'] = test['hogar_mayor'] / test['hogar_total']

test['rent_per_person'] = test['v2a1'] / test['hogar_total']
test['rent_per_adult'] = test['v2a1'] / (test['adult_num']+0.1)

test['head_is_adult'] = (test['adult_num'] > 0).astype(int)

test['bedroom_per_person'] = test['bedrooms'] / test['hogar_total']
test['bedroom_per_adult'] = test['bedrooms'] / (test['adult_num']+0.1)

test['rent_per_room'] = test['v2a1'] / test['rooms']
X = df.drop(['Id', 'idhogar', 'Target'], axis=1).fillna(0)
y = df.Target

test_X = test.drop(['Id', 'idhogar'], axis=1).fillna(0)
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

X_ss = pd.DataFrame(StandardScaler().fit(X).transform(X), columns=X.columns)

mds = manifold.MDS(2, max_iter=100, n_init=1);
trans_data = mds.fit_transform(X_ss)
y_ = df.Target -1
colors = ['r','g','y','c']

# for j, X_ in trans_data.items():
plt.figure(figsize=(6,4))
for i in [0,1,2,3]:
    plt.scatter(trans_data[y_==i,0], trans_data[y_==i,1], c=colors[i], s=5, label=i+1)
plt.legend()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
trans_data = tsne.fit_transform(X_ss)

# for j, X_ in trans_data.items():
plt.figure(figsize=(6,4))
for i in [0,1,2,3]:
    plt.scatter(trans_data[y_==i,0], trans_data[y_==i,1], c=colors[i], s=5, label=i+1)
plt.legend()
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
scores_knn_distance = pd.DataFrame(columns=['neighbours','score'])
for i_ in range(3,20):
    clf_knn = neighbors.KNeighborsClassifier(n_neighbors=i_, weights="distance")
    model_knn = clf_knn.fit(X_train,y_train)

    score = f1_score(y_test, model_knn.predict(X_test), average = 'macro')
    scores_knn_distance = scores_knn_distance.append({'neighbours':i_,'score':score},ignore_index=True)
    print('distance, neighbours=',i_,', f1_score=',score.mean())
scores_knn_uniform = pd.DataFrame(columns=['neighbours','score'])
for i_ in range(3,20):
    clf_knn = neighbors.KNeighborsClassifier(n_neighbors=i_, weights="uniform")
    model_knn = clf_knn.fit(X_train,y_train)

    score = f1_score(y_test, model_knn.predict(X_test), average = 'macro')
    scores_knn_uniform = scores_knn_uniform.append({'neighbours':i_,'score':score},ignore_index=True)
    print('uniform, neighbours=',i_,', f1_score=',score.mean())
scores_knn_distance.plot(kind='line',x='neighbours', y='score')
plt.ylabel('f1_score')
plt.xlabel('number of neighbours')
plt.title("the score of neighbours for distance")

scores_knn_uniform.plot(kind='line',x='neighbours', y='score')
plt.ylabel('f1_score')
plt.xlabel('number of neighbours')
plt.title("the score of neighbours for uniform")
from sklearn import svm
clf_svm = svm.SVC(gamma='auto')
model_svm = clf_svm.fit(X_train, y_train)
scores_svm = f1_score(y_test, model_svm.predict(X_test), average = 'macro')
print('svm, f1_score=',scores_svm.mean())
from sklearn.linear_model import SGDClassifier

clf_sgd = SGDClassifier(loss="hinge", penalty="l1", 
                    alpha=0.01, max_iter=200, fit_intercept=True)
model_sgd = clf_sgd.fit(X,y)
scores_sgd = f1_score(y_test, model_sgd.predict(X_test), average = 'macro')
print('sgd, f1_score=',scores_sgd.mean())
from sklearn.neural_network import MLPClassifier
clf_mlp = MLPClassifier(hidden_layer_sizes=(12,), random_state=1, max_iter=1000, warm_start=True)
model_mlp = clf_mlp.fit(X_train, y_train) 
scores_mlp = f1_score(y_test, model_mlp.predict(X_test), average = 'macro')
print('mlp f1_score=',scores_mlp.mean())
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
clf_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, min_samples_split=30, min_samples_leaf=30),
                         algorithm="SAMME.R",n_estimators=500, learning_rate=1)
model_ada = clf_ada.fit(X_train,y_train)
scores_ada = f1_score(y_test, model_ada.predict(X_test), average = 'macro')
print('adaboosting f1_score=',scores_ada.mean())
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

clf1 = clf1.fit(X_train,y_train)
clf2 = clf2.fit(X_train,y_train)
clf3 = clf3.fit(X_train,y_train)
model_voting = eclf.fit(X_train,y_train)
scores_voting = f1_score(y_test, model_voting.predict(X_test), average = 'macro')
print('voting, f1_score=',scores_voting.mean())
clf_gradient = GradientBoostingClassifier(n_estimators=800, learning_rate=0.5, 
                                 max_depth=3, random_state=8)
model_gradient = clf_gradient.fit(X_train,y_train)
scores_gradient = f1_score(y_test, model_gradient.predict(X_test), average = 'macro')
print('GradientBoosting f1_score=',scores_gradient.mean())
from xgboost import XGBClassifier
clf_xgb = XGBClassifier(max_depth=4,booster='dart',
                           min_child_weight=1,
                           learning_rate=0.01,
                           n_estimators=500,
                           silent=True,
                           objective='multi:softprob',
                           gamma=0,
                           max_delta_step=0,
                           subsample=0.9,
                           colsample_bytree=0.9,
                           colsample_bylevel=1,
                           reg_alpha=0,
                           reg_lambda=0,
                           scale_pos_weight=1,
                           seed=0,
                           missing=None)
model_xgb = clf_xgb.fit(X_train, y_train, eval_metric='mlogloss', verbose=100,
            eval_set=[(X_test, y_test)], early_stopping_rounds=20)
scores_xgb = f1_score(y_test, model_xgb.predict(X_test), average = 'macro')
print('xgb f1_score=',scores_xgb.mean())
clf_lgbm = lgb.LGBMClassifier(class_weight='balanced', boosting_type='dart',
                         drop_rate=0.9, min_data_in_leaf=100, 
                         max_bin=255,
                         n_estimators=500,
                         bagging_fraction=0.01,
                         min_sum_hessian_in_leaf=1,
                         importance_type='gain',
                         learning_rate=0.1, 
                         max_depth=-1, 
                         num_leaves=31)
model_lgbm = clf_lgbm.fit(X_train, y_train);
scores_lgbm = f1_score(y_test, model_lgbm.predict(X_test), average = 'macro')
print('lgbm f1_score=',scores_lgbm.mean())
scores_algorithm = pd.DataFrame(columns=['scores'])
scores_algorithm.loc['kNN_distance'] = scores_knn_distance[scores_knn_distance.neighbours==10].score.values
scores_algorithm.loc['kNN_uniform'] = scores_knn_distance[scores_knn_uniform.neighbours==4].score.values
scores_algorithm.loc['SVM'] = scores_svm
scores_algorithm.loc['SGD'] = scores_sgd
scores_algorithm.loc['Neural Network'] = scores_mlp
scores_algorithm.loc['AdaBoosting'] = scores_ada
scores_algorithm.loc['voting'] = scores_voting
scores_algorithm.loc['GradientBoosting'] = scores_gradient
scores_algorithm.loc['XGBoosting'] = scores_xgb
scores_algorithm.loc['LGBM'] = scores_lgbm

scores_algorithm.plot(kind='bar', use_index=True, y='scores')
plt.ylabel("f1_score")
plt.title("The f1_score of different algorithms")
# knn distance 10 neighbours 0.293
clf_knn_distance = neighbors.KNeighborsClassifier(n_neighbors=10, weights="distance")
model_knn_distance = clf_knn_distance.fit(X_train,y_train)

test['Target'] = model_knn_distance.predict(test_X)
    
result = test[['Id','Target']]
result.to_csv("result_kNN_10_distance.csv",index=False,sep=',')
# knn uniform 4 neighbours 0.310
clf_knn_uniform = neighbors.KNeighborsClassifier(n_neighbors=4, weights="uniform")
model_knn_uniform = clf_knn_uniform.fit(X_train,y_train)

test['Target'] = model_knn_uniform.predict(test_X)
    
result = test[['Id','Target']]
result.to_csv("result_kNN_4_uniform.csv",index=False,sep=',')
# svm 0.209
test['Target'] = model_svm.predict(test_X)
    
result = test[['Id','Target']]
result.to_csv("result_svm.csv",index=False,sep=',')
# sgd 0.310
test['Target'] = model_sgd.predict(test_X)
    
result = test[['Id','Target']]
result.to_csv("result_sgd.csv",index=False,sep=',')
# neural network 0.346
test['Target'] = model_mlp.predict(test_X)
    
result = test[['Id','Target']]
result.to_csv("result_neural_network.csv",index=False,sep=',')
# Adaboosting 0.376
test['Target'] = model_ada.predict(test_X)
    
result = test[['Id','Target']]
result.to_csv("result_adaboosting.csv",index=False,sep=',')
# voting 0.379
test['Target'] = model_voting.predict(test_X)
    
result = test[['Id','Target']]
result.to_csv("result_voting.csv",index=False,sep=',')
# gradient boosting 0.387
test['Target'] = model_gradient.predict(test_X)
    
result = test[['Id','Target']]
result.to_csv("result_gradientboosting.csv",index=False,sep=',')
# XGBoosting 0.342
test['Target'] = model_xgb.predict(test_X)
    
result = test[['Id','Target']]
result.to_csv("result_XGboosting.csv",index=False,sep=',')
# LGBM 0.416
test['Target'] = model_lgbm.predict(test_X)
    
result = test[['Id','Target']]
result.to_csv("result_LGBM.csv",index=False,sep=',')
score_result = pd.DataFrame(columns=['score'])

score_result.loc['kNN_distance'] = 0.293
score_result.loc['kNN_uniform'] = 0.310
score_result.loc['SVM'] = 0.209
score_result.loc['SGD'] = 0.310
score_result.loc['Neural Network'] = 0.346
score_result.loc['AdaBoosting'] = 0.376
score_result.loc['voting'] = 0.379
score_result.loc['GradientBoosting'] = 0.387
score_result.loc['XGBoosting'] = 0.342
score_result.loc['LGBM'] = 0.416

score_result.plot(kind='bar', y='score')
plt.ylabel("f1_score")
plt.title("The f1_score from the kaggle for different algorithms")
df_head = df[df.parentesco1==1]
X_head = df_head.drop(['Id', 'idhogar', 'Target'], axis=1).fillna(0)
y_head = df_head.Target
model_lgbm_final = clf_lgbm.fit(X_head,y_head)

test['Target'] = model_lgbm.predict(test_X)

result = test[['Id','Target']]
result.to_csv("result_LGBM_final.csv",index=False,sep=',')
df_corr_target = df.corr()['Target']
print(df_corr_target[df_corr_target>0.2])
print(df_corr_target[df_corr_target<-0.2])
model_lgbm_test = clf_lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
            early_stopping_rounds=20, verbose=100)

test['Target'] = model_lgbm_test.predict(test_X)

result = test[['Id','Target']]
result.to_csv("result_LGBM_test.csv",index=False,sep=',')
