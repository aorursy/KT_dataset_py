import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/glass-quality/Train.csv")
test = pd.read_csv("/kaggle/input/glass-quality/Test.csv")
print(train.shape)
print(test.shape)
train.head(3)
# train.info()
train.describe()
test.describe()
train.skew()
train.kurt()
train.columns
train['grade_A_Component_1'].value_counts()
# sns.countplot(data=train,x='grade_A_Component_1')
pd.DataFrame(train.groupby(['grade_A_Component_1','class'])['class'].count())
sns.countplot(data=train,x='grade_A_Component_1',hue='class')
train['grade_A_Component_2'].value_counts()   
pd.DataFrame(train.groupby(['grade_A_Component_2','class'])['class'].count())
sns.countplot(data=train,x='grade_A_Component_2',hue='class')
# train.drop("grade_A_Component_2",axis=1,inplace=True)
# test.drop("grade_A_Component_2",axis=1,inplace=True)
train['max_luminosity'].describe()
len(train['max_luminosity'].unique())
#Check for the distribution of max_luminosity
train['max_luminosity'].plot(kind='hist',figsize=(13,8),bins=100,edgecolor='k',
                              title='max_luminosity Distribution').autoscale(axis='x',tight=True)
train['thickness'].describe()
len(train['thickness'].unique())
#Check for the distribution of loan amount
train['thickness'].plot(kind='hist',figsize=(10,6),bins=100,edgecolor='k',
                              title='thickness Distribution').autoscale(axis='x',tight=True)
train['xmin'].describe()
# train['xmin'].value_counts()
len(train['xmin'].unique())
#Check for the distribution of loan amount
train['xmin'].plot(kind='hist',figsize=(10,6),bins=100,edgecolor='k',
                              title='xmin Distribution').autoscale(axis='x',tight=True)
train['xmax'].describe()
len(train['xmax'].unique())
#Check for the distribution of loan amount
train['xmax'].plot(kind='hist',figsize=(10,6),bins=100,edgecolor='k',
                              title='xmax Distribution').autoscale(axis='x',tight=True)
train['ymin'].describe()
len(train['ymin'].unique())
#Check for the distribution of loan amount
train['ymin'].plot(kind='hist',figsize=(10,6),bins=100,edgecolor='k',
                              title='ymin Distribution').autoscale(axis='x',tight=True)
train['ymax'].describe()
len(train['ymax'].unique())
#Check for the distribution of loan amount
train['ymax'].plot(kind='hist',figsize=(10,6),bins=100,edgecolor='k',
                              title='ymax Distribution').autoscale(axis='x',tight=True)
train['pixel_area'].describe()
train['pixel_area'].head(5)
#Check for the distribution of pixel_area 
train['pixel_area'].plot(kind='hist',figsize=(10,6),bins=100,edgecolor='k',
                              title='pixel_area Distribution').autoscale(axis='x',tight=True)
train['pixel_area'].max()
train['log_area'].describe()
#Check for the distribution of log_area 
train['log_area'].plot(kind='hist',figsize=(10,6),bins=100,edgecolor='k',
                              title='log_area Distribution').autoscale(axis='x',tight=True)
plt.figure(figsize=(12,5))
sns.scatterplot(data=train,x='pixel_area',y='log_area');
train['x_component_1'].value_counts()
sns.countplot(data=train,x='x_component_1',hue='class')
pd.DataFrame(train.groupby(['x_component_1','class'])['class'].count())
sns.countplot(data=train,x='x_component_1')
train['x_component_2'].value_counts()
sns.countplot(data=train,x='x_component_2',hue='class')
pd.DataFrame(train.groupby(['x_component_2','class'])['class'].count())
sns.countplot(data=train,x='x_component_2')
train['x_component_3'].value_counts()
sns.countplot(data=train,x='x_component_3',hue='class')
pd.DataFrame(train.groupby(['x_component_3','class'])['class'].count())
sns.countplot(data=train,x='x_component_3')
train['x_component_4'].value_counts()
sns.countplot(data=train,x='x_component_4',hue='class')
pd.DataFrame(train.groupby(['x_component_4','class'])['class'].count())
sns.countplot(data=train,x='x_component_4')
train['x_component_5'].value_counts()
sns.countplot(data=train,x='x_component_5',hue='class')
pd.DataFrame(train.groupby(['x_component_5','class'])['class'].count())
sns.countplot(data=train,x='x_component_5')
train['class'].value_counts()
print("the ratio of the two classes is {0:0.3f}".format(887/471))
#check for duplicated data
train.duplicated().sum()
# -------------
# Already Done by dropping
# train.drop("ymin",axis=1,inplace=True)
# test.drop("ymin",axis=1,inplace=True)
# train.drop("ymax",axis=1,inplace=True)
# test.drop("ymax",axis=1,inplace=True)
# train.drop("pixel_area",axis=1,inplace=True)
# test.drop("pixel_area",axis=1,inplace=True)
one = train[train['x_component_1']==1]
one = np.array(one.index)
# one

two = train[train['x_component_2']==1]
two = np.array(two.index)
# two

three = train[train['x_component_3']==1]
three = np.array(three.index)
# three

four = train[train['x_component_4']==1]
four = np.array(four.index)
# four

five = train[train['x_component_5']==1]
five = np.array(five.index)
# five


known_index_train = []

for i in range(len(one)):
    known_index_train.append(one[i])
    
for i in range(len(two)):
    known_index_train.append(two[i])
    
for i in range(len(three)):
    known_index_train.append(three[i])
    
for i in range(len(four)):
    known_index_train.append(four[i])
    
for i in range(len(five)):
    known_index_train.append(five[i])
    
    
print(len(known_index_train))
known_train_set = set(known_index_train)
print(len(known_train_set))      # f**k almost 609 out of 1358 are supposed to belong class "1"


known_index_train = np.array(known_index_train)
known_index_train.sort()
# known_index_train
one = test[test['x_component_1']==1]
one = np.array(one.index)
# one

two = test[test['x_component_2']==1]
two = np.array(two.index)
# two

three = test[test['x_component_3']==1]
three = np.array(three.index)
# three

four = test[test['x_component_4']==1]
four = np.array(four.index)
# four

five = test[test['x_component_5']==1]
five = np.array(five.index)
# five

known_index_test = []

for i in range(len(one)):
    known_index_test.append(one[i])
    
for i in range(len(two)):
    known_index_test.append(two[i])
    
for i in range(len(three)):
    known_index_test.append(three[i])
    
for i in range(len(four)):
    known_index_test.append(four[i])
    
for i in range(len(five)):
    known_index_test.append(five[i])
    
    
print(len(known_index_test))
known_set_test = set(known_index_test)
known_index_test = np.array(known_index_test)
known_index_test.sort()
print(len(known_set_test))  

print("Thus out of 583 we know the classes of {0} values i.e {1:0.03f} percentage from the testing dataset".format(257,(257/583)*100))
train.head(1)
known_train_series = []
for i in range(1358):
    known_train_series.append(3)

    
known_test_series = []
for i in range(583):
    known_test_series.append(3)
    
    
for i in range(1358):
    if i in known_index_train:         # if that element exist in the indexes we know, then
        known_train_series[i] = 1                           #we will assign it as ONE
    else:
        known_train_series[i]= 2                            #and rest as TWO        
        
for i in range(583):
    if i in known_index_test:
        known_test_series[i] = 1
    else:
        known_test_series[i]= 2
        
known_train_series = np.array(known_train_series)
known_test_series = np.array(known_test_series)


train['known'] =known_train_series
test['known'] =known_test_series

print(train['known'].value_counts())                   #i.e we are 100% sure about 609 values


print(test['known'].value_counts()  )                 #i.e we are 100% sure about 257 values
train['class'].value_counts()
X = train.drop(['class'],axis=1)
y = train['class']

from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2,sampling_strategy='all') 
X, y = sm.fit_sample(X, y) 
y.value_counts()
train = pd.merge(X,y,left_index=True,right_index=True)
train.head(2)
cor = train.corr()
plt.figure(figsize=(25,13))
sns.heatmap(cor,annot=True,cmap='plasma',linecolor='black')
plt.show()
train.head(2)
# #Log transfromations
# train['max_luminosity'] = np.log1p(train['max_luminosity'])
# train['log_thickness'] = np.log1p(train['thickness'])
# train['ymin'] = np.log1p(train['ymin'])
# train['ymax'] = np.log1p(train['ymax'])
# train['log_area'] = np.log1p(train['log_area'])

# #Log transfromations for Test

# test['max_luminosity'] = np.log1p(test['max_luminosity'])
# test['log_thickness'] = np.log1p(test['thickness'])
# test['ymin'] = np.log1p(train['ymin'])
# test['ymax'] = np.log1p(train['ymax'])
# test['log_area'] = np.log1p(train['log_area'])
#Log transfromations
train['log_max_luminosity'] = np.log1p(train['max_luminosity'])
train['log_thickness'] = np.log1p(train['thickness'])
train['log_ymin'] = np.log1p(train['ymin'])
train['log_ymax'] = np.log1p(train['ymax'])
train['log_log_area'] = np.log1p(train['log_area'])

#Log transfromations for Test

test['log_max_luminosity'] = np.log1p(test['max_luminosity'])
test['log_thickness'] = np.log1p(test['thickness'])
test['log_ymin'] = np.log1p(train['ymin'])
test['log_ymax'] = np.log1p(train['ymax'])
test['log_log_area'] = np.log1p(train['log_area'])
train['ratio_of_xmin_ymin'] = train['xmin'] / train['log_ymin']
train['ratio_of_xmin_ymin'] = np.log1p(train['ratio_of_xmin_ymin'])
train['ratio_of_xmax_ymax'] = train['xmax'] / train['log_ymax']
train['ratio_of_xmax_ymax'] = np.log1p(train['ratio_of_xmax_ymax'])
test['ratio_of_xmin_ymin'] = test['xmin'] / test['log_ymin']
test['ratio_of_xmin_ymin'] = np.log1p(test['ratio_of_xmin_ymin'])
test['ratio_of_xmax_ymax'] = test['xmax'] / train['log_ymax']
test['ratio_of_xmax_ymax'] = np.log1p(test['ratio_of_xmax_ymax'])
# # #Check for the distribution of loan amount
# test['ratio_of_xmin_ymin'].plot(kind='hist',figsize=(10,6),bins=100,edgecolor='k',
#                               title='ratio_of_xmax_ymax Distribution').autoscale(axis='x',tight=True)
train.skew()
train.kurt()
temp_test = test.copy()
temp_train = train.copy()
# test = temp_test
# train = temp_train
train['class'].value_counts()
def metric(y,y0):
    return log_loss(y,y0)
features = list(set(train.columns)-set(['class']))
target = 'class'
len(features)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.metrics import log_loss
def cross_valid(model,train,features,target,cv=3):
    results = cross_val_predict(model, train[features], train[target], method="predict_proba",cv=cv)
    return metric(train[target],results)
models = [lgb.LGBMClassifier(), xgb.XGBClassifier(), GradientBoostingClassifier(), 
#           LogisticRegression(max_iter=110), 
              RandomForestClassifier(),ExtraTreesClassifier(), 
#           CatBoostClassifier(),
             ]

for i in models:
    model = i
    error = cross_valid(model,train,features,target,cv=10)
    print(str(model).split("(")[0], error)
# error = cross_valid(CatBoostClassifier(),train,features,target,cv=10)
# print(error)
# cat_features = ['grade_A_Component_1','grade_A_Component_2', 'x_component_1', 'x_component_2',
#                 'x_component_3', 'x_component_4', 'x_component_5','known']
# num_features = ['max_luminosity', 'thickness', 'xmin', 'xmax', 'ymin', 'ymax', 'pixel_area', 'log_area']
# # train_c = train.copy()
# from sklearn.preprocessing import StandardScaler
# std = StandardScaler()
# train_scaled_num = std.fit_transform(train[num_features])
# test_scaled_num = std.transform(test[num_features])


# train[num_features] = pd.DataFrame(train_scaled_num, columns=num_features)
# test[num_features] = pd.DataFrame(test_scaled_num, columns=num_features)
X = train.drop(['class'],axis=1)
y = train['class']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)
y_test.value_counts()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
scaled_test = scaler.transform(test)    # to be run only once
from sklearn.decomposition import PCA
temp_pca = PCA(n_components=None)
X_temp =temp_pca.fit_transform(X_train)
variance = temp_pca.explained_variance_ratio_
print(variance)
plt.figure(figsize=(8,6))
plt.plot(np.cumsum(temp_pca.explained_variance_ratio_))
plt.xlim(0,21,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()
# ploting cum
# pca = PCA(n_components=13)
# X_train =pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# test = pca.transform(test)
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import log_loss
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
# def model(params):
#     clf = XGBClassifier(**params)
#     return cross_val_score(clf, X, y).mean()


# space = {
#     'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.2, 0.05)),
#     'max_depth':        hp.choice('max_depth',        np.arange(5, 20, 1, dtype=int)),
# #     'num_leaves': hp.choice('num_leaves', np.arange(16, 40, 2, dtype=int)),
# #     'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
#     'n_estimators':     hp.choice('n_estimators', np.arange(100,1000,10, dtype=int)),
#     'random_state': 51,
#     'boosting_type': 'gbdt'
# }


# def objective(params):
#     acc = model(params)
#     return {'loss': -acc, 'status': STATUS_OK}

# trials = Trials()
# best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)

# print(best)
# from xgboost import XGBClassifier

# clf= XGBClassifier(
#                    learning_rate=0.06, 
#                    n_estimators=400,
#                    max_depth=14,
# #                     objective= 'binary:logistic',
#                   )

# clf.fit(X_train, y_train)

# y_pred = clf.predict_proba(X_test)
# print('XGboost log_loss {}'. format(log_loss(y_test, y_pred)))
# clf.score(X_train, y_train)
# n_estimators=100,   ##
#     criterion='gini',  ## entropy
#     max_depth=None,
#     min_samples_split=2, ##The minimum number of samples required to split an internal node:
#     min_samples_leaf=1,  ##The minimum number of samples required to be at a leaf node.
#     min_weight_fraction_leaf=0.0,  #float
#     max_features='auto',  #sqrt , log2
#     max_leaf_nodes=None,  # If None then unlimited number of leaf nodes.
#     min_impurity_decrease=0.0,
#     min_impurity_split=None,
#     bootstrap=False,
#     oob_score=False,
#     n_jobs=None,
#     random_state=21,
#     verbose=0,
#     warm_start=False,  #bool
#     class_weight=None,
#     ccp_alpha=0.0,
#     max_samples=None,
def model(params):
    clf = ExtraTreesClassifier(**params)
    return cross_val_score(clf, X, y).mean()


space = {
    'n_estimators': hp.choice('n_estimators', np.arange(100, 300, 10, dtype=int)),
#     'max_depth':    hp.choice('max_depth', np.arange(5, 25, 1, dtype=int)),
    'min_samples_split' : hp.choice('min_samples_split', np.arange(2, 10, 1, dtype=int)),
    'min_samples_leaf' : hp.choice('min_samples_leaf', np.arange(1, 10, 1, dtype=int)),
    'max_features':'auto', #sqrt,log2
    'random_state':21,
    'warm_start':False,
}


def objective(params):
    acc = model(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)

print(best)


clf= ExtraTreesClassifier(
                    n_estimators=105,
#                     max_depth= 19,
                    min_samples_split= 2,
#                     min_samples_leaf= 1,
                    max_features=  'sqrt',
                    random_state= 21   ,
                    warm_start=  False
                    )

clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)
print('ExtraTreesClassifier log_loss {}'. format(log_loss(y_test, y_pred)))
clf.score(X_train, y_train)
# ExtC = ExtraTreesClassifier()
# from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
# kfold = StratifiedKFold(n_splits=10)

# ## Search grid for optimal parameters
# ex_param_grid = {"max_depth": [None],
# "min_samples_split": [2, 3, 10],
# "min_samples_leaf": [1, 3, 10],
# "bootstrap": [False],
# "n_estimators" :[100,300],
# "criterion": ["gini"]}


# gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

# gsExtC.fit(X_train,y_train)

# ExtC_best = gsExtC.best_estimator_

# # Best score
# gsExtC.best_score_
# ExtC_best
# y_pred = gsExtC.predict_proba(X_test)
# print('ExtraTreesClassifier log_loss {}'. format(log_loss(y_test, y_pred)))
# clf.score(X_train, y_train)
# def model(params):
#     clf = CatBoostClassifier(**params)
#     return cross_val_score(clf, X, y).mean()


# space = {
#     'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.2, 0.05)),
#     'max_depth':        hp.choice('max_depth',        np.arange(5, 20, 1, dtype=int)),
# #     'num_leaves': hp.choice('num_leaves', np.arange(16, 40, 2, dtype=int)),
# # #     'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
# # #     'num_leaves':        hp.choice('num_leaves', 200, 400,5, dtype=int),
#     'n_estimators':     hp.choice('n_estimators', np.arange(200,100,10, dtype=int)),
# #     'random_state': 51,
# #     'boosting_type': 'gbdt'
# }


# def objective(params):
#     acc = model(params)
#     return {'loss': -acc, 'status': STATUS_OK}

# trials = Trials()
# best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)


# print(best)
# from catboost import CatBoostClassifier

# clf= CatBoostClassifier(
#                          depth=6,
#                          random_seed=42, 
#                          iterations=1000, 
#                          learning_rate=0.07,
#                          leaf_estimation_iterations=1,
#                          l2_leaf_reg=1, 
#                          bootstrap_type='Bayesian', 
#                          bagging_temperature=1, 
#                          random_strength=1,
#                          od_type='Iter', 
#                          od_wait=200,
#                         )

# clf.fit(X_train, y_train)

# y_pred = clf.predict_proba(X_test)
# print('CatBoostClassifier log_loss {}'. format(log_loss(y_test, y_pred)))
# clf.score(X_train, y_train)
from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator=clf,X=X,y=y,cv=10)
print(acc)
print(acc.mean())
print(acc.std())
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, labels=[1, 2]))
#               precision    recall  f1-score   support

#            1       0.93      0.88      0.90        90
#            2       0.88      0.93      0.91        88

#     accuracy                           0.90       178
#    macro avg       0.91      0.90      0.90       178
# weighted avg       0.91      0.90      0.90       178
y_pred = clf.predict_proba(scaled_test)
y_pred[2]  # belongs to class 1
print(y_pred[2][0])
print(y_pred[2][1])
count =0
for i in range(583):
    if y_pred[i][1]>0.5:   # counting class 2 predictions
        count +=1
        
print("Class 2 ",count)
print("Class 1 ",583-count)
len(known_index_test)
# known_index_test
for i in range(583):
    if i in known_index_test:
        y_pred[i][0]=1
        y_pred[i][1]=0
count = 0
for i in range(583):
    if y_pred[i][1] > 0.95:
        count += 1

print(count)
# for i in range(583):
#     if y_pred[i][0] >0.95:
#         y_pred[i][0]=1
#         y_pred[i][1]=0
#     elif y_pred[i][1] >0.95:
#         y_pred[i][0]=0
#         y_pred[i][1]=1
result_data_df = pd.DataFrame(y_pred,columns=['1','2'])
result_data_df.to_excel('submission.xlsx',index=False)
result_data_df.head(10)