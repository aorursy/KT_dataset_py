import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

 

%matplotlib inline

 

import warnings

warnings.filterwarnings('ignore')
#from google.colab import drive

#drive.mount('/content/drive')
#train = pd.read_csv('/content/drive/My Drive/DSN AI Bootcamp Qualification Hackathon/Train.csv')

#test  = pd.read_csv('/content/drive/My Drive/DSN AI Bootcamp Qualification Hackathon/Test.csv')

 

train = pd.read_csv('../input/dsn-ai-bootcamp-data/Train.csv')

test = pd.read_csv('../input/dsn-ai-bootcamp-data/Test.csv')

sub = pd.read_csv('../input/dsn-ai-bootcamp-data/SampleSubmission.csv')
train.head()
test.head()
sub.head()
train.info()
sub.info()
train.describe()
train.duplicated().sum()
def missing(df):

    missing_values = pd.DataFrame({'Number of missing values': df.isnull().sum(),

                                  'Percentage of missing values': df.isnull().sum()/len(df) * 100

                                  })

    return missing_values
missing(train)
num_cols = train.select_dtypes(exclude='object').columns

len(num_cols)
train.form_field47.value_counts()
train.default_status.value_counts()
sns.countplot(train.default_status)
sns.countplot(train.form_field47)
def boxplot(df):

    for col in num_cols:

        plt.title('Boxplot of '+ col)

        sns.boxplot(df[col])

        plt.show()

        
boxplot(train)
def distribution(df):

    for col in num_cols:

        plt.title('Distribution of '+ col)

        df[col].hist()

        plt.show()
distribution(train)
def fill_na(df1, df2):

    for col in num_cols:

        df1.fillna(-999, inplace=True)

        df2.fillna(-999, inplace=True)
fill_na(train, test)


print(missing(test))

missing(train)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
def encoder(df1, df2, col):

    df1[col] = le.fit_transform(df1[col])

    df2[col] = le.fit_transform(df2[col])

    

    
encoder(train, test, 'form_field47')

#encoder(train, _, 'default_status')
train.default_status = le.fit_transform(train.default_status)
train.info()
features = train.select_dtypes(exclude='object').drop(columns=['default_status']).columns

len(features), train.shape
features1 = train.select_dtypes(exclude='object').drop(columns=['default_status', 'form_field48', 'form_field49']).columns

test.info()
#pip install catboost
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
#from sklearn.cluster import KMeans
#kmeans = KMeans(random_state = 42, n_clusters=2)
#train_ = train[features]

#kmeans.fit(train_)

#y_pred = kmeans.predict(train_)
#train['cluster'] =y_pred

#train.cluster.value_counts()
#clusters = pd.DataFrame()

#clusters['cluster_range'] = range(1, 10)

#inertia = []
#for k in clusters['cluster_range']:

#    kmeans = KMeans(n_clusters=k, random_state=8).fit(train_)

#    inertia.append(kmeans.inertia_)
#clusters['inertia'] = inertia

#clusters
plt.figure(figsize=(20,18))

sns.heatmap(train[features].corr(), annot = True)
#plt.plot(clusters.cluster_range,  clusters.inertia)

#plt.show()
X = train[features]

y = train['default_status']
from sklearn.model_selection import StratifiedKFold
def eval_metric(y, pred):

    return roc_auc_score(y, pred, labels=[0, 1])
from sklearn.model_selection import train_test_split, GridSearchCV
#cat =CatBoostClassifier( task_type='GPU',random_seed = 42, early_stopping_rounds=200, n_estimators=4000)
#params = {'max_depth':[7,8],

    

 #   'learning_rate': [0.01,0.008],}

    
#gscv = GridSearchCV(cat, params, cv=5)
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, stratify=y, random_state=200)
#model = gscv.fit(X_train, y_train)
#y_pred = gscv.predict_proba(X_test)[:,1]



#eval_metric(y_test, y_pred)
#pred = gscv.predict_proba(test[num_cols])[:,1]

#sub.default_status = pred
#sub.to_csv('submit20.csv', index=False)
#gscv.best_params_
# Specify number of folds

folds = 10

skf = StratifiedKFold(folds)

 

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier
#XGBClassifier?
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
#cat = CatBoostClassifier(**params, task_type='GPU',random_seed = i)

#cat1 = CatBoostClassifier()

#model = StackingClassifier(estimators=[('cat',CatBoostClassifier(**params, task_type='GPU',random_seed = i)), ('cat1',cat1)])
cat_params = {'max_depth':8,

    'n_estimators': 4000,

    'learning_rate': 0.01,

    'objective': 'CrossEntropy',

      'eval_metric':'AUC',

    

    'early_stopping_rounds': 200,

   #'use_best_model': True,

}
xgb_params = {'learning_rate' : 0.01,

                          'max_depth' : 8, 

                          'n_estimators' : 4000,

                          

                          'tree_method' :'gpu_hist', 'gpu_id': 0,

                          'verbosity' : 0, 'booster' : 'gbtree'}
score_list = []

score = 0

test_oofs = []

 

for i, (train_index, val_index) in enumerate(skf.split(X, y)):

    

    X_train, y_train = X.loc[train_index, features], y.loc[train_index]

    X_val, y_val = X.loc[val_index, features], y.loc[val_index]

 

    #model = CatBoostClassifier(**params, task_type='GPU',random_seed = i, verbose=0)

   # model = AdaBoostClassifier(base_estimator=CatBoostClassifier( task_type='GPU',random_seed = i))

    #model = StackingClassifier(estimators=[('cat',CatBoostClassifier(**params, task_type='GPU',random_seed = i)), ('cat1',CatBoostClassifier(task_type='GPU',random_seed = i))])    

    model = XGBClassifier(**xgb_params, 

                           random_state=i

                          )

    #model = StackingClassifier([('cat', cat), ('xgb', xgb), ('lgb', lgb)])                          

    #model = XGBClassifier()

    model.fit(X_train, y_train)

    #model.fit(xtrain, ytrain)

    

    

    p = model.predict_proba(X_val)[:, 1]

    sc = eval_metric(y_val, p)

    score_list.append(sc)

    score += sc/folds

    

    pred = model.predict_proba(test[features])[:, 1]

    test_oofs.append(pred)

 

    print('Fold {} : {}'.format(i, sc))

 

print()

print()

print('Avg log : ', score)
oof_prediction = pd.DataFrame(test_oofs).T

oof_prediction.columns = ['fold_'+ str(i) for i in range(1, folds + 1)]

sub['default_status'] = np.mean(test_oofs, axis = 0)

sub.to_csv('model.csv', index = False)

feature_importance_df = pd.DataFrame(model.feature_importances_, columns=['importance'])

feature_importance_df['feature'] = features



plt.figure(figsize=(12, 10));

sns.barplot(x="importance", y="feature", data=feature_importance_df.sort_values(by = ['importance'], ascending = False).head(50))

plt.title('XGBClassifier features importance:');
score_list = []

score = 0

test_oofs = []

 

for i, (train_index, val_index) in enumerate(skf.split(X, y)):

    

    X_train, y_train = X.loc[train_index, features], y.loc[train_index]

    X_val, y_val = X.loc[val_index, features], y.loc[val_index]

 

    #model = CatBoostClassifier(**params, task_type='GPU',random_seed = i, verbose=0)

   # model = AdaBoostClassifier(base_estimator=CatBoostClassifier( task_type='GPU',random_seed = i))

    #model = StackingClassifier(estimators=[('cat',CatBoostClassifier(**params, task_type='GPU',random_seed = i)), ('cat1',CatBoostClassifier(task_type='GPU',random_seed = i))])    

    model1 = XGBClassifier( tree_method ='gpu_hist', gpu_id= 0,

                          verbosity =0, booster ='gbtree',

                           random_state=i

                          )

    #model = StackingClassifier([('cat', cat), ('xgb', xgb), ('lgb', lgb)])                          

    #model = XGBClassifier()

    model1.fit(X_train, y_train)

    #model.fit(xtrain, ytrain)

    

    

    p = model1.predict_proba(X_val)[:, 1]

    sc = eval_metric(y_val, p)

    score_list.append(sc)

    score += sc/folds

    

    pred = model1.predict_proba(test[features])[:, 1]

    test_oofs.append(pred)

 

    print('Fold {} : {}'.format(i, sc))

 

print()

print()

print('Avg log : ', score)
oof_prediction = pd.DataFrame(test_oofs).T

oof_prediction.columns = ['fold_'+ str(i) for i in range(1, folds + 1)]

sub['default_status'] = np.mean(test_oofs, axis = 0)

sub.to_csv('model1.csv', index = False)

feature_importance_df = pd.DataFrame(model1.feature_importances_, columns=['importance'])

feature_importance_df['feature'] = features



plt.figure(figsize=(12, 10));

sns.barplot(x="importance", y="feature", data=feature_importance_df.sort_values(by = ['importance'], ascending = False).head(50))

plt.title('XGBClassifier features importance:');
score_list = []

score = 0

test_oofs = []

 

for i, (train_index, val_index) in enumerate(skf.split(X, y)):

    

    X_train, y_train = X.loc[train_index, features1], y.loc[train_index]

    X_val, y_val = X.loc[val_index, features1], y.loc[val_index]

 

    #model = CatBoostClassifier(**params, task_type='GPU',random_seed = i, verbose=0)

   # model = AdaBoostClassifier(base_estimator=CatBoostClassifier( task_type='GPU',random_seed = i))

    #model = StackingClassifier(estimators=[('cat',CatBoostClassifier(**params, task_type='GPU',random_seed = i)), ('cat1',CatBoostClassifier(task_type='GPU',random_seed = i))])    

    model2 = XGBClassifier( tree_method ='gpu_hist', gpu_id= 0,

                          verbosity =0, booster ='gbtree',

                           random_state=i

                          )

    #model = StackingClassifier([('cat', cat), ('xgb', xgb), ('lgb', lgb)])                          

    #model = XGBClassifier()

    model2.fit(X_train, y_train)

    #model.fit(xtrain, ytrain)

    

    

    p = model2.predict_proba(X_val)[:, 1]

    sc = eval_metric(y_val, p)

    score_list.append(sc)

    score += sc/folds

    

    pred = model2.predict_proba(test[features1])[:, 1]

    test_oofs.append(pred)

 

    print('Fold {} : {}'.format(i, sc))

 

print()

print()

print('Avg log : ', score)
oof_prediction = pd.DataFrame(test_oofs).T

oof_prediction.columns = ['fold_'+ str(i) for i in range(1, folds + 1)]

sub['default_status'] = np.mean(test_oofs, axis = 0)

sub.to_csv('model2.csv', index = False)

feature_importance_df = pd.DataFrame(model2.feature_importances_, columns=['importance'])

feature_importance_df['feature'] = features1



plt.figure(figsize=(12, 10));

sns.barplot(x="importance", y="feature", data=feature_importance_df.sort_values(by = ['importance'], ascending = False).head(50))

plt.title('XGBClassifier features importance:');
score_list = []

score = 0

test_oofs = []

 

for i, (train_index, val_index) in enumerate(skf.split(X, y)):

    

    X_train, y_train = X.loc[train_index, features1], y.loc[train_index]

    X_val, y_val = X.loc[val_index, features1], y.loc[val_index]

 

    #model = CatBoostClassifier(**params, task_type='GPU',random_seed = i, verbose=0)

   # model = AdaBoostClassifier(base_estimator=CatBoostClassifier( task_type='GPU',random_seed = i))

    #model = StackingClassifier(estimators=[('cat',CatBoostClassifier(**params, task_type='GPU',random_seed = i)), ('cat1',CatBoostClassifier(task_type='GPU',random_seed = i))])    

    model3 = XGBClassifier(**xgb_params, 

                           random_state=i

                          )

    #model = StackingClassifier([('cat', cat), ('xgb', xgb), ('lgb', lgb)])                          

    #model = XGBClassifier()

    model3.fit(X_train, y_train)

    #model.fit(xtrain, ytrain)

    

    

    p = model3.predict_proba(X_val)[:, 1]

    sc = eval_metric(y_val, p)

    score_list.append(sc)

    score += sc/folds

    

    pred = model3.predict_proba(test[features1])[:, 1]

    test_oofs.append(pred)

 

    print('Fold {} : {}'.format(i, sc))

 

print()

print()

print('Avg log : ', score)
oof_prediction = pd.DataFrame(test_oofs).T

oof_prediction.columns = ['fold_'+ str(i) for i in range(1, folds + 1)]

sub['default_status'] = np.mean(test_oofs, axis = 0)

sub.to_csv('model3.csv', index = False)

feature_importance_df = pd.DataFrame(model3.feature_importances_, columns=['importance'])

feature_importance_df['feature'] = features1



plt.figure(figsize=(12, 10));

sns.barplot(x="importance", y="feature", data=feature_importance_df.sort_values(by = ['importance'], ascending = False).head(50))

plt.title('XGBClassifier features importance:');
score_list = []

score = 0

test_oofs = []

 

for i, (train_index, val_index) in enumerate(skf.split(X, y)):

    

    X_train, y_train = X.loc[train_index, features1], y.loc[train_index]

    X_val, y_val = X.loc[val_index, features1], y.loc[val_index]

 

    #model = CatBoostClassifier(**params, task_type='GPU',random_seed = i, verbose=0)

   # model = AdaBoostClassifier(base_estimator=CatBoostClassifier( task_type='GPU',random_seed = i))

    #model = StackingClassifier(estimators=[('cat',CatBoostClassifier(**params, task_type='GPU',random_seed = i)), ('cat1',CatBoostClassifier(task_type='GPU',random_seed = i))])    

    model4 = LGBMClassifier( n_estimator = 3000, learning_rate = 0.01,

                           random_state=i

                          )

    #model = StackingClassifier([('cat', cat), ('xgb', xgb), ('lgb', lgb)])                          

    #model = XGBClassifier()

    model4.fit(X_train, y_train)

    #model.fit(xtrain, ytrain)

    

    

    p = model4.predict_proba(X_val)[:, 1]

    sc = eval_metric(y_val, p)

    score_list.append(sc)

    score += sc/folds

    

    pred = model4.predict_proba(test[features1])[:, 1]

    test_oofs.append(pred)

 

    print('Fold {} : {}'.format(i, sc))

 

print()

print()

print('Avg log : ', score)
oof_prediction = pd.DataFrame(test_oofs).T

oof_prediction.columns = ['fold_'+ str(i) for i in range(1, folds + 1)]

sub['default_status'] = np.mean(test_oofs, axis = 0)

sub.to_csv('model4.csv', index = False)

feature_importance_df = pd.DataFrame(model4.feature_importances_, columns=['importance'])

feature_importance_df['feature'] = features1



plt.figure(figsize=(12, 10));

sns.barplot(x="importance", y="feature", data=feature_importance_df.sort_values(by = ['importance'], ascending = False).head(50))

plt.title('LGBClassifier features importance:');
score_list = []

score = 0

test_oofs = []

 

for i, (train_index, val_index) in enumerate(skf.split(X, y)):

    

    X_train, y_train = X.loc[train_index, features1], y.loc[train_index]

    X_val, y_val = X.loc[val_index, features1], y.loc[val_index]

 

    #model = CatBoostClassifier(**params, task_type='GPU',random_seed = i, verbose=0)

   # model = AdaBoostClassifier(base_estimator=CatBoostClassifier( task_type='GPU',random_seed = i))

    #model = StackingClassifier(estimators=[('cat',CatBoostClassifier(**params, task_type='GPU',random_seed = i)), ('cat1',CatBoostClassifier(task_type='GPU',random_seed = i))])    

    model5 = LGBMClassifier( 

                           random_state=i

                          )

    #model = StackingClassifier([('cat', cat), ('xgb', xgb), ('lgb', lgb)])                          

    #model = XGBClassifier()

    model5.fit(X_train, y_train)

    #model.fit(xtrain, ytrain)

    

    

    p = model5.predict_proba(X_val)[:, 1]

    sc = eval_metric(y_val, p)

    score_list.append(sc)

    score += sc/folds

    

    pred = model5.predict_proba(test[features1])[:, 1]

    test_oofs.append(pred)

 

    print('Fold {} : {}'.format(i, sc))

 

print()

print()

print('Avg log : ', score)
oof_prediction = pd.DataFrame(test_oofs).T

oof_prediction.columns = ['fold_'+ str(i) for i in range(1, folds + 1)]

sub['default_status'] = np.mean(test_oofs, axis = 0)

sub.to_csv('model5.csv', index = False)

score_list = []

score = 0

test_oofs = []

 

for i, (train_index, val_index) in enumerate(skf.split(X, y)):

    

    X_train, y_train = X.loc[train_index, features], y.loc[train_index]

    X_val, y_val = X.loc[val_index, features], y.loc[val_index]

 

    model6 = CatBoostClassifier(**cat_params, task_type='GPU',random_seed = i, verbose=0)

   # model = AdaBoostClassifier(base_estimator=CatBoostClassifier( task_type='GPU',random_seed = i))

    #model = StackingClassifier(estimators=[('cat',CatBoostClassifier(**params, task_type='GPU',random_seed = i)), ('cat1',CatBoostClassifier(task_type='GPU',random_seed = i))])    

    

    #model = StackingClassifier([('cat', cat), ('xgb', xgb), ('lgb', lgb)])                          

    #model = XGBClassifier()

    model6.fit(X_train, y_train)

    #model.fit(xtrain, ytrain)

    

    

    p = model6.predict_proba(X_val)[:, 1]

    sc = eval_metric(y_val, p)

    score_list.append(sc)

    score += sc/folds

    

    pred = model6.predict_proba(test[features])[:, 1]

    test_oofs.append(pred)

 

    print('Fold {} : {}'.format(i, sc))

 

print()

print()

print('Avg log : ', score)
feature_importance_df = pd.DataFrame(model6.feature_importances_, columns=['importance'])

feature_importance_df['feature'] = features



plt.figure(figsize=(12, 10));

sns.barplot(x="importance", y="feature", data=feature_importance_df.sort_values(by = ['importance'], ascending = False).head(50))

plt.title('CatBoostClassifier features importance:');
oof_prediction = pd.DataFrame(test_oofs).T

oof_prediction.columns = ['fold_'+ str(i) for i in range(1, folds + 1)]

sub['default_status'] = np.mean(test_oofs, axis = 0)

sub.to_csv('model6.csv', index = False)

oof_prediction = pd.DataFrame(test_oofs).T

oof_prediction.columns = ['fold_'+ str(i) for i in range(1, folds + 1)]

sub['default_status'] = np.mean(test_oofs, axis = 0)

sub.to_csv('XGB_subwith_tuned.csv', index = False)

oof_prediction.columns = ['fold_'+ str(i) for i in range(1, folds + 1)]
oof_prediction.head()
#sub = pd.read_csv('/content/drive/My Drive/DSN AI Bootcamp Qualification Hackathon/SampleSubmission.csv')
sub['default_status'] = np.mean(test_oofs, axis = 0)
sub.to_csv('XGB_subwith_tuned.csv', index = False)
len(model.feature_importances_)
feature_importance_df = pd.DataFrame(model.feature_importances_, columns=['importance'])

feature_importance_df['feature'] = features



plt.figure(figsize=(12, 10));

sns.barplot(x="importance", y="feature", data=feature_importance_df.sort_values(by = ['importance'], ascending = False).head(50))

plt.title('XGBClassifier features importance:');
feature_importance_df = pd.DataFrame(model.feature_importances_, columns=['importance'])

feature_importance_df['feature'] = features



plt.figure(figsize=(12, 10));

sns.barplot(x="importance", y="feature", data=feature_importance_df.sort_values(by = ['importance'], ascending = False).head(50))

plt.title('XGBClassifier features importance:');
feature_importance_df = pd.DataFrame(model.feature_importances_, columns=['importance'])

feature_importance_df['feature'] = features



plt.figure(figsize=(12, 10));

sns.barplot(x="importance", y="feature", data=feature_importance_df.sort_values(by = ['importance'], ascending = False).head(50))

plt.title('LGBClassifier features importance:');
feature_importance_df = pd.DataFrame(model.feature_importances_, columns=['importance'])

feature_importance_df['feature'] = features



plt.figure(figsize=(12, 10));

sns.barplot(x="importance", y="feature", data=feature_importance_df.sort_values(by = ['importance'], ascending = False).head(50))

plt.title('LGBClassifier features importance:');
feature_importance_df = pd.DataFrame(model.feature_importances_, columns=['importance'])

feature_importance_df['feature'] = features



plt.figure(figsize=(12, 10));

sns.barplot(x="importance", y="feature", data=feature_importance_df.sort_values(by = ['importance'], ascending = False).head(50))

plt.title('LGBClassifier features importance with 48, 49:');