import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from sklearn.compose import make_column_transformer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.svm import  SVC

from sklearn.decomposition import PCA

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler,RobustScaler

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.pipeline import make_pipeline

pd.set_option('display.max_columns', 10)

pd.set_option('display.max_rows', 1000)

training = pd.read_csv('../input/summeranalytics2020/train.csv')

train_data = training.copy()

test_data = pd.read_csv('../input/summeranalytics2020/test.csv')
train_data.head()
train_data.info()
print(train_data.describe())
train_id = train_data.Id

train_data = train_data.drop(['Behaviour','Id'],axis = 1)



test_id = test_data.Id

test_data = test_data.drop(['Behaviour','Id'],axis = 1)
train_data['PerformanceRating'] = train_data['PerformanceRating'].apply(lambda x: 0 if x == 3 else 1)

test_data['PerformanceRating'] = test_data['PerformanceRating'].apply(lambda x: 0 if x == 3 else 1)
train_data['Attrition'].value_counts().plot(kind = 'bar')
print('Number of duplicates: ',train_data.duplicated().sum())
train_data[train_data.duplicated()]['Attrition'].value_counts().plot(kind = 'bar')
# drop them



train_unq = train_data.drop_duplicates()

print('New train set: ',train_unq.shape)

X = train_unq.drop('Attrition',axis = 1)

y = train_unq['Attrition']

y.value_counts().plot(kind = 'bar')

plt.show()
# Standard Scaling

skf = StratifiedKFold(n_splits = 10,random_state=42,shuffle=True)



categorical = [f for f in training.columns if training[f].dtype == object]

numeric = [f for f in X.columns if f not in categorical+['Id','Attrition','Behaviour','PerformanceRating']]



pre_pipe = make_column_transformer((OneHotEncoder(),categorical),(StandardScaler(),numeric))
pipe_rf = make_pipeline(pre_pipe,RandomForestClassifier())

pipe_xgb = make_pipeline(pre_pipe,XGBClassifier())

pipe_svc = make_pipeline(pre_pipe,SVC(probability=True))





print('RF: ',np.mean(cross_val_score(X=X,y=y,cv=skf,estimator=pipe_rf,scoring='roc_auc')))

print('XGB: ',np.mean(cross_val_score(X=X,y=y,cv=skf,estimator=pipe_xgb,scoring='roc_auc')))

print('SVC:',np.mean(cross_val_score(X=X,y=y,cv=skf,estimator=pipe_svc,scoring='roc_auc')))
n = 46

pipe_svc = make_pipeline(pre_pipe,PCA(n_components=n),SVC(probability=True,C = 1,kernel='rbf'))

print('SVC: ',np.mean(cross_val_score(X=X,y=y,cv=skf,estimator=pipe_svc,scoring='roc_auc')))



plt.figure(figsize=(10,8))

pipe_svc.fit(X,y)

plt.plot(range(1,n+1),pipe_svc.named_steps['pca'].explained_variance_ratio_.cumsum())

plt.xticks(range(1,n+1,2))

plt.title('Explained Variance')

plt.grid()

plt.show()
n = 34

pre_pipe = make_column_transformer((OneHotEncoder(),categorical),(StandardScaler(),numeric),remainder = 'passthrough')

pipe_svc = make_pipeline(pre_pipe,PCA(n_components=n),SVC(probability=True,C = 1,kernel='rbf'))

print('SVC: ',np.mean(cross_val_score(X=X,y=y,cv=skf,estimator=pipe_svc,scoring='roc_auc')))
n = 34

pre_pipe = make_column_transformer((OneHotEncoder(),categorical),(StandardScaler(),numeric),remainder = 'passthrough')

pipe_svc = make_pipeline(pre_pipe,PCA(n_components=n),SVC(probability=True,C = 1,kernel = 'rbf'))



param_grid = {

    

    'svc__C':[0.001,0.01,0.1,1,10,100,1000],

    'svc__gamma': ['auto','scale'],

    'svc__class_weight': ['balanced',None]

}    



grid_search = GridSearchCV(pipe_svc,param_grid=param_grid,cv = skf, verbose=2, n_jobs = -1,scoring='roc_auc')

grid_search.fit(X,y)

print('Best score ',grid_search.best_score_)

print('Best parameters ',grid_search.best_params_)

best_svc = grid_search.best_estimator_
pipe_svc = make_pipeline(pre_pipe,PCA(n_components=n),SVC(probability=True,C = 1,kernel='rbf',class_weight=None,gamma='auto'))

param_grid={

    'svc__C':[0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7,1]  

}

grid_search = GridSearchCV(pipe_svc,param_grid=param_grid,cv = skf, verbose=2, n_jobs = -1,scoring = 'roc_auc')

grid_search.fit(X,y)

print('Best score ',grid_search.best_score_)

print('Best parameters ',grid_search.best_params_)

best_svc = grid_search.best_estimator_ # final model - 0.808 private LB*
best_svc.fit(X,y)

prediction = best_svc.predict_proba(test_data)[:,1]

submission = pd.DataFrame(prediction,columns=['Attrition'])

submission['Id'] = test_id

submission = submission[['Id','Attrition']]

submission.to_csv('submissionfile_postcomp.csv',index = None)