import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
fig,ax = plt.subplots(1,2,figsize=(10,5))
train.isnull().sum().plot.barh(rot=0,ax=ax[0],title='Number of missing values in the train dataset');
test.isnull().sum().plot.barh(rot=0,ax=ax[1], title='Number of missing values in the test dataset');
plt.tight_layout()
full_data = pd.concat([train,test],ignore_index=False)
full_data[:3]
missing_age_map = full_data.groupby(['Sex','Pclass','Parch','Embarked'],as_index=False).agg({'Age':'mean'})

train_age_merge = train.loc[train.Age.isnull(),:].drop(['Age'],axis=1).merge(missing_age_map,on=['Sex','Pclass','Parch','Embarked'],how='inner')
train_age_merge.index = train.loc[train.Age.isnull(),:].index
train.loc[train.Age.isnull(),'Age'] = train_age_merge['Age']

test_age_merge = test.loc[test.Age.isnull(),:].drop(['Age'],axis=1).merge(missing_age_map,on=['Sex','Pclass','Parch','Embarked'],how='inner')
test_age_merge.index = test.loc[test.Age.isnull(),:].index
test.loc[test.Age.isnull(),'Age'] = test_age_merge['Age']

train.Age.fillna(full_data.Age.mean(),inplace=True)
test.Age.fillna(full_data.Age.mean(),inplace=True)
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
train['Embarked'].fillna('S',inplace=True)
test['Embarked'].fillna('S',inplace=True)
fig,ax = plt.subplots(1,2,figsize=(10,5))
train.isnull().sum().plot.barh(rot=0,ax=ax[0],title='Number of missing values in the train dataset');
test.isnull().sum().plot.barh(rot=0,ax=ax[1], title='Number of missing values in the test dataset');
plt.tight_layout()
f = pd.concat([train,test],ignore_index=False).copy()
bins = [0,4,8,16,32,48,64,75,120]
bins_labels = ['1-4','4-8','8-16','16-32','32-48','48-64','64-75','75+']
f['AgeGroup'] = pd.cut(f.Age,bins=bins,labels=bins_labels)
f[:3]
pd.crosstab(f.Sex,f.AgeGroup).plot.bar(rot=0,cmap='Paired',figsize=(12,7));
sns.heatmap(f[['Pclass','Sex','Age','Parch','Fare','Embarked','Has_Cabin']].corr(),annot=True);
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
grid_params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'n_estimators': [100,300,600,1000]
        }
x_cols = [
    'Pclass',
    'Sex',
    'Age',
    'Parch',
    'Fare',
    'Embarked',
    'Has_Cabin'
]
encoding_maps = dict()
for col,tp in zip(x_cols,train[x_cols].dtypes):
    if tp == 'object':
        encoding_maps[col] = dict(zip(full_data[col].unique(),list(range(full_data[col].nunique()+1))))
        train[col] = train[col].replace(encoding_maps[col])
        test[col] = test[col].replace(encoding_maps[col])
        
print('Encoding Done')
xgc = xgb.XGBClassifier()
grid = GridSearchCV(xgc,grid_params,cv=5,verbose=10)
grid.fit(train[x_cols],train[['Survived']])
grid.best_params_
results = test.copy()
results['y_pred'] = grid.best_estimator_.predict(test[x_cols])
xgb.plot_importance(grid.best_estimator_);
results[['PassengerId','y_pred']].rename(columns={'y_pred':'Survived'}).to_csv('submission.csv',index=False)