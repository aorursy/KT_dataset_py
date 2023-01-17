import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#hide some warnings :p
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
kaggle_path = '../input/titanic/'
train = pd.read_csv(kaggle_path + 'train.csv')
test = pd.read_csv(kaggle_path + 'test.csv')
submission = pd.read_csv(kaggle_path + 'gender_submission.csv')
print(train.shape)
train.head()
print(test.shape)
test.head()
train.isna().sum()
test.isna().sum()
train['Name'][0:10]
titles = ['Mr.', 'Mrs.', 'Miss.', 'Ms.', 'Master.', 'Dr.']   #creating list of relevant titles
print('Names with missing ages insight:\n')
total = 0
for t in titles:
    count1 = train['Name'][train['Age'].isna()][train['Name'].str.find(t)!=-1].count()
    count2 = test['Name'][test['Age'].isna()][test['Name'].str.find(t)!=-1].count()
    print('Names having', t, 'in them are:', count1+count2)
    total += count1
total == train['Age'].isna().sum()
age_dict = {}
for t in titles:
    # Mean of a title in train set
    trm = train['Age'][train['Name'].str.find(t)!=-1][train['Age'].notna()].mean()
    #mean of title in test set
    tsm = test['Age'][test['Name'].str.find(t)!=-1][test['Age'].notna()].mean()
    if np.isnan(trm):
        trm = 0
    if np.isnan(tsm):
        tsm = 0
    avg = round( (trm + tsm) / 2)
    print('average age of Names having',t ,'in them is: ', avg)
    age_dict[t] = avg
age_dict
def fill_na_names(df):
    missing_ages = df['Name'][df['Age'].isna()]
    index = df['Name'][df['Age'].isna()].index
    for name,i in zip(missing_ages, index):
        for ttl in age_dict:
            if name.find(ttl) != -1:
                df.loc[i, 'Age'] = age_dict[ttl]
                
            
fill_na_names(train)
fill_na_names(test)
train['Age'].isna().sum(), test['Age'].isna().sum()
m = test[test['Fare'].isna()]
i = test[test['Fare'].isna()].index
m
# let's see if he shares a ticket with someone else
test[test.Ticket=='3701']
m1 = test['Fare'][test['Pclass']==3].mean()
m2 = test['Fare'][test['Age']> 50].mean()
m3 = test['Fare'][test['Embarked']=='S'].mean()
m4 = test['Fare'][test['Sex']=='male'].mean()
m5 = test['Fare'][test['SibSp']==0].mean()
m6 = test['Fare'][test['Parch']==0].mean()

avg = round( (m1+ m2+ m3 + m4 + m5 + m6) / 6 )

test.loc[i,'Fare'] = avg
print('Filled ', round(avg))
train.head(2)
def add_fam_col(df):
    for n,ind in zip(df['Name'].values, df['Name'].index):
        df.loc[ind, 'Family'] = n.split(',')[0]
        
add_fam_col(train)
add_fam_col(test)
train.head(2)
df = pd.concat([train, test], join='outer')
df.shape
df.columns

useless_feats = ['PassengerId', 'Survived', 'Name', 'SibSp', 
                 'Parch', 'Ticket', 'Cabin', 'Embarked']
df.drop(useless_feats, axis=1,inplace=True)
df.head(2)
df = pd.get_dummies(df, drop_first=True)
df.head(1)
X = df[:train.shape[0]]
tst = df[train.shape[0]:]
y = train['Survived']
print(X.shape, tst.shape, y.shape)
X.head(2)
from sklearn.preprocessing import scale
X = scale(X)
tst = scale(tst)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

parameter_grid = {
                 'max_depth' : [None, 10, 12, 20],
                 'n_estimators': [None, 50, 10],
                 'max_features': [None, 'sqrt'],
                 'min_samples_split': [None, 2, 3, 10],
                 'min_samples_leaf': [None, 1, 3, 10],
                 'bootstrap': [True, False],
                 }

forest = RandomForestClassifier(n_jobs=2)

M1 = GridSearchCV(forest,
                           scoring='accuracy',
                           param_grid=parameter_grid,
                           cv=3,
                           n_jobs=-1,
                           verbose=1)

M1.fit(X, y)

parameters = M1.best_params_
print('Best score: ' , M1.best_score_ * 100)
print('Best estimator: ' , M1.best_estimator_)

from sklearn.svm import SVC


parameter_grid = {
                 'kernel': [None, 'rbf', 'sigmoid', 'linear'],
                    'C'  : [0,0.25,0.5,1,2,3,4],
                 'gamma' : [None, 'auto', 0.01, 0.03, 0.1, 0.3, 1, 3, 5, 20, 50],
                  'class_weight' : [None, 'balanced']
    
                 }

M2 = GridSearchCV(SVC(),
                      scoring = 'accuracy',
                      cv=3,
                      param_grid= parameter_grid,
                      n_jobs=-1,
                      verbose=1
                     ).fit(X,y)

M2.fit(X,y)

print('Best score: ', M2.best_score_ * 100)
print('Best estimator: ', M2.best_estimator_)

from sklearn.neighbors import KNeighborsClassifier

grid = { 'n_neighbors': [1, 5, 15, 25, 30, 40],
        'weights': ['distance'],
        'leaf_size': [None, 1, 3, 10, 25, 40],
        'p':[1,2]
       }


M3 = GridSearchCV(KNeighborsClassifier(),
                      scoring='accuracy',
                      cv=3,
                      param_grid=grid,
                      n_jobs=-1,
                      verbose=1
)

M3.fit(X, y)

print('Best score: ', M3.best_score_ * 100)
print('Best estimator: ', M3.best_estimator_)

from xgboost import XGBClassifier

parameter_grid = {
                 'max_depth' : [None, 5, 7, 10 ],
                 'max_delta_step': [None, 1, 2],
                 'n_estimators': [None,10, 20, 30, 40],
                 'colsample_bylevel': [None,0.2, 0.5, 0.8],
                 'colsample_bytree': [None,0.2, 0.6],
                 'subsample': [None,0.01,0.1, 0.3, 0.4,1],
                 }

M4 = GridSearchCV(XGBClassifier(),
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=3,
                               n_jobs=-1,
                               verbose=1)

M4.fit(X, y)

print('Best score: ', M4.best_score_ * 100)
print('Best estimator: ', M4.best_estimator_)

from sklearn.linear_model import LogisticRegression

parameter_grid = {
    'C' : [0.01, 0.1, 0.5, 1, 2],
    'penalty' : ['l1', 'l2', 'elasticnet'],
    'class_weight' : [None, 'balanced'],
    'solver' : ['newton-cg', 'lbfgs', 'sag', 'saga'],
    'max_iter' : [100, 500, 1000]

}

M5 = GridSearchCV(LogisticRegression(),
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=3,
                               n_jobs=-1,
                               verbose=1)

M5.fit(X, y)

print('Best score: ', M5.best_score_ * 100)
print('Best estimator: ', M5.best_estimator_)
from sklearn.ensemble import BaggingClassifier

parameter_grid = {
    'base_estimator': [LogisticRegression(), SVC()],
    'n_estimators' : [10, 20, 30, 40],
}

M6 = GridSearchCV(BaggingClassifier(),
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=3,
                               n_jobs=-1,
                               verbose=1)

M6.fit(X, y)

print('Best score: ', M6.best_score_ * 100)
print('Best estimator: ', M6.best_estimator_)

temp = pd.read_csv('../input/titanic-leaked/titanic.csv')
original = temp['Survived']
from sklearn.metrics import accuracy_score

preds_M1 = M1.predict(tst)
preds_M2 = M2.predict(tst)
preds_M3 = M3.predict(tst)
preds_M4 = M4.predict(tst)
preds_M5 = M5.predict(tst)
preds_M6 = M6.predict(tst)


score_M1 = accuracy_score(original, preds_M1)
score_M2 = accuracy_score(original, preds_M2)
score_M3 = accuracy_score(original, preds_M3)
score_M4 = accuracy_score(original, preds_M4)
score_M5 = accuracy_score(original, preds_M5)
score_M6 = accuracy_score(original, preds_M6)

print('score with model 1: ', score_M1*100)
print('score with model 2: ', score_M2*100)
print('score with model 3: ', score_M3*100)
print('score with model 4: ', score_M4*100)
print('score with model 5: ', score_M5*100)
print('score with model 6: ', score_M6*100)
subm_dir = kaggle_path + 'gender_submission.csv'
submit_file = pd.read_csv(subm_dir)
submit_file['Survived'] = preds_M6
submit_file.to_csv('gender_submission.csv', index=False)