# import the libraries

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Load dataset

train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
# training data description
print(train.shape)
train.head()
# test data description
print(test.shape)
test.head()
# check the submission file format.
print(gender_submission.shape)
gender_submission.head()
train.info()
test.info()
# combine dataset
train_test = pd.concat([train, test], sort=True)
train_test.info()
#Simply check the value counts
print(train_test['Pclass'].value_counts(), '\n')
print(train_test['Sex'].value_counts(), '\n')
print(train_test['SibSp'].value_counts(), '\n')
print(train_test['Parch'].value_counts(), '\n')
print(train_test['Ticket'].value_counts().head(10), '\n')
print(train_test['Fare'].value_counts().head(10), '\n')
print(train_test['Cabin'].value_counts().head(10), '\n')
print(train_test['Embarked'].value_counts(), '\n')
# Split data for visualization
df_survived = train_test[train_test['Survived'] == 1.0]
df_nonsurvived = train_test[train_test['Survived'] == 0.0]

labels = ['Survived', 'Nonsurvived']

fig, axs = plt.subplots(3, 2, figsize=(10, 10))
axs[0,0].hist([df_survived['Pclass'], df_nonsurvived['Pclass']], color=['blue', 'red'], alpha=0.5, bins='auto', label=labels)
axs[0,0].set_title('Pclass')
axs[0,0].legend()
axs[0,1].hist([df_survived['Sex'], df_nonsurvived['Sex']], color=['blue', 'red'], alpha=0.5, bins='auto', label=labels)
axs[0,1].set_title('Sex')
axs[0,1].legend()
axs[1,0].hist([df_survived['Age'], df_nonsurvived['Age']], color=['blue', 'red'], alpha=0.5, range=(0, 90), label=labels)
axs[1,0].set_title('Age')
axs[1,0].legend()
axs[1,1].hist([df_survived['SibSp'], df_nonsurvived['SibSp']], color=['blue', 'red'], alpha=0.5, bins='auto', label=labels)
axs[1,1].set_title('SibSp')
axs[1,1].legend()
axs[2,0].hist([df_survived['Parch'], df_nonsurvived['Parch']], color=['blue', 'red'], alpha=0.5, bins='auto', label=labels)
axs[2,0].set_title('Parch')
axs[2,0].legend()
axs[2,1].hist([df_survived['Embarked'].fillna('missval'), df_nonsurvived['Embarked'].fillna('missval')], color=['blue', 'red'], alpha=0.5, label=labels)
axs[2,1].set_title('Embarked')
axs[2,1].legend()
# separate alphabetic character and cabin number
df_cabin = pd.DataFrame()
df_cabin['survived'] = train_test['Survived']

# unavailable value settings
df_cabin['cab'] = train_test['Cabin'].fillna('N').apply(lambda x : x[:1])
df_cabin['cabnum'] = train_test['Cabin'].fillna('N').apply(lambda x : x[1:3])
df_cabin.loc[df_cabin['cabnum'].str.contains(' '), 'cabnum'] = -1
df_cabin.loc[df_cabin['cabnum'] == '', 'cabnum'] = -1

df_cabin['cabnum'].astype(int)
#pd.set_option('display.max_rows', 90)
df_cabin['cabnum'].value_counts().head(10)
df_cabin['cab'].value_counts()
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.hist([df_cabin[df_cabin['survived']==0.0]['cab'], df_cabin[df_cabin['survived']==1.0]['cab']], alpha=0.5, bins='auto', color=['red', 'blue'])
ax.set_title('Cabin')
ax.legend(['Survived', 'Nonsurvived'])
#ax[1].hist([temp[temp['survived']==0.0]['cabnum'], temp[temp['survived']==1.0]['cabnum']], alpha=0.5, color=['red', 'blue'], bins='auto')
#ax[1].set_title('number')
df_feature = train_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# binary description about sex
le = LabelEncoder()
df_feature['Sex'] = le.fit_transform(df_feature['Sex'])

# one-hot description for embarked feature
df_feature = pd.get_dummies(df_feature, columns=['Embarked'], drop_first=True)
df_feature.head()
df_feature.info()
df_feature.describe()
df_feature['Age'].fillna(-1, inplace=True)
# only one NaN value is in Fare feature
df_feature[df_feature['Fare'].isnull()]
#　Interpolate with mean value
#feature.loc[feature['Fare'].isnull(), 'Fare'] = feature['Fare'].mean()
df_feature.loc[df_feature['Fare'].isnull(), 'Fare'] = df_feature.query('Pclass == 3 and Parch == 0 and SibSp == 0 and Embarked_Q == 0 and Embarked_S == 1 and Sex == 1')['Fare'].mean()
train_X = df_feature[df_feature['Survived'].notnull()].drop(['Survived'], axis=1)
train_y = df_feature.loc[df_feature['Survived'].notnull(), 'Survived']
test_X = df_feature[df_feature['Survived'].isnull()].drop(['Survived'], axis=1)
test_y = df_feature.loc[df_feature['Survived'].isnull(), 'Survived']
# AdaBoost Classifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1)
model = AdaBoostClassifier(base_estimator=tree, n_estimators=200, learning_rate=00.1, random_state=1)

# Cross validation
#score = cross_val_score(estimator=model, X=train_X, y=train_y, cv=4, n_jobs=-1)
#score

'''
# Grid Search
param_grid = {'base_estimator__max_depth': [2], 
                          'n_estimators': np.array([50, 100, 200, 300, 400, 500]), 
                          'learning_rate': [0.01]}
gs = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', random_state=1), random_state=1), 
                                  param_grid=param_grid, 
                                  cv=4)
gs.fit(train_X, train_y)
print('Best cross validation score:', gs.best_score_)
print('Best parameters: ', gs.best_params_)
'''

model.fit(train_X, train_y)
model.score(train_X, train_y)

z = model.predict(test_X).astype(int)

gender_submission['Survived'] = z
gender_submission.to_csv('adaboost.csv', index=False)
# Random Forest Classifier
model = RandomForestClassifier(random_state=1)

# Cross validation
#score = cross_val_score(estimator=model, X=train_X, y=train_y, cv=4, n_jobs=-1)
#score

model.fit(train_X, train_y)
model.score(train_X, train_y)

z = model.predict(test_X).astype(int)

gender_submission['Survived'] = z
gender_submission.to_csv('random_forest.csv', index=False)
# LightGBM Classifier
# Cross validation with accuracy

kfold = StratifiedKFold(n_splits=4, random_state=1, shuffle=True).split(train_X, train_y)
scores = []
lgbm_params = {'objective': 'binary', 'seed': 1, 'metric': 'binary_logloss'}

for k, (train_id, test_id) in enumerate(kfold):
    
    eval_X, ktest_X, eval_y, ktest_y = train_test_split(train_X.iloc[test_id], train_y.iloc[test_id], test_size=0.5, random_state=1, stratify=train_y.iloc[test_id])
    
    lgb_train = lgb.Dataset(train_X.iloc[train_id], train_y.iloc[train_id])
    lgb_eval = lgb.Dataset(eval_X, eval_y)
    
    model = lgb.train(lgbm_params, lgb_train, num_boost_round=1000, valid_names=['train', 'valid'], valid_sets=[lgb_train, lgb_eval], early_stopping_rounds=5) 
    pred = model.predict(ktest_X, num_iteration=model.best_iteration)
    
    # convert class probability into binary
    pred_class = np.empty((len(pred), ))
    
    for i, prob in enumerate(pred):
        if prob > 0.5:
            pred_class[i] = 1.0
        else:
            pred_class[i] = 0.0
            
    scores.append(accuracy_score(ktest_y, pred_class))
    
scores
# Light GBM Classifier 
# Cross validationb with logloss 

lgbm_params = {'objective': 'binary', 'seed': 1, 'metric': 'binary_logloss'}

lgb_train = lgb.Dataset(train_X, train_y)

cv_results = lgb.cv(lgbm_params, lgb_train, nfold=4)
cv_logloss = cv_results['binary_logloss-mean']
round_n = np.arange(len(cv_logloss))

plt.xlabel('round')
plt.ylabel('logloss')
plt.plot(round_n, cv_logloss)
plt.show()
# Training LightGBM

lgbm_params = {'objective': 'binary', 'metric': 'binary_logloss', 'seed': 1}

lgbm_train = lgb.Dataset(train_X, train_y)
model = lgb.train(lgbm_params, lgbm_train, num_boost_round=30)

pred = model.predict(test_X)
        
# convert class probability into binary
pred_class = np.empty((len(pred), ), dtype=int)

for i, prob in enumerate(pred):
    if prob > 0.5:
        pred_class[i] = 1
    else:
        pred_class[i] = 0
        
gender_submission['Survived'] = pred_class
gender_submission.to_csv('lightgbm.csv', index=False)
# SVM Classifier

sc = StandardScaler()
sc.fit(train_X)
train_X_std = sc.transform(train_X)

# SVM Grid Search
'''
param_grid = {'C': np.logspace(-3, 2, num=6), 'gamma': np.logspace(-3, 2, num=6)}
gs = GridSearchCV(estimator=SVC(kernel='rbf', random_state=1), param_grid=param_grid, cv=5)
gs.fit(train_X_std, train_y)
print('Best cross validation score:', gs.best_score_)
print('Best parameters: ', gs.best_params_)
'''

#svc = SVC(kernel='rbf', random_state=1, gamma=0.01, C=100.0)

# Cross validation
#svc_score = cross_val_score(estimator=svc, X=train_X_std, y=train_y, cv=4, n_jobs=-1)

#svc.fit(train_X, train_y)
#print(svc.score(train_X, train_y))

#z = svc.predict(test_X).astype(int)