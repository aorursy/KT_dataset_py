import pandas as pd

import numpy as np
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

sample_submission = pd.read_csv('../input/titanic/gender_submission.csv')
#import pandas_profiling as pdp

#pdp.ProfileReport(train)
all_data=pd.concat([train,test])



all_data['Age'].fillna(all_data['Age'].median(), inplace=True)

all_data.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)



all_data["Family"] = all_data["SibSp"]+all_data["Parch"]+1
import seaborn as sns

sns.countplot(all_data['Family'], hue=all_data['Survived'])
all_data.loc[(all_data['Family'] >= 8), 'Family_group'] = '4'

all_data.loc[(all_data['Family'] >= 5) & (all_data['Family'] <= 7),'Family_group'] = '3'

all_data.loc[(all_data['Family'] >= 2) & (all_data['Family'] <= 4), 'Family_group'] = '2'

all_data.loc[(all_data['Family'] == 1), 'Family_group'] = '1'

all_data.drop("Family",axis=1,inplace=True)
all_data.info()
all_data['Family_group']=all_data['Family_group'].astype(int)
all_data.head()
all_data['Fareround'] = pd.qcut(all_data['Fare'], 4)

all_data[['Fareround', 'Survived']].groupby(['Fareround'], as_index=False).mean().sort_values(by='Fareround', ascending=True)
sns.countplot(all_data['Fareround'], hue=all_data['Survived'])
all_data.info()
all_data.loc[(all_data['Fare']<=7.19), 'Fare'] = 0

all_data.loc[(all_data['Fare']>=7.18) & (all_data['Fare']<=14.454),'Fare'] = 1

all_data.loc[(all_data['Fare']>14.454) & (all_data['Fare']<=31.0),'Fare'] = 2

all_data.loc[(all_data['Fare']>31.0),'Fare'] = 3



all_data = all_data.drop(['Fareround'], axis=1)

all_data.head()
train = all_data[:891].copy()

test = all_data[891:].copy()
train = pd.get_dummies(train, columns=['Sex', 'Embarked'])

test = pd.get_dummies(test, columns=['Sex', 'Embarked'])
train.head()
X_train = train.drop(['Survived'], axis=1) 

Y_train = train['Survived']  
import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



train_x, valid_x, train_y, valid_y = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)



gbm = lgb.LGBMClassifier(objective='binary')



gbm.fit(train_x, train_y, eval_set = [(valid_x, valid_y)],early_stopping_rounds=20, verbose=20)
oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration_) 

print('score', round(accuracy_score(valid_y, oof)*100,2), '%')

gbm.best_iteration_
test.drop("Survived",axis=1,inplace=True)
test.head()
test_pred = gbm.predict(test, num_iteration=gbm.best_iteration_)

sample_submission['Survived'] = test_pred 

sample_submission.to_csv('train_test_split.csv', index=False) 
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)



# スコアとモデルを格納するリスト

score_list = []

models = []



for fold_, (train_index, valid_index) in enumerate(kf.split(X_train, Y_train)):

    train_x = X_train.iloc[train_index]

    valid_x = X_train.iloc[valid_index]

    train_y = Y_train[train_index]

    valid_y = Y_train[valid_index]

    

    print(f'fold{fold_ + 1} start')



    gbm = lgb.LGBMClassifier(objective='binary')

    gbm.fit(train_x, train_y, eval_set = [(valid_x, valid_y)],early_stopping_rounds=20,verbose= -1)

    

    oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)

    score_list.append(round(accuracy_score(valid_y, oof)*100,2))

    models.append(gbm)  

    print(f'fold{fold_ + 1} end\n' )

print(score_list, '平均score', np.mean(score_list), "%")  
# testの予測

test_pred = np.zeros((len(test), 5)) 



for fold_, gbm in enumerate(models):  

    pred_ = gbm.predict(test, num_iteration=gbm.best_iteration_)  

    test_pred[:, fold_] = pred_  

pred = (np.mean(test_pred, axis=1) > 0.5).astype(int) 

sample_submission['Survived'] = pred

sample_submission.to_csv('5-fold_cross-validation.csv',index = False)
gbm.get_params()
# GridSearchCVをimport

from sklearn.model_selection import GridSearchCV



gbm = lgb.LGBMClassifier(objective='binary')



# 試行するパラメータを羅列する

params = {

    'max_depth': [2, 3, 4, 5],

    'reg_alpha': [0, 1, 10, 100],

    'reg_lambda': [0, 1, 10, 100],

}



grid_search = GridSearchCV(

                           gbm,  # 分類器を渡す

                           param_grid=params,  # 試行してほしいパラメータを渡す

                           cv=3,  # 3分割交差検証でスコアを確認

                          )



grid_search.fit(X_train, Y_train)  # データを渡す



print(grid_search.best_score_)  # ベストスコアを表示

print(grid_search.best_params_)  # ベストスコアのパラメータを表示
# スコアとモデルを格納するリスト

score_list = []

test_pred = np.zeros((len(test), 5))





for fold_, (train_index, valid_index) in enumerate(kf.split(X_train, Y_train)):

    train_x = X_train.iloc[train_index]

    valid_x = X_train.iloc[valid_index]

    train_y = Y_train[train_index]

    valid_y = Y_train[valid_index]

    

    print(f'fold{fold_ + 1} start')



    gbm = lgb.LGBMClassifier(objective='binary', max_depth=3, reg_alpha=1,

                             reg_lambda=0)

    gbm.fit(train_x, train_y,

            eval_set = [(valid_x, valid_y)],

            early_stopping_rounds=20,

            verbose= -1)

    

    oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)

    score_list.append(round(accuracy_score(valid_y, oof)*100,2))

    test_pred[:, fold_] = gbm.predict(test, num_iteration=gbm.best_iteration_)

    print(f'fold{fold_ + 1} end\n' )

print(score_list, '平均score', np.mean(score_list))

pred = (np.mean(test_pred, axis=1) > 0.5).astype(int)

sample_submission['Survived'] = pred

sample_submission.to_csv('glid_search.csv', index=False)  # scoreは0.77511