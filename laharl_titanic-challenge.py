import numpy as np

import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

import re



import xgboost as xgb

%matplotlib inline



from sklearn.ensemble import (

    RandomForestClassifier,

    GradientBoostingClassifier,

    AdaBoostClassifier,

    ExtraTreesClassifier

)

from sklearn.tree import (DecisionTreeClassifier)

from sklearn.svm import SVC



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import KFold



from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



PassengerId = test['PassengerId']



train.head(2)
train.shape
full_data = [train, test]

# 新しい特徴量「客室の有無 Has_Cabin」 客室を取っている客と取っていない客がいる

train['Has_Cabin'] = train['Cabin'].apply(lambda x:0 if type(x) == float else 1)

test['Has_Cabin'] = test['Cabin'].apply(lambda x:0 if type(x) == float else 1)



# (lambda x:0 if type(x) == float else 1)の中身

# a = np.nan

# if(type(a) == float):

#     print('NAN')

# else:

#     print('N')



# 性別をマッピングする

for dataset in full_data:

    # 値の抜けを平均値で埋める

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)



    # 性別をマッピング

    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    

    dataset.loc[dataset['Age'] <= 16,'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32) ,'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48) ,'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64) ,'Age'] = 3

    dataset.loc[dataset['Age'] > 64,'Age'] = 4;

    

for dataset in full_data:

    #家族サイズ

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
train.head()
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Fare', 'Embarked', 'Cabin', 'SibSp', 'Parch']



train = train.drop(drop_elements, axis = 1)

test = test.drop(drop_elements, axis = 1)

train.head()
# ヒートマップで相関関係を確認する

colormap = plt.cm.RdBu

plt.figure(figsize=(14, 12))

plt.title('Pearson Correlation of Feature', y=1.05, size=15)



#train.corr() 各データの相関指数をだす関数

sns.heatmap(train.astype(float).corr(), linewidth=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
ntrain = train.shape[0]

ntest = test.shape[0]



# 乱数の固定値

SEED = 0

# 交差検証の回数

NFOLDS = 5

#交差検証でパラメータを検証して、過学習が起こらないパラメータを決定する

kf = KFold(n_splits=NFOLDS, random_state=SEED)



# 生存したかどうかのラベル

y_train = train['Survived'].ravel()

train = train.drop(['Survived'], axis=1)

X_train = train.values

X_test = test.values
forest = RandomForestClassifier(random_state=0, n_estimators=500, max_depth=5)

forest.fit(X_train, y_train)

forest.score(X_train, y_train)
gbrt = GradientBoostingClassifier(random_state=0, n_estimators=500, max_depth=5)

gbrt.fit(X_train, y_train)



gbrt.score(X_train, y_train)
svm = SVC(kernel='linear', C=0.025)

svm.fit(X_train, y_train)

svm.score(X_train, y_train)
# ヘルパー関数

class SklearnHelper(object):

    def __init__(self, clf_model, seed=0, params=None):

        params['random_state'] = seed

        self.clf_model = clf_model(**params)

        

    def train(self, X_train, y_train):

        self.clf_model.fit(X_train, y_train)

        

    def predict(self, X):

        return self.clf_model.predict(X)

    

    def fit(self, X, y):

        self.clf_model.fit(X, y)

        

    def feature_importances(self, X, y):

        return self.clf_model.fit(X, y).feature_importances_

    

    def print_feature_importances(self, X, y):

        print(self.clf_model.fit(X, y).feature_importances_)

        
forest_params = {

    'n_jobs': -1,

    'n_estimators': 500,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0 

}



gb_params = {

    'n_estimators': 500,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'learning_rate': 0.1,

    'verbose': 0

}



svc_params = {

    'kernel': 'linear',

    'C': 0.025

}



et_params = {

    'n_jobs': -1,

    'n_estimators':500,

    #'max_features': 0.5,

    'max_depth': 8,

    'min_samples_leaf': 2,

    'verbose': 0

}



# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.75

}
forest_helper = SklearnHelper(clf_model=RandomForestClassifier, seed=SEED, params=forest_params)

gb_helper = SklearnHelper(clf_model=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc_helper = SklearnHelper(clf_model=SVC, seed=SEED, params=svc_params)
forest_importances = forest_helper.feature_importances(X_train, y_train)

gb_importances = gb_helper.feature_importances(X_train, y_train)



feature_dataframe = pd.DataFrame({

    'features': train.columns.values,

    'random forest': forest_importances,

    'gb': gb_importances

})



feature_dataframe.head()
# 交差検証

forest_scores = cross_val_score(forest, X_train, y_train, cv=kf)

forest_scores.mean()
gbrt_scores = cross_val_score(gbrt, X_train, y_train, cv=kf)

gbrt_scores.mean()
svc_scores = cross_val_score(svm, X_train, y_train, cv=kf)

svc_scores.mean()
def get_oof(clf_model, X_train, y_train, X_test):

    # 引数の分のサイズの値が0の配列を生成する

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    # 未初期化の配列を生成する。この場合は[5, ntestの数]のサイズの配列。入っている値はデタラメ

    oof_test_skf = np.empty((NFOLDS, ntest))

    

    for i, (train_index, test_index) in enumerate(kf.split(train)):

        x_tr = X_train[train_index]

        y_tr = y_train[train_index]

        x_te = X_train[test_index]



        clf_model.fit(x_tr, y_tr)

        oof_train[test_index] = clf_model.predict(x_te)

        oof_test_skf[i, :] = clf_model.predict(X_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
ar = np.array([[1,2,3], [4,5,6]])

ar[:,0]
dtree = DecisionTreeClassifier(random_state=0)

dtree.fit(X_train, y_train)
train_data, test_data, train_target, test_target = train_test_split(X_train, y_train, random_state=0)
dtree = DecisionTreeClassifier(random_state=0, max_depth=8)

dtree.fit(X_train, y_train)



ada = AdaBoostClassifier(random_state=0, n_estimators=500, learning_rate=0.8)

ada.fit(X_train, y_train)



ex = ExtraTreesClassifier(random_state=0, n_estimators=500, max_depth=8)

ex.fit(X_train, y_train)
dtree_pre = dtree.predict(X_train).reshape(-1, 1)

dtree_pre_test = dtree.predict(X_test).reshape(-1, 1)



ada_pre = ada.predict(X_train).reshape(-1, 1)

ada_pre_test = ada.predict(X_test).reshape(-1, 1)



ex_pre = ex.predict(X_train).reshape(-1, 1)

ex_pre_test = ex.predict(X_test).reshape(-1, 1)
forest_pre = forest.predict(X_train).reshape(-1, 1)

forest_pre_test = forest.predict(X_test).reshape(-1, 1)



svm_pre = svm.predict(X_train).reshape(-1, 1)

svm_pre_test = svm.predict(X_test).reshape(-1, 1)
gb_pre = gbrt.predict(X_train).reshape(-1, 1)

gb_pre_test = gbrt.predict(X_test).reshape(-1, 1)
forest_rf_tr, forest_rf_te = get_oof(forest, X_train, y_train, X_test)

gbrt_rf_tr, gbrt_rf_te = get_oof(gbrt, X_train, y_train, X_test)

svm_rf_tr, svm_rf_te = get_oof(svm, X_train, y_train, X_test)

dtree_rf_tr, dtree_rf_te = get_oof(dtree, X_train, y_train, X_test)

ada_rf_tr, ada_rf_te = get_oof(ada, X_train, y_train, X_test)

ex_rf_tr, ex_rf_te = get_oof(ex, X_train, y_train, X_test)
x_train = np.concatenate(

    (

        forest_rf_tr,

        gbrt_rf_tr,

        svm_rf_tr,

        dtree_rf_tr,

        ada_rf_tr,

        ex_rf_tr

    ),

    axis=1

)



x_test = np.concatenate(

    (

        forest_rf_te,

        gbrt_rf_te,

        svm_rf_te,

        dtree_rf_te,

        ada_rf_te,

        ex_rf_te

    ),

    axis=1

)



# x_train = x_train.reshape(-1, 1)

# x_test = x_test.reshape(-1, 1)

# y_train.reshape(-1, 1)



xg = xgb.XGBClassifier(

    n_estimators = 2000,

    max_depth = 4,

    min_child_weight = 2,

    gamma = 0.9,

    subsample = 0.8,

    colsample_bytree=0.8,

    objective= 'binary:logistic'

)
xg.fit(x_train, y_train)

xg.score(x_train, y_train)
predictions = xg.predict(x_test)
# StackingSubmission = pd.DataFrame({

#     'PassengerID': PassengerId,

#     'Survived': predictions

# })



# StackingSubmission.to_csv('StackingSubmission.csv', index = False)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



PassengerId = test['PassengerId']

full_data = [train, test]





# 新しい特徴量「客室の有無 Has_Cabin」 客室を取っている客と取っていない客がいる

train['Has_Cabin'] = train['Cabin'].apply(lambda x:0 if type(x) == float else 1)

test['Has_Cabin'] = test['Cabin'].apply(lambda x:0 if type(x) == float else 1)



# (lambda x:0 if type(x) == float else 1)の中身

# a = np.nan

# if(type(a) == float):

#     print('NAN')

#     x = 0

# else:

#     print('N')

#     x = 1



#乗客の名前から敬称(Mr.Msなど)を取得する関数

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return



#Create a new feature Title

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)



# 性別をマッピングする

for dataset in full_data:

    # 値の抜けを平均値で埋める

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)



    # 性別をマッピング

    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    

    dataset.loc[dataset['Age'] <= 16,'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32) ,'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48) ,'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64) ,'Age'] = 3

    dataset.loc[dataset['Age'] > 64,'Age'] = 4;

    

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

for dataset in full_data:

    #Mapping titles

    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    #家族サイズ

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

    dataset['NameLen'] = dataset['Name'].apply(len)    

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

    dataset.loc[dataset['Fare'] <= 7.9, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.9 ) & (dataset['Fare'] <= 15.8), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 15.8) & (dataset['Fare'] <= 23.7), 'Fare'] = 2

    dataset.loc[(dataset['Fare'] > 23.7) & (dataset['Fare'] <= 31.6), 'Fare'] = 3

    dataset.loc[dataset['Fare'] > 31.6, 'Fare'] = 4;

    

    dataset['Fare'] = dataset['Fare'].astype(int)
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']



train = train.drop(drop_elements, axis = 1)

test = test.drop(drop_elements, axis = 1)

train.head()
# ヒートマップで相関関係を確認する

colormap = plt.cm.RdBu

plt.figure(figsize=(14, 12))

plt.title('Pearson Correlation of Feature', y=1.05, size=15)



#train.corr() 各データの相関指数をだす関数

sns.heatmap(train.astype(float).corr(), linewidth=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
forest = RandomForestClassifier(random_state=0, n_estimators=500, max_depth=6, min_samples_leaf=2, verbose=0)

forest.fit(X_train, y_train)



gbrt = GradientBoostingClassifier(random_state=0, n_estimators=500, max_depth=8, verbose=0)

gbrt.fit(X_train, y_train)



svm = SVC(kernel='linear', C=0.025)

svm.fit(X_train, y_train)



dtree = DecisionTreeClassifier(random_state=0, max_depth=8)

dtree.fit(X_train, y_train)



ada = AdaBoostClassifier(random_state=0, n_estimators=500, learning_rate=0.8)

ada.fit(X_train, y_train)



ex = ExtraTreesClassifier(random_state=0, n_estimators=500, max_depth=8, min_samples_leaf=2, verbose=0)

ex.fit(X_train, y_train)
forest.score(X_train, y_train)
fm = cross_val_score(forest, X_train, y_train, cv=kf)

fm.mean()
gbrt.score(X_train, y_train)
gm = cross_val_score(gbrt, X_train, y_train, cv=kf)

gm.mean()
svm.score(X_train, y_train)
sm = cross_val_score(svm, X_train, y_train, cv=kf)

sm.mean()
dtree.score(X_train, y_train)
dm = cross_val_score(dtree, X_train, y_train, cv=kf)

dm.mean()
ada.score(X_train, y_train)
adam = cross_val_score(ada, X_train, y_train, cv=kf)

adam.mean()
ex.score(X_train, y_train)
exm = cross_val_score(ex, X_train, y_train, cv=kf)

exm.mean()
# 生存したかどうかのラベル

y_train = train['Survived'].ravel()

train = train.drop(['Survived'], axis=1)

X_train = train.values

X_test = test.values
forest_rf_tr, forest_rf_te = get_oof(forest, X_train, y_train, X_test)

gbrt_rf_tr, gbrt_rf_te = get_oof(gbrt, X_train, y_train, X_test)

svm_rf_tr, svm_rf_te = get_oof(svm, X_train, y_train, X_test)

dtree_rf_tr, dtree_rf_te = get_oof(dtree, X_train, y_train, X_test)

ada_rf_tr, ada_rf_te = get_oof(ada, X_train, y_train, X_test)

ex_rf_tr, ex_rf_te = get_oof(ex, X_train, y_train, X_test)
x_train = np.concatenate(

    (

        forest_rf_tr,

        gbrt_rf_tr,

        svm_rf_tr,

        dtree_rf_tr,

        ada_rf_tr,

        ex_rf_tr

    ),

    axis=1

)



x_test = np.concatenate(

    (

        forest_rf_te,

        gbrt_rf_te,

        svm_rf_te,

        dtree_rf_te,

        ada_rf_te,

        ex_rf_te

    ),

    axis=1

)



xg = xgb.XGBClassifier(

    n_estimators = 2000,

    max_depth = 4,

    min_child_weight = 2,

    gamma = 0.9,

    subsample = 0.8,

    colsample_bytree=0.8,

    objective= 'binary:logistic'

)
xg.fit(x_train, y_train)

xg.score(x_train, y_train)
xg = xgb.XGBClassifier(

    n_estimators = 2000,

    learning_rate =0.5,

    max_depth = 8,

    min_child_weight = 1,

    gamma = 0.4,

    subsample = 0.8,

    colsample_bytree=0.8,

    objective= 'binary:logistic',

    nthread= -1,

    scale_pos_weight = 1

)
xg.fit(x_train, y_train)

xg.score(x_train, y_train)
gb = GradientBoostingClassifier(

    n_estimators = 500,

    max_features= 0.2,

    max_depth= 5,

    min_samples_leaf= 2,

    verbose=0

)
gb.fit(X_train, y_train)

gb.score(X_train, y_train)
score = cross_val_score(gb, X_train, y_train, cv=kf)

score.mean()
predictions = xg.predict(x_test)

StackingSubmission = pd.DataFrame({

    'PassengerID': PassengerId,

    'Survived': predictions

})



StackingSubmission.to_csv('StackingSubmission.csv', index = False)