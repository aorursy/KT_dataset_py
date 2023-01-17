# 参考サイト
# https://qiita.com/silva0215/items/79fffc7a4185c9222e14

# 使用するライブラリのインポート
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb

# 使用するモデルのインポート
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

PassengerId = test['PassengerId']
full_data = [train, test]

# データの整理・整形、新しい特徴量の追加
# 乗客の名前の長さを追加
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
# 客室ありの乗客かどうかを追加
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# 同乗した兄弟、配偶者の人数（SibSp）と同乗した親、子供の人数（Parch）からファミリーサイズを追加する
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# ファミリーサイズから1名の乗客かどうかを追加する
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# 乗船した港（Embarked）に欠損値があるため、Southamptonで補完する
# C = Cherbourg, Q = Queenstown, S = Southampton
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# 運賃（Fare）に欠損値があるため、中央値で補完する
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

# 年齢（Age）に欠損値があるため、平均と標準偏差から乱数を生成し、補完する
for dataset in full_data:
    # 年齢の平均
    age_avg = dataset['Age'].mean()
    # 年齢の標準偏差
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

# 乗客の名前から敬称（MrやMsなど）を抽出してくる関数を定義する
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

# 乗客の名前（Name）から敬称を抽出し、追加する
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# 敬称（Title）をグルーピングしたもので書き換える
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# データの整形
for dataset in full_data:
    # 性別（Sex）を定数化する
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # 敬称（Title）を定数化する
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    # 敬称（Title）がないレコードを0埋めする
    dataset['Title'] = dataset['Title'].fillna(0)

    # 乗船した港（Embarked）を定数化する
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    # 運賃（Fare）を定数化する
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']                               = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare']                                  = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # 年齢（Age）を定数化する
    dataset.loc[ dataset['Age'] <= 16, 'Age']                          = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;

# 特徴量の選択（不要なカラムを落とす）
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)

# 学習と予測
ntrain = train.shape[0]
ntest = test.shape[0]

# 交差検証の回数
NFOLDS = 5
# 同じ乱数を発生させるための固定値
SEED = 0
# パラメータの検証、学習モデルの精度を評価
# 交差検証（クロスバリデーション）でパラメータを検証し、精度がよく、過学習が起こらないパラメータを決定する
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# Sklearn classifier を拡張
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self,x,y):
        return self.clf.fit(x,y)

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# モデルのパラメータ定義
# Random Forest のパラメータ
rf_params = { 'n_jobs': -1, 'n_estimators': 500, 'warm_start': True, 'max_depth': 6, 'min_samples_leaf': 2, 'max_features' : 'sqrt', 'verbose': 0 }
# Extra Trees のパラメータ
et_params = { 'n_jobs': -1, 'n_estimators':500, 'max_depth': 8, 'min_samples_leaf': 2, 'verbose': 0 }
# AdaBoost のパラメータ
ada_params = { 'n_estimators': 500, 'learning_rate' : 0.75 }
# Gradient Boosting のパラメータ
gb_params = { 'n_estimators': 500, 'max_depth': 5, 'min_samples_leaf': 2, 'verbose': 0 }
# Support Vector Classifier のパラメータ 
svc_params = { 'kernel' : 'linear', 'C' : 0.025 }

# モデルのオブジェクト生成
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# 学習データの生存（Survived）データ、学習データ、テストデータで配列を作成する
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values
x_test = test.values

# 第1段階の学習と予測を実行する
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees Classifier
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest Classifier
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost Classifier
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost Classifier
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier


# 第2段階の学習と予測を実行する
gbm = xgb.XGBClassifier(n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test)

# 予測結果をCSV出力
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)

