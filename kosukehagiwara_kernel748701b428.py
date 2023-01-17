# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
sns.set(style='darkgrid')


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#各種データの取り込み
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
#どんなデータがあるのか、欠損値の有無を確認
print(train.info())
print()
print(test.info())
train.head(10)
test.head(10)
test_PassengerId = test['PassengerId']
test_PassengerId
#欠損値をカウント、パーセンテージを出力する関数
def count_missing_rate(df):
    count = 0
    for column in df.columns:
        total = df[column].isnull().sum()#欠損値のカウント
        percent = round(total/len(df[column])*100,2)#データ数に対する欠損値の割合
        if count == 0:
            df1 = pd.DataFrame([[total,percent]], columns=['total', 'percent'], index=[column])
            count+=1
        else:#作成したカラム毎のDataFrameを結合
            df2 = pd.DataFrame([[total, percent]], columns=['total', 'percent'], index=[column])
            df1 = pd.concat([df1, df2], axis=0)
            count+=1
    return df1
count_missing_rate(train)
count_missing_rate(test)
#Survivedを追加し欠損値で埋める
test['Survived'] = np.nan

#trainデータとtestデータの結合
df = pd.concat([train, test], ignore_index = True, sort = False)

#結合したデータの確認
print(df.info())
print()
df.head(10)
count_missing_rate(df)
#性別と生存率の関係
sns.barplot(x = 'Sex', y = 'Survived', data = df, palette = 'Set3')
plt.show()
from sklearn.ensemble import RandomForestRegressor
#推定に使用する項目を指定
age_df = df[['Age', 'Pclass', 'Sex', 'Parch', 'SibSp']]

#ラベル特徴量をワンホットエンコーディング
age_df = pd.get_dummies(age_df)#

#学習データとテストデータに分類し、numpyに変換する
known_age = age_df[age_df.Age.notnull()].values
unknown_age = age_df[age_df.Age.isnull()].values

X = known_age[:, 1:]
y = known_age[:, 0]

#ランダムフォレストで推定モデルを構築する
rfr = RandomForestRegressor(random_state = 0, n_estimators = 100, n_jobs = -1)
rfr.fit(X, y)

#推定モデルを使ってテストデータのAgeを予測し補完する
pred_age = rfr.predict(unknown_age[:, 1::])
df.loc[(df.Age.isnull()), 'Age'] = pred_age
#Ageの欠損値が全て埋まっているか確認する
print(df.info())
df.head(10)
df['Title'] = df['Name'].map(lambda x: x.split(', ')[1].split('.')[0])
df['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
df['Title'].replace(['Don', 'Sir', 'the Countess', 'Lady', 'Dona'], 'Royalty', inplace=True)
df['Title'].replace(['Mme', 'Ms'], 'Miss', inplace=True)
df['Title'].replace(['Mlle'], 'Miss', inplace=True)
df['Title'].replace(['Jonkheer'], 'Master', inplace=True)
sns.barplot(x='Title', y='Survived', data=df, palette='Set3')
#一緒に乗船していた兄弟、配偶者の数 + 一緒に乗船していた親、子供、孫の数 + 本人
df['Family_size'] = df['SibSp'] + df['Parch'] + 1 
#同じTicketナンバーの人をグルーピングし生存率をグラフで確認する
Ticket_Count = dict(df['Ticket'].value_counts())
df['Ticket_Group'] = df['Ticket'].map(Ticket_Count)
sns.barplot(x='Ticket_Group', y='Survived', data=df, palette='Set3')
plt.show()
#生存率に基づいて３つのグループに分ける
df.loc[(df['Ticket_Group']>=2) & (df['Ticket_Group']<=4), 'Ticket_label'] = 2
df.loc[(df['Ticket_Group']>=5) & (df['Ticket_Group']<=8) | (df['Ticket_Group']==1), 'Ticket_label'] = 1
df.loc[(df['Ticket_Group']>=11), 'Ticket_label'] = 0
sns.barplot(x='Ticket_label', y='Survived', data=df, palette='Set3')
plt.show()
#Embarkedの欠損値を最頻値で埋める
df['Embarked']= df['Embarked'].fillna(df['Embarked'].mode().iloc[0])

#Cabinを削除する
df = df.drop('Cabin', axis=1)
df.head(10)
#インデックスの取得
index = df[df['Fare'].isnull()].index

#どのようなデータか表示
df[df['Fare'].isnull()]
#Fareの算出
fare = df.loc[(df['Embarked']== 'S') & (df['Pclass'] == 3), 'Fare'].median()

#欠損値を埋める
df['Fare'] = df['Fare'].fillna(fare)
#dfの中身を確認
df.info()
df = df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Ticket_Group'], axis=1)
df = pd.get_dummies(df)
df
train = df[df['Survived'].notnull()]
test = df[df['Survived'].isnull()].drop('Survived', axis=1)
train.head(10)
#trainデータの統計量を確認
train.describe()
plt.figure(figsize=(5,7.5))
sns.boxplot(data=train, y='Fare')#乗車料金の箱ひげ図
sns.distplot(train.Fare)#乗車料金のヒストグラム
drop_train = train.dropna(subset=['Age'])#年齢のヒストグラム
sns.distplot(drop_train.Age)
#"Fare"の外れ値のインデックスを取得
drop = train.index[train['Fare']>(train.describe().at['75%', 'Fare']+train.describe().at['75%', 'Fare']-train.describe().at['25%','Fare']*1.5)]

#外れ値をドロップして再度trainに入れる
train = train.drop(drop)
train
train.describe()
train = train.drop('Sex_male',axis=1)
test = test.drop('Sex_male',axis=1)
train_set, test_set = train_test_split(train, test_size = 0.3, random_state = 0)

X_train = train_set.iloc[:,1:] #全ての行の２列目以降を説明変数とする
y_train = train_set.iloc[:, 0] #全ての行の１列目（Survived）を目的変数とする

X_test = test_set.iloc[:,1:] #同様
y_test = test_set.iloc[:, 0] #同様

def objective(trial):
    min_samples_split = trial.suggest_int('min_samples_split', 8, 16)
    max_leaf_nodes = int(trial.suggest_discrete_uniform('max_leaf_nodes', 4, 64, 4))
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    #ランダムフォレストでモデルの作成
    Rf = RandomForestClassifier(min_samples_split = min_samples_split,
                                max_leaf_nodes = max_leaf_nodes,
                                criterion = criterion,
                                n_estimators = 100,
                                random_state=0)
    Rf.fit(X_train, y_train)
    return 1.0 - accuracy_score(y_test, Rf.predict(X_test))
import warnings
warnings.filterwarnings('ignore')

study = optuna.create_study()
study.optimize(objective, n_trials = 100)

print(study.best_params)
print()
print(1.0 - study.best_value)
import xgboost as xgb#xgboostをインポート
from xgboost import XGBClassifier

# Objective Functionの作成
def opt(trial):
    n_estimators = trial.suggest_int('n_estimators', 0, 1000)
    max_depth = trial.suggest_int('max_depth', 1, 20)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
    subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)
    colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)
    Xgb = XGBClassifier(random_state=0,
                        n_estimators = n_estimators,
                        max_depth = max_depth,
                        min_child_weight = min_child_weight,
                        subsample = subsample,
                        colsample_bytree = colsample_bytree)
    Xgb.fit(X_train,y_train)
    return 1.0 - accuracy_score(y_test, Xgb.predict(X_test))
#XGBの探索
study = optuna.create_study()
study.optimize(objective, n_trials = 100)

print(study.best_params)
print()
print(1.0 - study.best_value)
import lightgbm as lgb#lightgbmをインポート
from lightgbm import LGBMClassifier

# Objective Functionの作成
def opt(trial):
    n_estimators = trial.suggest_int('n_estimators', 0, 1000)
    max_depth = trial.suggest_int('max_depth', 1, 20)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
    subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)
    colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)
    Lgbm = LGBMClassifier(random_state=0,
                        n_estimators = n_estimators,
                        max_depth = max_depth,
                        min_child_weight = min_child_weight,
                        subsample = subsample,
                        colsample_bytree = colsample_bytree)
    Lgbm.fit(X_train,y_train)
    return 1.0 - accuracy_score(y_test, Lgbm.predict(X_test))
#Lgbmの探索
study = optuna.create_study()
study.optimize(objective, n_trials = 100)

print(study.best_params)
print()
print(1.0 - study.best_value)
#ランダムフォレストでモデルの作成
Rf = RandomForestClassifier(min_samples_split=8,
                            max_leaf_nodes=60,
                            criterion='gini',
                            n_estimators = 100,
                            random_state=0)

#XGBMでモデルの作成
Xgb = XGBClassifier(min_samples_split=8,
                    max_leaf_nodes=56,
                    criterion='entropy',
                    n_estimators = 100,
                    random_state=0)

#LGBMでモデルの作成
Lgb = LGBMClassifier(min_samples_split=8,
                     max_leaf_nodes=52,
                     criterion='entropy',
                     n_estimators = 100,
                     random_state=0)

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2

k_range = np.arange(1,16)

scores_rf = []
scores_xg = []
scores_lg = []

std_rf = []
std_xg = []
std_lg = []

train_X = train.iloc[:, 1:].values#trainデータ全ての行の２列目以降説明変数とする
train_y = train.iloc[:, 0].values#trainデータ全ての行の１列目（Survived）を目的変数とする

count = 0

for k in k_range:
    
    ss = ShuffleSplit(n_splits=10,
                  train_size=0.8,
                  test_size=0.2, 
                  random_state=0)
    score_rf = []
    score_xg = []
    score_lg = []

    for train_index, test_index in ss.split(train_X, train_y):
    
        count+=1
    
        X_train, X_test = train_X[train_index], train_X[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        
        skb = SelectKBest(chi2, k=k) 
        skb.fit(X_train, y_train)
        
        X_new_train = skb.transform(X_train)
        X_new_test = skb.transform(X_test)
    
        Rf.fit(X_new_train, y_train)
        Xgb.fit(X_new_train, y_train)
        Lgb.fit(X_new_train, y_train)
    
        score_rf.append(Rf.score(X_new_test, y_test))
        score_xg.append(Xgb.score(X_new_test, y_test))
        score_lg.append(Lgb.score(X_new_test, y_test))
        
    scores_rf.append(np.array(score_rf).mean())
    scores_xg.append(np.array(score_xg).mean())
    scores_lg.append(np.array(score_lg).mean())
        
    std_rf.append(np.array(score_rf).std())
    std_xg.append(np.array(score_xg).std())
    std_lg.append(np.array(score_lg).std())        

scores_rf = np.array(scores_rf)
scores_xg = np.array(scores_xg)
scores_lg = np.array(scores_lg)

std_rf = np.array(std_rf)
std_xg = np.array(std_xg)
std_lg = np.array(std_lg)
plt.plot(k_range, scores_rf)
plt.errorbar(k_range, scores_rf, yerr=std_rf)
plt.ylabel('rf_accuracy')
plt.plot(k_range, scores_xg)
plt.errorbar(k_range, scores_xg, yerr=std_xg)
plt.ylabel('xg_accuracy')
plt.plot(k_range, scores_lg)
plt.errorbar(k_range, scores_lg, yerr=std_lg)
plt.ylabel('lg_accuracy')
best_k_rf = k_range[np.argmax(scores_rf)]
best_k_xg = k_range[np.argmax(scores_xg)]
best_k_lg = k_range[np.argmax(scores_lg)]

print('RF_Kbest: {}'.format(best_k_rf))
print('XGB_Kbest: {}'.format(best_k_xg))
print('LGBM_Kbest: {}'.format(best_k_lg))
#ランダムフォレストでモデルの作成
Rf = RandomForestClassifier(min_samples_split=8,
                            max_leaf_nodes=60,
                            criterion='entropy',
                            n_estimators = 100,
                            random_state=0)

#XGBMでモデルの作成
Xgb = XGBClassifier(min_samples_split=8,
                    max_leaf_nodes=56,
                    criterion='entropy',
                    n_estimators = 100,
                    random_state=0)

#LGBMでモデルの作成
Lgb = LGBMClassifier(min_samples_split=8,
                     max_leaf_nodes=52,
                     criterion='entropy',
                     n_estimators = 100,
                     random_state=0)

#トレーニングデータとテストデータに分ける
from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=10,
                  train_size=0.8,
                  test_size=0.2, 
                  random_state=0)

train_X = train.iloc[:, 1:].values#trainデータ全ての行の２列目以降説明変数とする
train_y = train.iloc[:, 0].values#trainデータ全ての行の１列目（Survived）を目的変数とする

count = 0
rf = []
xg = []
lg = []

for train_index, test_index in ss.split(train_X, train_y):
    
    count+=1
    
    X_train, X_test = train_X[train_index], train_X[test_index]
    y_train, y_test = train_y[train_index], train_y[test_index]
    
    #先ほど求めたKbestを使って次元を下げる
    skb_rf = SelectKBest(chi2, k=best_k_rf)
    skb_xg = SelectKBest(chi2, k=best_k_xg)
    skb_lg = SelectKBest(chi2, k=best_k_lg)
    
    #次元を下げるために学習させる
    skb_rf.fit(X_train, y_train)
    skb_xg.fit(X_train, y_train)
    skb_lg.fit(X_train, y_train)
    
    #トレーニングデータの次元を下げる
    X_train_best_rf = skb_rf.transform(X_train)
    X_train_best_xg = skb_xg.transform(X_train)
    X_train_best_lg = skb_lg.transform(X_train)
    
    #テストデータの次元を下げる
    X_test_best_rf = skb_rf.transform(X_test)
    X_test_best_xg = skb_xg.transform(X_test)
    X_test_best_lg = skb_lg.transform(X_test)
    
    #各モデルに次元を下げたデータで学習させる
    Rf.fit(X_train_best_rf, y_train)
    Xgb.fit(X_train_best_xg, y_train)
    Lgb.fit(X_train_best_lg, y_train)
    
    print('{}回目'.format(count))
    print('ランダムフォレストのスコア ： ', Rf.score(X_test_best_rf, y_test))
    print('xgboostのスコア ： ', Xgb.score(X_test_best_xg, y_test))
    print('lightgbmのスコア ： ', Lgb.score(X_test_best_lg, y_test))
    print()
    
    rf.append(Rf.score(X_test_best_rf, y_test))
    xg.append(Xgb.score(X_test_best_xg, y_test))
    lg.append(Lgb.score(X_test_best_lg, y_test))

print('ランダムフォレストのスコア平均 ： ', np.mean(rf))
print('xgboostのスコア平均 ： ', np.mean(xg))
print('lightgbmのスコア平均 ： ', np.mean(lg))
test_X = test.iloc[:, :].values#testデータ全ての行の２列目以降説明変数とする
#test_y = test.iloc[:, 0].values#testデータ全ての行の１列目（Survived）を目的変数とする

test_X_best_rf = skb_rf.transform(test_X)

Rf_pred = Rf.predict(test_X_best_rf)#分割したテストデータの予測
Rf_pred
solution = pd.DataFrame(Lgbm_pred.astype(int), test_PassengerId, columns = ['Survived']) #最初に取得したtest_PassengerIdを使用する
solution
solution.to_csv('solution.csv', index_label = ['PassengerId'])
