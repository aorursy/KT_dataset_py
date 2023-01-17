#必要なライブラリをインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
#カラムを確認
train.columns, test.columns
#基本統計量を確認
train.describe(include="all")
test.describe(include="all")
train.info(), test.info()
#Null数確認
pd.isnull(train).sum()
pd.isnull(test).sum()
#Sex別に生存率確認
df = train.pivot_table('PassengerId', 
                  aggfunc='count', 
                  index='Sex', 
                  columns='Survived',
                 margins=True)
df['Survived ratio'] = df[1] / df['All']
df
#SibSp別に生存率確認
#df2 = train.pivot_table('PassengerId', 
                  #aggfunc='count', 
                  #index='SibSp', 
                  #columns='Survived',
                 #margins=True)
#df2['Survived ratio'] = df2[1] / df2['All']
#df2
#fig = plt.figure(figsize=(8, 4))
#ax = fig.add_subplot(1,1,1)
#df2['Survived ratio'].plot(kind='bar', ax=ax)
#ax.legend(bbox_to_anchor=(1, 1))
#plt.show
#Parch別に生存率確認
#df3 = train.pivot_table('PassengerId', 
                  #aggfunc='count', 
                  #index='Parch', 
                  #columns='Survived',
                 #margins=True)
#df3['Survived ratio'] = df3[1] / df3['All']
#df3
#fig = plt.figure(figsize=(8, 4))
#ax = fig.add_subplot(1,1,1)
#df3['Survived ratio'].plot(kind='bar', ax=ax)
#ax.legend(bbox_to_anchor=(1, 1))
#plt.show
#家族連れの数を計算
train['Family'] = train['SibSp'] + train['Parch'] + 1
test['Family'] = test['SibSp'] + test['Parch'] + 1
df2 = train.pivot_table('PassengerId', 
                  aggfunc='count', 
                  index='Family', 
                  columns='Survived',
                 margins=True)
df2['Survived ratio'] = df2[1] / df2['All']

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1,1,1)
df2['Survived ratio'].plot(kind='bar', ax=ax)
ax.legend(bbox_to_anchor=(1, 1))

print(df2, plt.show)
train["IsAlone"] = train.Family.apply(lambda x: 1 if x == 1 else 0)
test["IsAlone"] = test.Family.apply(lambda x: 1 if x == 1 else 0)
df3 = train.pivot_table('PassengerId', 
                  aggfunc='count', 
                  index='IsAlone', 
                  columns='Survived',
                 margins=True)
df3['Survived ratio'] = df3[1] / df3['All']
df3
#集計がやりやすいよう、Ageをグルーピング
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', '1-5', '6-12', '13-18', '19-24', '25-35', '36-60', '60-']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)
#作成したAgeGroupごとに生存率確認
df4 = train.pivot_table('PassengerId', 
                  aggfunc='count', 
                  index='AgeGroup', 
                  columns='Survived',
                 margins=True)
df4['Survived ratio'] = df4[1] / df4['All']

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1,1,1)
df4['Survived ratio'].plot(kind='bar', ax=ax)
ax.legend(bbox_to_anchor=(1, 1))

print(df4, plt.show)
#部屋がある人=1 / 部屋がある人=0となるよう分類
train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))
#部屋があるなしで生存率確認
df5 = train.pivot_table('PassengerId', 
                  aggfunc='count', 
                  index='CabinBool', 
                  columns='Survived',
                 margins=True)
df5['Survived ratio'] = df5[1] / df5['All']
df5
#Cabin内のアルファベットを抽出
train['Cabin'] = train['Cabin'].fillna('Unknown')
train['Deck']=train['Cabin'].str.get(0)
test['Cabin'] = test['Cabin'].fillna('Unknown')
test['Deck']=test['Cabin'].str.get(0)
df6 = train.pivot_table('PassengerId', 
                  aggfunc='count', 
                  index='Deck', 
                  columns='Survived',
                 margins=True)
df6['Survived ratio'] = df6[1] / df6['All']

print(df6, sns.barplot(x="Deck", y="Survived", data=train, palette='Set3'))
#集計のためFareをグルーピング
test["Fare"] = test["Fare"].fillna(-2.0)
bins = [-10, -1, 1, 8, 14, 31, np.inf]
labels = ['Unknown', '0-1', '2-8', '9-14', '15-31', '31-']
train['FareGroup'] = pd.cut(train["Fare"], bins, labels = labels)
test['FareGroup'] = pd.cut(test["Fare"], bins, labels = labels)
#FareGroupごとに生存率確認
df7 = train.pivot_table('PassengerId', 
                  aggfunc='count', 
                  index='FareGroup', 
                  columns='Survived',
                 margins=True)
df7['Survived ratio'] = df7[1] / df7['All']

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1,1,1)
df7['Survived ratio'].plot(kind='bar', ax=ax)
ax.legend(bbox_to_anchor=(1, 1))

print(df7, plt.show)
#Embarkedごとに生存率確認
df8 = train.pivot_table('PassengerId', 
                  aggfunc='count', 
                  index='Embarked', 
                  columns='Survived',
                 margins=True)
df8['Survived ratio'] = df8[1]/ df8['All']

print(df8, sns.barplot(x="Embarked", y="Survived", data=train, palette='Set3'))
#NameからTitleを抽出
combine = [train, test]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
#表記ゆれを修正
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
#Ticket_Lettごとに生存率確認
df9 = train.pivot_table('PassengerId', 
                  aggfunc='count', 
                  index='Title', 
                  columns='Survived',
                 margins=True)
df9['Survived ratio'] = df9[1] / df9['All']
    
print(df9, sns.barplot(x="Title", y="Survived", data=train, palette='Set3'))
#Ticket内のアルファベット、長さを抽出
for dataset in combine: 
        dataset['Ticket_Lett'] = dataset['Ticket'].apply(lambda x: str(x)[0])
        dataset['Ticket_Lett'] = dataset['Ticket_Lett'].apply(lambda x: str(x)) 
        dataset['Ticket_Lett'] = np.where((dataset['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), dataset['Ticket_Lett'], np.where((dataset['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0')) 
        dataset['Ticket_Len'] = dataset['Ticket'].apply(lambda x: len(x)) 
#Ticket_Lettごとに生存率確認
df10 = train.pivot_table('PassengerId', 
                  aggfunc='count', 
                  index='Ticket_Lett', 
                  columns='Survived',
                 margins=True)
df10['Survived ratio'] = df10[1] / df10['All']

print(df9, sns.barplot(x="Ticket_Lett", y="Survived", data=train, palette='Set3'))
#Ticket_Letnごとに生存率確認
df11 = train.pivot_table('PassengerId', 
                  aggfunc='count', 
                  index='Ticket_Len', 
                  columns='Survived',
                 margins=True)
df11['Survived ratio'] = df11[1] / df11['All']

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1,1,1)
df11['Survived ratio'].plot(kind='bar', ax=ax)
ax.legend(bbox_to_anchor=(1, 1))

print(df11, plt.show)
train.info(), test.info()
#不要なカラムを削除
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)

train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)

train = train.drop(['SibSp'], axis = 1)
test = test.drop(['SibSp'], axis = 1)

train = train.drop(['Parch'], axis = 1)
test = test.drop(['Parch'], axis = 1)

train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)
#EmbarkedのNullをSで埋める
train = train.fillna({"Embarked": "S"})

#欠損値をFare=中央値, Age=平均値で埋める
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())
#カテゴリカル変数のエンコード
from sklearn import preprocessing

for column in ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 'Deck', 'Ticket_Len', 'Ticket_Lett']:
    le = preprocessing.LabelEncoder()
    le.fit(train[column])
    train[column] = le.transform(train[column])
    
for column in ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 'Deck', 'Ticket_Len', 'Ticket_Lett']:
    le = preprocessing.LabelEncoder()
    le.fit(test[column])
    test[column] = le.transform(test[column])
#各変数の相関係数を確認
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
train.columns, test.columns
#Validation用
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Modeling用
import xgboost as xgb
X = train.drop(['Survived', 'PassengerId'], axis=1)
y = train["Survived"]
#学習データを検証用にスプリット
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Xgboost用のMatrix形式にデータを変換
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)

#Hyper Parameterを指定。（本当は工夫すべきだが今回はほぼデフォルト値）
xgb_params = {'max_depth':3, 
              'learning_rate': 0.1, 
              'objective':'binary:logistic',
              'eval_metric': 'logloss'}


# 学習時に用いる検証用データ
evals = [(dtrain, 'train'), (dtest, 'eval')]

# 学習過程を記録するための辞書
evals_result = {}
clf = xgb.train(xgb_params,
                dtrain,
                num_boost_round=1000,
                early_stopping_rounds=100,
                evals=evals,
                evals_result=evals_result,
                )

#検証用データでモデルのAccracyを確認
y_pred_proba = clf.predict(dtest)
y_pred = np.where(y_pred_proba > 0.5, 1, 0)
acc = accuracy_score(y_test, y_pred)
print('Accuracy:', acc)
# 学習の過程を折れ線グラフとしてプロット
train_metric = evals_result['train']['logloss']
plt.plot(train_metric, label='train logloss')
eval_metric = evals_result['eval']['logloss']
plt.plot(eval_metric, label='eval logloss')
plt.grid()
plt.legend()
plt.xlabel('rounds')
plt.ylabel('logloss')
#モデルのFeature Importanceを確認
_, ax = plt.subplots(figsize=(12, 4))
xgb.plot_importance(clf,
                    ax=ax,
                    importance_type='gain',
                    show_values=False)
plt.show()
#Testデータで予測
target = xgb.DMatrix(test.drop('PassengerId', axis=1))
xgb_pred = clf.predict(target, ntree_limit=clf.best_ntree_limit)
#Submit用のデータに変換
test["Survived"] = np.where(xgb_pred > 0.5, 1, 0)
test[["PassengerId","Survived"]].to_csv(('submit.csv'),index=False)
