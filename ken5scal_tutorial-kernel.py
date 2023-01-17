# Kaggleのススメかた
## トレインデータ、テストデータの統計を確認 
## 適当に特徴量を追加-> 例: Sibpar(兄弟・パートナー)とParch(親子共)から家族カラムの追加、Fareから運賃分布を区切る、NameのMr.Mrs/Miss/Captain/Col/Dr...
## 前処理を頑張る（新しい特徴量の追加、データの正規化）
## 適当なアルゴリズムを書く 
## 動くことを確認する 
## トレインデータ、テストデータの統計を確認 
## 改善を確認する 
## 繰り返す

# 前処理
## 欠損値(Null)の補正
## 外れ値の検出と処理
## ダミー変数の作成
## 連続データの離散化
## 特徴量選択
## 入力データの余計なデータ正規化？
## 入力データの一般的なテータ型を推測する？ -> 例: Tickets

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Grasp Column Info
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_gender_submission = pd.read_csv('../input/gender_submission.csv')
df_train.head(5)
#  Read Frame
print('-'*40)
print('-'*10, " Shapes ", '-'*10)
print('-'*40)
print('submission data: ', df_gender_submission.shape)

print('-'*40)
print('-'*10, " Train Data Info ", '-'*10)
print('-'*40)
print('train data: ', df_train.shape)
print(df_train.columns)
df_train.info()
print(df_train.isnull().sum())

print('-'*40)
print('-'*10, " Test Data Info ", '-'*10)
print('-'*40)
print('test data: ', df_test.shape)
print(df_test.columns)
df_test.info()

print('-'*40)
print('-'*10, " Stats  ", '-'*10)
print('-'*40)
df_full = pd.concat([df_train, df_test], axis = 0, ignore_index=True)
print(df_full.shape)
df_full.describe() # Summary  Statistics, df_full.describe(include='all') displays all types of data. By default, it only displays integer type stats
# Checking stats
plt.subplot(1, 2, 1)
sns.countplot(x='Survived', hue='Sex', data=df_train)
plt.title('Survival rate by Sex')
plt.xticks([0,1],['Dead', "Alive"])
plt.ylim(0, 500)
df_train[['Survived','Sex']].groupby(['Sex']).mean()

plt.subplot(1, 2, 2)
sns.countplot(x='Survived', hue='Pclass', data=df_train)
plt.title('Survival rate  by Ticket Class')
plt.xticks([0,1],['Dead', "Alive"])
plt.ylim(0, 500)
df_train[['Survived','Pclass']].groupby(['Pclass'],  as_index=False).mean()
# Checking stats
sns.distplot(df_train['Age'].dropna(), kde=False, bins=30, label='Whole')
sns.distplot(df_train[df_train['Survived'] == 0].Age.dropna(), kde=False, bins=30, label='Dead')
sns.distplot(df_train[df_train['Survived'] == 1].Age.dropna(), kde=False, bins=30, label='Alive')
plt.title('Histogram of Age')
plt.legend()

df_train['CategoricalAge'] = pd.cut(df_train['Age'], 8)
df_train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean()
# Checking stats
plt.subplot(1, 2, 1)
sns.countplot(x='SibSp', data = df_train)
plt.title('# of siblings/partner')
df_train['SibSp_0_1_2over'] = [i if i <= 1 else 2 for i in df_train['SibSp']]

plt.subplot(1, 2, 2)
sns.countplot(x='Survived', hue='SibSp_0_1_2over', data=df_train)
plt.legend(['0','1','2 < '],loc='upper right')
plt.title('Survival rate by #of sib/partner')
df_train[['SibSp_0_1_2over', 'Survived']].groupby(['SibSp_0_1_2over']).mean()
# Checking stats
plt.subplot(1, 2, 1)
sns.countplot(x='Parch', data = df_train)
plt.title('# of parents/children')

plt.subplot(1, 2, 2)
df_train['Parch_0_1_2_3over'] = [i if i <= 2 else 3 for i in df_train['Parch']]
sns.countplot(x='Survived', hue='Parch_0_1_2_3over', data=df_train)
plt.legend(['0','1','2,','3 < '],loc='upper right')
plt.title('Survival rate by #of Parents/Children')
df_train[['Parch_0_1_2_3over', 'Survived']].groupby(['Parch_0_1_2_3over']).mean()
# Adding New Fature 1
# SibSpとParchの統計から、１人で乗船している場合の生存率が低い事が判明した。
# そこから家族数の統計を作成する
## SibSpとParchが同上している家族の数。１を足すと家族の人数になる。
df_train['FamilySize']=df_train['SibSp'] + df_train['Parch']+1 

df_train['IsAlone'] = 0
df_train.loc[df_train['FamilySize'] == 1, 'IsAlone' ] = 1
sns.countplot(x='Survived', hue='IsAlone', data=df_train)
plt.xticks([0,1],['Dead',"Alive"])
plt.legend(['Not Alone','Alone'],loc='upper right')
plt.title('Survival rate of being Alone')
# Adding New Feature 2
sns.distplot(df_train['Fare'].dropna(), kde=False, hist=True) #kde = kernel density distribution
plt.title('Distribution of Fare')

df_train['CategoricalFare'] = pd.cut(df_train['Fare'], 4)
df_train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean()
# Extract new fature(prefix)
set(df_train.Name.str.extract('([A-Za-z]+)\.', expand=False))
import collections

df_train["Title"] = df_train.Name.str.extract('([A-Za-z]+)\.', expand=False)
df_test["Title"] = df_test.Name.str.extract('([A-Za-z]+)\.', expand=False)
df_train.groupby('Title').mean()['Age'] # Master's average age is 4.6 meaning it's Child. Age can influence survival rate, so this can be an useful feature.

def title_to_num(title):
    if title == 'Master':
        return 1
    elif title == 'Miss':
        return 2
    elif title == 'Mr':
        return 3
    elif title == 'Mrs':
        return 4
    else:
        return 5

df_train['Title_num'] = [title_to_num(i) for i in df_train['Title']]
df_test['Title_num'] = [title_to_num(i) for i in df_test['Title']]
                    
# To Improve Score
## Drop not important feature
## Utilize following Features
### Use Title_num generated from Name
### Check order of embarked and check survival rate, guess why there are differences
### Research how `cabin` works and generate the feature
## Guess and change Random Forest's hyper parameter. Use Grid Search
## Check other kernels using random forest
## Use different algorithm

# 再ロード
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

# 前処理
## Ageの欠損値をPclassごとの平均値で保管 
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 39 # Pclass 1 の年齢の中央値
        elif Pclass == 2:
            return 30
        else:
            return 25
    else:
        return Age
        
df_train['Age'] = df_train[['Age','Pclass']].apply(impute_age, axis=1)
df_test['Age'] = df_test[['Age','Pclass']].apply(impute_age, axis=1)

df_train["Title"] = df_train.Name.str.extract('([A-Za-z]+)\.', expand=False)
df_test["Title"] = df_test.Name.str.extract('([A-Za-z]+)\.', expand=False)
def title_to_num(title):
    if title == 'Master':
        return 1
    elif title == 'Miss':
        return 2
    elif title == 'Mr':
        return 3
    elif title == 'Mrs':
        return 4
    else:
        return 5

df_train['Title_num'] = [title_to_num(i) for i in df_train['Title']]
df_test['Title_num'] = [title_to_num(i) for i in df_test['Title']]

## 乗船港の欠損値の補完: 
### 欠損データ
df_train[df_train['Embarked'].isnull()]
### 特に出掛かりになる情報がないので、乗船者が多い乗船港「C」で補完
df_train.loc[df_train['PassengerId'].isin([62,830]), 'Embarked']='C'

## 運賃の欠損値を平均値で補完
### 乗客クラスごとの運賃の平均値
df_train[['Pclass','Fare']].groupby('Pclass').mean()
### 欠損値。Pclassが3なので、↑で求めた運賃の平均値を適用
df_test[df_test['Fare'].isnull()]
df_test.loc[df_test['PassengerId'] == 1044, 'Fare'] = 13.675550

print('-'*40)
print('-'*10, " df_trainの欠損値  ", '-'*10)
print('-'*40)
print(df_train.isnull().sum())

print('-'*40)
print('-'*10, " df_testの欠損値  ", '-'*10)
print('-'*40)
print(df_test.isnull().sum())

genders = {'male':0, 'female':1}
df_train['Sex'] = df_train['Sex'].map(genders)
df_test['Sex'] = df_test['Sex'].map(genders)

##  ダミー変数化
# df_train = pd.get_dummies(df_train, columns=['Embarked'])
# df_test = pd.get_dummies(df_test, columns=['Embarked'])
df_train['Embarked'] = df_train['Embarked'].map({'S':0, 'C':1, 'Q': 2}).astype(int)
df_test['Embarked'] = df_test['Embarked'].map({'S':0, 'C':1, 'Q': 2}).astype(int)

## 連続変数の離散化（Age, Fare)
data = [df_train, df_test]
for df in data:
    df.loc[df['Fare'] <=7.91, 'Fare'] = 0
    df.loc[ (df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[ (df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)

    df.loc[df['Age'] <=16, 'Age'] = 0
    df.loc[ (df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[ (df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[ df['Age'] > 48, 'Age'] = 3
    df['Age'] = df['Age'].astype(int)
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone' ] = 1
    
## 不要な列の削除
df_train.drop(['Name','Cabin','Ticket', 'SibSp', 'Parch', 'IsAlone','Title','Embarked','Age'], axis=1, inplace=True)
df_test.drop(['Name','Cabin','Ticket', 'SibSp', 'Parch', 'IsAlone','Title','Embarked','Age'], axis=1, inplace=True)
    
# Random FOrest
X_train = df_train.drop(['PassengerId', 'Survived'], axis=1)
Y_train = df_train['Survived']
X_test = df_test.drop('PassengerId',axis=1).copy()

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=1)

# Learn to predict Y_train from X_train (Over Estimation - it learns a model from X-train and Y_train and use it to predict Y_train from X_train)
rf.fit(X_train, Y_train)
acc_log = round(rf.score(X_train, Y_train) * 100, 2)
print(round(acc_log,2,), '%')

# Cross Validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score 

## Splitting data in 3 parts. Use 2 as training data, and 1 as testing data.
### Repeat  3 times so that individual data becomes testing data for at least 1 time.
skf = StratifiedKFold(n_splits=3)
for train_idx, test_idx in skf.split(X_train, Y_train):
    X_cv_train = X_train.iloc[train_idx]
    Y_cv_train = Y_train.iloc[train_idx]
    
    X_cv_test = X_train.iloc[test_idx]
    Y_cv_test = Y_train.iloc[test_idx]
    
    # Learning
    rf.fit(X_cv_train, Y_cv_train)
    
    # Predicting
    pred = rf.predict(X_cv_test)
    print(round(accuracy_score(Y_cv_test,  pred)*100, 2))
    
# Now use this model to make submission data
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)
submission = pd.DataFrame({'PassengerId': df_test['PassengerId'],'Survived': Y_pred})
submission.to_csv('submission.csv', index=False)


print('-'*40)
print('-'*10, " Importance Feature", '-'*10)
print('-'*40)
for i,k in zip(X_train.columns, rf.feature_importances_):
    print(i,round(k,4))

X_train.head()
