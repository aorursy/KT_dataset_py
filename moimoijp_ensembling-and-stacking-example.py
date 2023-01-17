# Load in our libraries

import pandas as pd

import numpy as np

import re

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC

from sklearn.model_selection import KFold
# Load in the train and test datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Store our passenger ID for easy access

PassengerId = test['PassengerId']



train.head(4)
full_data = [train, test]
# Some features of my own that I have added in

# Gives the length of the name

train['Name_length'] = train['Name'].apply(len) # pandasの行列における全項目に対して関数を適用する際はapplyを使う

test['Name_length'] = test['Name'].apply(len) # この場合はlenで文字列からその長さに変換することで数値化し、計算可能なものに変数変換している

# 客室を持っているかどうかを示すダミー変数

# lambdaは無名関数。xを引数にとり、xが浮動小数点数であれば0でそれ以外は1

# Cabinには、客室を所有していると客室番号の文字列が格納されている。

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
# Feature engineering steps taken from Sina

# Create new feature FamilySize as a combination of SibSp and Parch

# SibSp(siblings兄弟 and spouses配偶者)とParch(parents両親 and children子供)から

# FamilySize家族人数を新たに作り出す

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 # +1は自分
# Create new feature IsAlone from FamilySize

# 独身かどうかを示すダミー変数を作る

# おそらく、独身だと守る人がいないので、生き残ることに専念できる。

# そういう考えから、この特徴量を生成していると思われる。

# この課題では、生き残ることを予測する際、生存に起因する要素を現存データからいかに作り出すか、が腕の見せ所

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column

# fillna('S')は欠損値をSで埋める。欠損値とは、データがない状態のこと

# Embarkedは乗船港の頭文字 C = Cherbourg, Q = Queenstown, S = Southampton

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Remove all NULLS in the Fare column and create a new feature CategoricalFare

# 欠損値に対して、median中央値で埋めている。このテクニックはよく使われる。

# 平均値で埋めることもあるが、明らかに２値に分かれている分布において平均値を用いると不自然な値が導出されるため、

# 中央値で埋めるのが無難

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

# pd.qcutはビニング処理と呼ばれる。これは、第１引数のデータを第２引数の数だけ分割を行う処理。

# 今回は4を指定しているので、四分位数毎に分割が行われている。箱ひげ図で見られる分け方。

# つまり、Fare運賃が、安い、ちょっと安い、ちょっと高い、高い、の４つのグループに分けており、

# この特徴量によって一般客かセレブかの線引きができそう。

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

# Create a New feature CategoricalAge

for dataset in full_data:

    age_avg = dataset['Age'].mean() # 平均値

    age_std = dataset['Age'].std() # 標準偏差

    age_null_count = dataset['Age'].isnull().sum() # 欠損値の合計（データを見てみると案外欠損値が多い）

    # 平均から上下の標準偏差の区間において一様乱数を発生させる。

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    # 欠損値を先ほどの乱数で埋める（すごく独特な書き方...）

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int) # int変換（いらないと思うけど念のためかな？）

# pd.cutは第１引数のデータを第２引数の区間ごとに分割を行う。

# pd.qcutは分割数を、pd.cutは分割範囲を指定する。

train['CategoricalAge'] = pd.cut(train['Age'], 5)
# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

# 敬称を取得

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

    print(dataset['Title'])


# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    # よくわからん敬称は全部Rareにする

    # また、同じ意味と思われる敬称は呼び方を統一する

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



# それぞれをダミー変数化

# ちなみに、今回のようにカラム名が意味のある名前になっており、人間が特徴量エンジニアリングしやすい状況にあればよいが、

# コンペの中にはそれぞれのカラム名が何を意味しているのかが伏せられている場合がある。

# そういうときはそういうときで進め方があるので、またそのときに紹介する。

for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} )

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
# Feature selection 特徴量選択：不必要な特徴量を取捨選択すること

# 以下の要素は、既に別の特徴量として変換してあるのでいらない

# 残しておくとエラーや精度悪化の原因になる場合もある

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)
train.head(3)
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',

       u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

g.set(xticklabels=[])
# Some useful parameters which will come in handy later on

# shapeは頻出。ベクトルの次元を取得する。3*4*5のベクトルの場合、shape[0]で3を取得する。

ntrain = train.shape[0] # 891

ntest = test.shape[0] # 418

SEED = 0 # for reproducibility



# Class to extend the Sklearn classifier

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

    

    def feature_importances(self,x,y):

        print(self.clf.fit(x,y).feature_importances_)

    

# Class to extend XGboost classifer
# Put in our parameters for said classifiers

# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}



# Extra Trees Parameters

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



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }
# Create 5 objects that represent our 4 models

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models

y_train = train['Survived'].ravel() # Survivedを１次元配列として取得、つまり、y_trainが解答データ

train = train.drop(['Survived'], axis=1) # Survivedを除去 axisは言葉で説明できない。。。次元に似ているが、ここでは配列の深さのようなもの

x_train = train.values # 訓練データの入力ベクトル

x_test = test.values # テストデータの入力ベクトル（この入力を用いたときの出力を解答として提出）
tetete = np.zeros((10,10))

hoge = [[1,1,1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2,2,2]]

tetete[[7,9]] = hoge

print(tetete)

teru = np.zeros((10))

print(teru)

teru[:] = [1,1,1,1,1,1,1,1,1,1]

print(teru)
NFOLDS = ntrain

kf = KFold(2, shuffle=False, random_state=SEED)

def get_oof(clf, x_train, y_train, x_test): # oof=Out-of-Fold

    oof_train = np.zeros((ntrain)) # ntrain=891 891項の0で満たされた1次元配列

    oof_test = np.zeros((ntest)) # ntest=418

    oof_test_skf = np.empty((ntest)) # 418人分の生存予測解答



    # x_train(学習データ)を学習データとテストデータに分割し、

    for train_index, test_index in kf.split(x_train):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

        y_te = y_train[test_index]



        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te) # clf.predictにより、Surviveしたかの予測結果を返す

        print(test_index)

        print(oof_train)

        # [:]により、代入変数が行列の形に合っていないとエラーになる

        # oof_test_skfは418要素の１次元配列

        # x_test使っているので、提出用の予測解答(418人分の生存予測)が得られる

        oof_test_skf[:] = clf.predict(x_test) # skfってなんやねん。しっかり書けクソ



    oof_test[:] = oof_test_skf.mean(axis=0) # 全要素に平均値を代入

    print(oof_test)

    # reshape(a,b)でa行b列の行列に変換する

    # ここでは、a=-1だが、これは任意の数を意味する。そのため、b=1だけが考慮され、?行1列の行列に変換する。

    # すなわち、縦に物凄く長い行列になる

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1) # skf使わないの！？！？！？
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost

svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier



print("Training is complete")
rf_feature = rf.feature_importances(x_train,y_train)

et_feature = et.feature_importances(x_train, y_train)

ada_feature = ada.feature_importances(x_train, y_train)

gb_feature = gb.feature_importances(x_train,y_train)
rf_features = [0.10474135,  0.21837029,  0.04432652,  0.02249159,  0.05432591,  0.02854371

  ,0.07570305,  0.01088129 , 0.24247496,  0.13685733 , 0.06128402]

et_features = [ 0.12165657,  0.37098307  ,0.03129623 , 0.01591611 , 0.05525811 , 0.028157

  ,0.04589793 , 0.02030357 , 0.17289562 , 0.04853517,  0.08910063]

ada_features = [0.028 ,   0.008  ,      0.012   ,     0.05866667,   0.032 ,       0.008

  ,0.04666667 ,  0.     ,      0.05733333,   0.73866667,   0.01066667]

gb_features = [ 0.06796144 , 0.03889349 , 0.07237845 , 0.02628645 , 0.11194395,  0.04778854

  ,0.05965792 , 0.02774745,  0.07462718,  0.4593142 ,  0.01340093]
cols = train.columns.values

# Create a dataframe with features

feature_dataframe = pd.DataFrame( {'features': cols,

     'Random Forest feature importances': rf_features,

     'Extra Trees  feature importances': et_features,

      'AdaBoost feature importances': ada_features,

    'Gradient Boost feature importances': gb_features

    })
# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['Random Forest feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['Random Forest feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Random Forest Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')



# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['Extra Trees  feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['Extra Trees  feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Extra Trees Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')



# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['AdaBoost feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['AdaBoost feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'AdaBoost Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')



# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['Gradient Boost feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['Gradient Boost feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Gradient Boosting Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')
# Create the new column containing the average of values



feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis=1で平均値を縦に計算

feature_dataframe.head(11)
y = feature_dataframe['mean'].values

x = feature_dataframe['features'].values

data = [go.Bar(

            x= x,

             y= y,

            width = 0.5,

            marker=dict(

               color = feature_dataframe['mean'].values,

            colorscale='Portland',

            showscale=True,

            reversescale = False

            ),

            opacity=0.6

        )]



layout= go.Layout(

    autosize= True,

    title= 'Barplots of Mean Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='bar-direct-labels')
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel()

    })

base_predictions_train.head(40)
data = [

    go.Heatmap(

        z= base_predictions_train.astype(float).corr().values ,

        x=base_predictions_train.columns.values,

        y= base_predictions_train.columns.values,

          colorscale='Viridis',

            showscale=True,

            reversescale = True

    )

]

py.iplot(data, filename='labelled-heatmap')
# np.concatenateは結合。axisでどの軸で結合するかを決定する。詳しくは調べてほしい。

x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

print(x_train)
gbm = xgb.XGBClassifier(

    #learning_rate = 0.02,

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test)
# Generate Submission File 

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)