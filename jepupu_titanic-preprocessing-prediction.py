import numpy as np
import pandas as pd

# CSV を読み込む
train = pd.read_csv( "../input/train.csv" )
test = pd.read_csv( "../input/test.csv" )

# 訓練データの概要を出力する
print( "-" * 60 )
print( "train :" )
train.info()

# テストデータの概要を出力する
print( "-" * 60 )
print( "test :" )
test.info()
# 訓練データの統計情報を出力する
train.describe( include="all" )
# テストデータの統計情報を出力する
test.describe( include="all" )
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use( "fivethirtyeight" )

# グラフを描画する
f, ax = plt.subplots( 2, 3, figsize=( 15, 10 ) ) 

ax[ 0, 0 ].set_title( "Pclass vs Survived" )
sns.countplot( "Pclass", hue="Survived", data=train, ax=ax[ 0, 0 ] )

ax[ 0, 1 ].set_title( "Sex vs Survived" )
sns.countplot( "Sex", hue="Survived", data=train, ax=ax[ 0, 1 ] )

ax[ 0, 2 ].set_title( "Embarked vs Survived" )
sns.countplot( "Embarked", hue="Survived", data=train, ax=ax[ 0, 2 ] )

ax[ 1, 0 ].set_title( "Age vs Survived" )
sns.violinplot( None, "Age", hue="Survived", split=True, data=train, ax=ax[ 1, 0 ] )

ax[ 1, 1 ].set_title( "Fare vs Survived" )
sns.violinplot( None, "Fare", hue="Survived", split=True, data=train, ax=ax[ 1, 1 ] )

plt.show()

# train[ [ "Pclass", "Survived" ] ].groupby( [ "Pclass" ] ).count()
# train[ [ "Sex", "Survived" ] ].groupby( [ "Sex" ] ).count()

# sns.set( style="darkgrid" )
# sns.barplot( x="Pclass", y="Survived", data=train )
# sns.barplot( x="Sex", y="Survived", data=train )
# 欠損値の集計テーブルを返す
def missing_value_table( df ):
    missing_values = df.isnull().sum()
    
    totals = ( np.ones( len( df.columns ), dtype="int" ) * len( df.index ) ).reshape( ( -1, 1 ) )
    totals = pd.DataFrame( totals, index=df.columns, columns=[ "行数" ] )
    
    table = pd.concat( [ totals, missing_values, missing_values / len( df ) * 100 ], axis=1 )
    table = table.rename( columns={ 0: "欠損値", 1: "欠損値 ( % )" } )
    
    return table
# 訓練データの欠損値を確認する
missing_value_table( train )
# テストデータの欠損値を確認する
missing_value_table( test )
from sklearn import preprocessing

label_encoders = {}
one_hot_encoders = {}

# データ前処理
def data_preprocess( df, is_train ):
    # 年齢の欠損値を中央値で置き換える
    df[ "Age" ].fillna( df[ "Age" ].median(), inplace=True )

    # 運賃の欠損値を中央値で置き換える
    df[ "Fare" ].fillna( df[ "Fare" ].median(), inplace=True )
    
    # 乗船港の欠損値を最頻値で置き換える
    df[ "Embarked" ].fillna( df[ "Embarked" ].mode()[0], inplace=True )
    
    # 客室番号の欠損値を '-' で置き換える
    df[ "Cabin" ].fillna( "-", inplace=True )
    
    
    # 客室番号の先頭 1 文字だけに注目する
    df[ "Cabin_Initial" ] = df[ "Cabin" ].apply( lambda s: s[ 0 ] );
    
    # チケット番号の先頭 1 文字だけに注目する
    df[ "Ticket_Initial" ] = df[ "Ticket" ].apply( lambda s: s[ 0 ] )
    
    # sns.barplot( x="Cabin", y="Survived", data=df )
    # print( df[ "Cabin" ].value_counts() )
    
    # 称号
    df[ "Title" ] = df[ "Name" ].str.split( ",", expand=True )[ 1 ].str.split( ".", expand=True )[ 0 ]
    
    # マイナーな称号は 'Misc' に統一する
    is_title_misc = df[ "Title" ].value_counts() < 10
    df[ "Title" ] = df[ "Title" ].apply( lambda title: "Misc" if is_title_misc[ title ] else title )
    
    # 家族数
    df[ "FamilySize" ] = df[ "SibSp" ] + df[ "Parch" ] + 1
    
    # 独り？
    df[ "IsAlone" ] = 0
    df.loc[ df[ "FamilySize" ] == 1, "IsAlone" ] = 1;
    
    
    # カテゴリカルデータに置き換える
    # 年齢
    df[ "AgeBin" ] = 0
    df.loc[ ( df[ "Age" ] >= 12 ) & ( df[ "Age" ] < 20 ), "AgeBin" ] = 1
    df.loc[ ( df[ "Age" ] >= 20 ) & ( df[ "Age" ] < 40 ), "AgeBin" ] = 2
    df.loc[ ( df[ "Age" ] >= 40 ), "AgeBin" ] = 3
    # 運賃
    df[ "FareBin" ] = 0
    df.loc[ ( df[ "Fare" ] >=  50 ) & ( df[ "Fare" ] < 100 ), "FareBin" ] = 1
    df.loc[ ( df[ "Fare" ] >= 100 ) & ( df[ "Fare" ] < 200 ), "FareBin" ] = 2
    df.loc[ ( df[ "Fare" ] >= 200 ), "FareBin" ] = 4
    
    
    # カテゴリカルデータを変換する
    categorical_columns = [ "Pclass", "Sex", "Embarked", "Cabin_Initial", "Ticket_Initial", "Title", "AgeBin", "FareBin" ]
    
    
    for col in categorical_columns:
        
        global label_encoders
        global one_hot_encoders
        
        # カテゴリカルデータを数字に変換する
        if is_train:
            label_encoders[ col ] = preprocessing.LabelEncoder()
            label_encoders[ col ].fit( df[ col ] )
        
        df[ col + "_Code" ] = label_encoders[ col ].transform( df[ col ] )

        
        # カテゴリカルデータを OneHot に変換する
        if is_train:
            one_hot_encoders[ col ] = preprocessing.OneHotEncoder( handle_unknown="ignore" )
            one_hot_encoders[ col ].fit( df[ [ col + "_Code" ] ] )
        
        e = one_hot_encoders[ col ]
        one_hot = e.transform( df[ [ col + "_Code" ] ] ).astype( np.int ).toarray()
        one_hot_df = pd.DataFrame( one_hot, columns=[ col + "_" + str( s ) for s in e.categories_[ 0 ] ] )
        df = pd.concat( [ df, one_hot_df ], axis=1 )
    
    # 不要な列を削除する
    drop_columns = [ "Pclass", "Name", "Sex", "Ticket", "Ticket_Initial", "Cabin", "Cabin_Initial", "Embarked", "Title", "FareBin", "AgeBin" ];
    df.drop( columns=drop_columns, inplace=True )
    
    return df
# 訓練データ前処理
train = data_preprocess( train, True )
train.head( 10 )
# テストデータ前処理
test = data_preprocess( test, False )    
test.head( 10 )
y = train[ "Survived" ]
x = train.drop( columns=[ "Survived" ] )

from sklearn import ensemble
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import svm
from xgboost import XGBClassifier

# globals()[ 'ensemble' ].RandomForestClassifier
model_params = {
    "RandomForestClassifier" : {
        "class": ensemble.RandomForestClassifier,
        "param_grid": {
            "criterion": [ "entropy" ], # [ "gini", "entropy" ],
            "max_depth": list( range( 5, 20, 1 ) ),
            "min_samples_split" : list( range( 2, 20, 1 ) ),
            "n_estimators": list( range( 10, 200, 5 ) ), 
            "oob_score": [ True ],
        },
        "best_params": {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 10, 'n_estimators': 60, 'oob_score': True}
    },
    "XGBClassifier": {
        "class": XGBClassifier,
        "param_grid": {
            "max_depth": range( 2, 10, 1 ),
            "learning_rate": [ 0.01, 0.03, 0.05, 0.1, 0.25 ],
            "n_estimators": [ 10, 20, 25, 30, 40, 50, 100, 200 ],
        },
        "best_params": {
            "max_depth": 3,
            "learning_rate": 0.03,
            "n_estimators": 20,
        },
    },
    "SVC": {
        "class": svm.SVC,
        "param_grid": {
            "kernel": [ "rbf", "sigmoid" ], # [ "linear", "poly", "rbf", "sigmoid" ],
            "C": [ 0.1, 0.3, 0.5, 0.7, 0.8 ], # range( 2, 3 ),
            "gamma": [ 0.01, 0.05, 0.1 ], # [ 0.1, 0.25, 0.5, 0.75, 1.0 ],
            "decision_function_shape": [ "ovo" ], # [ "ovo", "ovr" ],
        },
        "best_params": {
            "kernel": "sigmoid",
            "C": 0.1,
            "decision_function_shape": "ovo",
            "gamma": "auto", 
        },
    },
}

model_param = model_params[ "RandomForestClassifier" ]
# model_param = model_params[ "XGBClassifier" ]
# model_param = model_params[ "SVC" ]

if False:
    # 自動で最適なハイパーパラメータを探して最適なモデルを取得する
    model = model_param[ "class" ]( random_state=0 )
    search = model_selection.GridSearchCV( model, param_grid=model_param[ "param_grid" ], scoring="accuracy", n_jobs=-1, iid=False, cv=5, return_train_score=False )
    search.fit( x.values, y.values )
    model = search.best_estimator_
    # print( search.cv_results_ )
    print( "best score :", search.best_score_ )
    print( "best params :", search.best_params_ )
else:
    # 手動でハイパーパラメータを指定してモデルを生成し学習する    
    model = model_param[ "class" ]( random_state=0, **model_param[ "best_params" ] )
    model.fit( x.values, y.values )

print( model )
# 予測する
result = model.predict( test.values )
print( result )

# 学習データに対してのスコアを出力する
score = model_selection.cross_val_score( model, x.values, y.values, cv=5 )
print( "scores :", score )
print( "mean score : ", np.mean( score ) )

# 3, 3 : 0.8384
# 2, 2 : 0.8373

# CSV に保存する
result = pd.DataFrame( result, columns = ["Survived"] )

submission = pd.concat( [ test[ "PassengerId" ], result ], axis=1 )
submission.to_csv( "submission.csv", index=False )