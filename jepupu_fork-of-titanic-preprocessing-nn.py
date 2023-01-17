import numpy as np
import pandas as pd

# CSV を読み込む
train = pd.read_csv( "../input/train.csv" )
test = pd.read_csv( "../input/test.csv" )
# 訓練データの概要を出力する
train.info()
# テストデータの概要を出力する
test.info()
# 訓練データの統計情報を出力する
train.describe( include="all" )
# テストデータの統計情報を出力する
test.describe( include="all" )
# 欠損値の集計テーブルを返す
def missing_value_table( df ):
    missing_values = df.isnull().sum()
    
    totals = ( np.ones( len( df.columns ), dtype="int" ) * len( df.index ) ).reshape( ( -1, 1 ) )
    totals = pd.DataFrame( totals, index=df.columns, columns=["行数"] )
    
    table = pd.concat( [ totals, missing_values, missing_values / len( df ) * 100 ], axis=1 )
    table = table.rename( columns={ 0: "欠損値", 1: "欠損値 ( % )" } )
    
    return table
# 訓練データの欠損値を確認する
missing_value_table( train )
# テストデータの欠損値を確認する
missing_value_table( test )
from sklearn import preprocessing

# データ前処理
def data_preprocess( df ):
    encoder = preprocessing.LabelEncoder()
    
    # 年齢の欠損値を中央値で置き換える
    df[ "Age" ].fillna( df[ "Age" ].median(), inplace=True )

    # 運賃の欠損値を中央値で置き換える
    df[ "Fare" ].fillna( df[ "Fare" ].median(), inplace=True )
    
    # 乗船港の欠損値を最頻値で置き換える
    df[ "Embarked" ].fillna( df[ "Embarked" ].mode()[0], inplace=True )
    
    
    # 称号
    df[ "Title" ] = df[ "Name" ].str.split( ",", expand=True )[ 1 ].str.split( ".", expand=True )[ 0 ]
    
    # マイナーな称号は 'Misc' に統一する
    is_title_misc = df[ "Title" ].value_counts() < 10
    df[ "Title" ] = df[ "Title" ].apply( lambda title: "Misc" if is_title_misc[ title ] else title )
    
    # 家族数
    df[ "FamilySize" ] = df[ "SibSp" ] + df[ "Parch" ] + 1
    
    # 独り？
    df[ "IsAlone" ] = 0
    df[ "IsAlone" ].loc[ df[ "FamilySize" ] == 1 ];
    
    
    # カテゴリカルデータに置き換える
    df[ "FareBin" ] = pd.qcut( df[ "Fare" ], 4 )
    df[ "AgeBin" ] = pd.cut( df[ "Age" ], 8 )
    
    
    # 性別を数値に変換する
    df[ "Sex_Code" ] = encoder.fit_transform( df[ "Sex" ] )

    # 乗船港を数値に変換する
    df[ "Embarked_Code" ] = encoder.fit_transform( df[ "Embarked" ] )
    
    # 称号を数値に変換する
    df[ "Title_Code" ] = encoder.fit_transform( df[ "Title" ] )
    df[ "FareBin_Code" ] = encoder.fit_transform( df[ "FareBin" ] )
    df[ "AgeBin_Code" ] = encoder.fit_transform( df[ "AgeBin" ] )
    
    categorical_columns = [ "Pclass", "Sex_Code", "Title_Code", "AgeBin_Code", "FareBin_Code", "Embarked_Code" ]
    
    for col in categorical_columns:
        df = pd.concat( [ df, pd.get_dummies( df[ col ], prefix=col ) ], axis=1 )
    
    drop_columns = categorical_columns + [ "Name", "Sex", "Ticket", "Cabin", "Embarked", "Title", "FareBin", "AgeBin" ];
    df.drop( columns=drop_columns, inplace=True )
    
    return df
# 訓練データ前処理
train = data_preprocess( train )
# train.head( 10 )
# train.sample( 100 )[ [ "Title", "Title_Code" ] ]
# train.sample( 20 )[ [ "Fare", "FareBin", "Age", "AgeBin" ] ]
# train.sample( 20 )[ [ "Fare", "FareBin_Code", "Age", "AgeBin_Code" ] ]

# train[ "Title" ].value_counts()

train.head( 10 )
# テストデータ前処理
test = data_preprocess( test )    
test.head( 10 )
y = train[ "Survived" ]
x = train.drop( columns=[ "Survived" ] )

from sklearn import neural_network
from sklearn import feature_selection
from sklearn import model_selection

model = neural_network.MLPClassifier( max_iter=1000 )

param_test = {
    "hidden_layer_sizes": [
        ( 10 ),
        ( 50 ),
        ( 100 ),
        ( 200 ),
        ( 10, 10 ),
        ( 50, 50 ),
        ( 100, 100 ),
    ]
}

search = model_selection.GridSearchCV( model, param_grid=param_test, scoring="accuracy", cv=5 )
search.fit( x.values, y.values )
print( search.best_params_, search.best_score_ )
model = neural_network.MLPClassifier( hidden_layer_sizes=( 10 ), max_iter=1000 )
model.fit( x.values, y.values )

result = model.predict( test.values )
result
# CSV に保存する
result = pd.DataFrame( result, columns = ["Survived"] )

submission = pd.concat( [ test[ "PassengerId" ], result ], axis=1 )
submission.to_csv( "submission.csv", index=False )