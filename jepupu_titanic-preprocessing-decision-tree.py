import numpy as np
import pandas as pd

# CSV を読み込む
train = pd.read_csv( "../input/train.csv" )
test = pd.read_csv( "../input/test.csv" )
# 訓練データの概要を出力する
train.info()
# テストデータの概要を出力する
test.info()
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
# データ前処理
def data_preprocess( df ):
    # 年齢の欠損値を中央値で置き換える
    df[ "Age" ] = df[ "Age" ].fillna( df[ "Age" ].median() )

    # 運賃の欠損値を中央値で置き換える
    df[ "Fare" ] = df[ "Fare" ].fillna( df[ "Fare" ].median() )
    
    # 性別を数値に置き換える
    df[ "Sex" ] = df[ "Sex" ].replace( { "male": 1, "female": 2 } )
# 訓練データ前処理
data_preprocess( train )
train.head( 10 )
# テストデータ前処理
data_preprocess( test )    
test.head( 10 )
from sklearn import tree

feature_columns = [ "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare" ]

model = tree.DecisionTreeClassifier()
model.fit( train[ feature_columns ].values, train[ "Survived" ].values )

result = model.predict( test[ feature_columns ].values )
result
# CSV に保存する
result = pd.DataFrame( result, columns = ["Survived"] )

submission = pd.concat( [ test[ "PassengerId" ], result ], axis=1 )
submission.to_csv( "submission.csv", index=False )