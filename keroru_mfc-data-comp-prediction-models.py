# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df=pd.read_csv("../input/result.csv")
df.info()
df=df[['Unique ID','Label','no_of_word_1','no_of_word_2','Similarity','VerbsSimilarity'
       ,'Nums_match','Propn_match','Lemma_match','word_match']]
#df=df[['Unique ID','Label','no_of_word_1','no_of_word_2','Similarity']]

df['diff_word']=df['no_of_word_1']-df['no_of_word_2']
df['diff_word']=df['diff_word']**2
#df.assign(absdiff_word=df.apply(abs(df['diff_word'])))
df.info()
# XGBoost

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report

# 説明変数、目的変数
#features = ['no_of_word_1','no_of_word_2','Similarity','VerbsSimilarity']
features = ['diff_word','no_of_word_1','no_of_word_2','Similarity','VerbsSimilarity'
       ,'Nums_match','Propn_match','Lemma_match','word_match']

X = df[df['Label']>=0][features]
y = df[df['Label']>=0]['Label']
X_submit = df[df['Label'].isna()][features]
# 学習用、検証用データ作成
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = 0.3, random_state = 666)

# XGboostのライブラリをインポート
import xgboost as xgb
# モデルのインスタンス作成
clf = xgb.XGBClassifier()

# ハイパーパラメータの探索
clf_cv = GridSearchCV(clf, {'max_depth': [2,3,4,5,6], 'n_estimators': [25,50,75,100]}, verbose=1)
clf_cv.fit(X_train, y_train)
print (clf_cv.best_params_, clf_cv.best_score_)

# 改めて最適パラメータで学習
clf = xgb.XGBClassifier(**clf_cv.best_params_)
clf.fit(X_train, y_train)

# 学習モデルの評価
pred = clf.predict(X_test)
print("学習モデルの評価")
print( confusion_matrix(y_test, pred))
print( classification_report(y_test, pred))

# Submissionデータの作成
pred_submit = clf.predict(X_submit)
Submission = pd.concat([df[df['Label'].isna()]['Unique ID'].reset_index()
                        ,pd.Series(pred_submit.T.astype(np.int32))] ,axis=1)
Submission.columns=['index','Unique ID','Label']
#Submission = Submission[['Unique ID','Label']]
#print("Submission= ",Submission.info())
#print("ID=",df[df['Label'].isna()]['Unique ID'])
#print("Submission data")
#print(Submission)
Submission = Submission[['Unique ID','Label']]
Submission.to_csv("Submission.csv")
#Submission
# Multi models by Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators

allAlgorithms = all_estimators(type_filter = "classifier")

for(name, algorithm) in allAlgorithms:
    try:
        clf = algorithm()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(name, "の正解率 = ", accuracy_score(y_test, y_pred))
    except ValueError as e:
        print("type:{0}".format(type(e)))
        print("args:{0}".format(e.args))
 
# Neural Network by Keras
# From http://aidiary.hatenablog.com/entry/20161108/1478609028

from sklearn import datasets
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from sklearn import preprocessing

def build_multilayer_perceptron(input_size,output_size):
    """多層パーセプトロンモデルを構築"""
    model = Sequential()
        
    model.add(Dense(12, input_shape=(input_size, )))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    
    """ acc=0.760 30ep
    model.add(Dense(12, input_shape=(input_size, )))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    """
    
    """ acc=0.725 30ep
    model.add(Dense(12, input_shape=(input_size, )))
    model.add(Activation('relu'))
    model.add(Dense(12))
    model.add(Activation('relu'))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    """
    
    """ acc=0.739 50ep
    model.add(Dense(10, input_shape=(input_size, )))
    model.add(Activation('relu'))
    model.add(Dense(6))
    model.add(Activation('relu'))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    """    
    
    return model


#features = ['diff_word','no_of_word_1','no_of_word_2','Similarity','VerbsSimilarity']
features = ['diff_word','no_of_word_1','no_of_word_2','Similarity','VerbsSimilarity'
       ,'Nums_match','Propn_match','Lemma_match','word_match']
X = df[df['Label']>=0][features]
Y = df[df['Label']>=0]['Label']
X_submit = df[df['Label'].isna()][features]

if __name__ == "__main__":

    # データの標準化
    X = preprocessing.scale(X)

    # ラベルをone-hot-encoding形式に変換
    # 0 => [1, 0]
    # 1 => [0, 1]
    Y = np_utils.to_categorical(Y)

    # 訓練データとテストデータに分割
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.8)
    print(train_X.shape, test_X.shape, train_Y.shape, test_Y.shape)

    # モデル構築
    model = build_multilayer_perceptron(len(features),2)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # モデル訓練
    model.fit(train_X, train_Y, nb_epoch=30, batch_size=1, verbose=1)

    # モデル評価
    loss, accuracy = model.evaluate(test_X, test_Y, verbose=0)
    print("Accuracy = {:.3f}".format(accuracy))
    
    
pred_submit_NN = model.predict(X_submit)
pred_submit_NN = np.round(model.predict(X_submit))
pred_submit_NN = pred_submit_NN[:,1]

#print(pred_submit_NN)
Submission_NN = pd.concat([df[df['Label'].isna()]['Unique ID'].reset_index()
                        ,pd.Series(pred_submit_NN.T.astype(np.int32))] ,axis=1)
Submission_NN.columns=['index','Unique ID','Label']
Submission_NN = Submission[['Unique ID','Label']]
Submission_NN.to_csv("Submission_NN.csv")

print(Submission_NN)
"""
# 全訓練データを使って再学習
X = df[df['Label']>=0][features]
Y = df[df['Label']>=0]['Label']
X_submit = df[df['Label'].isna()][features]

if __name__ == "__main__":

    # データの標準化
    X = preprocessing.scale(X)

    # ラベルをone-hot-encoding形式に変換
    # 0 => [1, 0]
    # 1 => [0, 1]
    Y = np_utils.to_categorical(Y)

    # 訓練データとテストデータに分割
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=1)
    
    # モデル構築
    model = build_multilayer_perceptron(len(features),2)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # モデル訓練
    model.fit(train_X, train_Y, nb_epoch=50, batch_size=1, verbose=1)

pred_submit_NN = model.predict(X_submit)
pred_submit_NN = np.round(model.predict(X_submit))
pred_submit_NN = pred_submit_NN[:,1]

#print(pred_submit_NN)
Submission_NN = pd.concat([df[df['Label'].isna()]['Unique ID'].reset_index()
                        ,pd.Series(pred_submit_NN.T.astype(np.int32))] ,axis=1)
Submission_NN.columns=['index','Unique ID','Label']
Submission_NN = Submission[['Unique ID','Label']]
Submission_NN.to_csv("Submission_NN.csv")

print(Submission_NN)
"""