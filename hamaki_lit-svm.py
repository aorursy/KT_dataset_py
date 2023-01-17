import numpy as np
import pandas as pd
import csv as csv
from sklearn import svm

##前処理を定義
def preprocess(data):
	#年齢の欠損値を中央値で補完
	data.Age = data.Age.fillna(data.Age.median())
	#料金の欠損値を中央値で補完
	data.Fare = data.Fare.fillna(data.Fare.median())
	#料金の欠損値を中央値で補完
	data.Embarked = data.Embarked.fillna("S")
	#男→1、女→0に変換
	data["Sex"] = data["Sex"].map( {'female': 0, "male": 1} ).astype(int)
	data["Embarked"] = data["Embarked"].map( {'S': 0, "C": 1, "Q": 2} ).astype(int)
	#不要な列を削除
	del data['Name']
	del data['Ticket']
	del data['Cabin']
	del data['SibSp']
	del data['Parch']
	return data

def main():
	#トレーニングデータ読み込み
	train = pd.read_csv("train.csv")
	#トレーニングデータ前処理
	train_pre = preprocess(train)
	#2列目を正解ラベルに
	label = train_pre.iloc[:,1]
	#3列目以降を特徴ベクトルに
	features_train = train_pre.iloc[:,2:]
	
	##予想モデルの作成
	#SVMのインスタンスを生成
	clf = svm.SVC()
	#学習
	clf.fit(features_train,label)
	
	#テストデータ読み込み
	test  = pd.read_csv("test.csv")
	#テストデータ前処理
	test_pre = preprocess(test)
	#2列目以降を特徴ベクトルに
	features_test = test_pre.iloc[:,1:]
	
	#テストデータに予想モデルを適用
	prediction = clf.predict(features_test)
	
	f = open('write1.csv', 'w')
	writer = csv.writer(f)
	writer.writerow(["PassengerId", "Survived"])
	for pid, survived in zip(test.iloc[:,0].astype(int), prediction.astype(int)):
		writer.writerow([pid,survived])
	f.close()

main()
