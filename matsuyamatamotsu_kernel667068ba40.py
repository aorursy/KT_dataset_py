# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# データフレーム用クラスをインポート
import pandas as pd

# 描画用ライブラリをインポート
import matplotlib.pyplot as plt
import seaborn as sns

# numpy のデータが扱えるよう、numpy をインポート
import numpy as np

# サンプル分割用の関数をインポート
from sklearn.model_selection import train_test_split

# データの標準化を行うためのモジュールをインポート
#from sklearn.preprocessing import StandardScaler

# 勾配ブースティング決定木を行うためのモジュールをインポート
from sklearn.ensemble import GradientBoostingClassifier

# 正解率を作成するためのモジュールをインポート
from sklearn.metrics import accuracy_score

# 混合行列を作成するためのモジュールをインポート
from sklearn.metrics import confusion_matrix

# アンダーサンプリングを行うためのモジュールをインポート
#from imblearn.under_sampling import RandomUnderSampler

# カテゴリ変数をOneHotベクトル化するためのモジュールをインポート
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# ROCを作成するためのモジュールをインポート
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# ハイパーパラメータの探索用
from sklearn import model_selection, metrics
from sklearn.model_selection import GridSearchCV   #Perforing grid search
#from sklearn.externals import joblib # for save
# CSVデータを読み込む
# 学習用データ（train.tsv)
trainData = pd.read_csv('/kaggle/input/titanic/train.csv', delimiter=',')
# CSVデータの概要を把握
trainData.describe()
# 欠損値の有無を把握
trainData.isnull().sum()
# 実データを確認
trainData.head()
# 目的変数の分布状況を把握
sns.countplot(x='Survived', data=trainData)
# 説明変数の分布状況を把握
sns.countplot(x='Pclass', data=trainData)
# Pclass別の生存数カウント
sns.countplot(trainData['Pclass'], hue=trainData['Survived'])
# 欠損値の穴埋め
# -99で埋める
trainData["Age"].fillna(-99, inplace=True)
# Age_NaN
trainData["Age_NaN"] = trainData.apply(lambda x: 1 if x["Age"] == -99 else  0, axis=1)
# 中央値で埋める
#val = median(trainData["age"].dropna())
#print(val)
#trainData["age"].fillna(val, inplace=True)

# 説明変数の分布状況を把握
sns.distplot(trainData["Age"].dropna())
# 生存と低年齢
trainData[(trainData["Age"] <= 10)][["Age", "Survived", "Pclass", "Fare", "Name"]]
# 年齢不詳の男性
trainData[(trainData["Name"].str.contains("Mr\.")) & (trainData["Age"] == -99)].describe()
# Mrの生存と平均年齢
tmpData = trainData[trainData["Name"].str.contains("Mr\.")][["Age", "Survived", "Pclass", "Parch"]]
#print(tmpData)
tmpData1 = tmpData[(tmpData["Age"] != -99) & (tmpData["Pclass"] == 1) & (tmpData["Parch"] == 0)]
tmpData1.groupby("Survived").agg("mean")
tmpData1.describe()
tmpData1 = tmpData[(tmpData["Age"] != -99) & (tmpData["Pclass"] == 1) & (tmpData["Parch"] == 1)]
tmpData1.groupby("Survived").agg("mean")
tmpData1.describe()
tmpData1 = tmpData[(tmpData["Age"] != -99) & (tmpData["Pclass"] == 1) & (tmpData["Parch"] == 2)]
tmpData1.groupby("Survived").agg("mean")
tmpData1.describe()
tmpData2 = tmpData[(tmpData["Age"] != -99) & (tmpData["Pclass"] == 2) & (tmpData["Parch"] == 0)]
tmpData2.groupby("Survived").agg("mean")
tmpData2.describe()
tmpData2 = tmpData[(tmpData["Age"] != -99) & (tmpData["Pclass"] == 2) & (tmpData["Parch"] == 1)]
tmpData2.groupby("Survived").agg("mean")
tmpData2.describe()
tmpData2 = tmpData[(tmpData["Age"] != -99) & (tmpData["Pclass"] == 2) & (tmpData["Parch"] == 2)]
tmpData2.groupby("Survived").agg("mean")
tmpData2.describe()
tmpData3 = tmpData[(tmpData["Age"] != -99) & (tmpData["Pclass"] == 3) & (tmpData["Parch"] == 0)]
tmpData3.groupby("Survived").agg("mean")
tmpData3.describe()
tmpData3 = tmpData[(tmpData["Age"] != -99) & (tmpData["Pclass"] == 3) & (tmpData["Parch"] == 1)]
tmpData3.groupby("Survived").agg("mean")
tmpData3.describe()
tmpData3 = tmpData[(tmpData["Age"] != -99) & (tmpData["Pclass"] == 3) & (tmpData["Parch"] == 2)]
tmpData3.groupby("Survived").agg("mean")
tmpData3.describe()
# 男性
trainData[trainData["Name"].str.contains("Mr\.")][["Survived", "Age", "Name", "Pclass", "Parch"]]
trainData[(trainData["Name"].str.contains("Mr\.")) & (trainData["Age"] == -99)].describe()
trainData.loc[
    (trainData["Pclass"] == 1) & 
    (trainData["Parch"] == 0) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 0),
    'Age'
] = 41.000000
trainData.loc[
    (trainData["Pclass"] == 1) & 
    (trainData["Parch"] == 0) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 1), 
    'Age'
] = 41.000000
trainData.loc[
    (trainData["Pclass"] == 1) & 
    (trainData["Parch"] == 1) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 0),
    'Age'
] = 38.000000
trainData.loc[
    (trainData["Pclass"] == 1) & 
    (trainData["Parch"] == 1) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 1), 
    'Age'
] = 38.000000
trainData.loc[
    (trainData["Pclass"] == 1) & 
    (trainData["Parch"] == 2) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 0),
    'Age'
] = 27.000000
trainData.loc[
    (trainData["Pclass"] == 1) & 
    (trainData["Parch"] == 2) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 1), 
    'Age'
] = 27.000000
trainData.loc[
    (trainData["Pclass"] == 2) & 
    (trainData["Parch"] == 0) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 0),
    'Age'
] = 30.000000
trainData.loc[
    (trainData["Pclass"] == 2) & 
    (trainData["Parch"] == 0) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 1), 
    'Age'
] = 30.000000
trainData.loc[
    (trainData["Pclass"] == 2) & 
    (trainData["Parch"] == 1) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 0),
    'Age'
] = 31.000000
trainData.loc[
    (trainData["Pclass"] == 2) & 
    (trainData["Parch"] == 1) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 1), 
    'Age'
] = 31.000000
trainData.loc[
    (trainData["Pclass"] == 2) & 
    (trainData["Parch"] == 2) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 0),
    'Age'
] = 36.00
trainData.loc[
    (trainData["Pclass"] == 2) & 
    (trainData["Parch"] == 2) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 1), 
    'Age'
] = 36.00
trainData.loc[
    (trainData["Pclass"] == 3) & 
    (trainData["Parch"] == 0) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 0),
    'Age'
] = 26.000000
trainData.loc[
    (trainData["Pclass"] == 3) & 
    (trainData["Parch"] == 0) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 1), 
    'Age'
] = 26.000000
trainData.loc[
    (trainData["Pclass"] == 3) & 
    (trainData["Parch"] == 1) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 0),
    'Age'
] = 20.000000
trainData.loc[
    (trainData["Pclass"] == 3) & 
    (trainData["Parch"] == 1) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 1), 
    'Age'
] = 20.000000
trainData.loc[
    (trainData["Pclass"] == 3) & 
    (trainData["Parch"] == 2) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 0),
    'Age'
] = 19.000000
trainData.loc[
    (trainData["Pclass"] == 3) & 
    (trainData["Parch"] == 2) & 
    (trainData["Name"].str.contains("Mr\.")) & 
    (trainData["Age"] == -99) & 
    (trainData["Survived"] == 1), 
    'Age'
] = 19.000000
trainData[trainData["Name"].str.contains("Mr\.")][["Survived", "Age", "Name", "Pclass"]]
# 説明変数の分布状況を把握
sns.distplot(trainData["Age"])
# Mrの生存と平均年齢
tmpData = trainData[trainData["Name"].str.contains("Mrs\.")][["Age", "Survived"]]
#print(tmpData)
tmpData = tmpData[tmpData["Age"] != -99]
tmpData.groupby("Survived").agg("mean")
tmpData.describe()
trainData[trainData["Name"].str.contains("Mrs\.")][["Survived", "Age", "Name"]]
trainData.loc[(trainData["Name"].str.contains("Mrs\.")) & (trainData["Age"] == -99) & (trainData["Survived"] == 0), 'Age'] = 35.000000
trainData.loc[(trainData["Name"].str.contains("Mrs\.")) & (trainData["Age"] == -99) & (trainData["Survived"] == 1), 'Age'] = 35.000000
# 説明変数の分布状況を把握
sns.distplot(trainData["Age"])
# Mrの生存と平均年齢
tmpData = trainData[trainData["Name"].str.contains("Miss\.")][["Age", "Survived"]]
#print(tmpData)
tmpData = tmpData[tmpData["Age"] != -99]
tmpData.groupby("Survived").agg("mean")
tmpData.describe()
trainData[trainData["Name"].str.contains("Miss\.")][["Survived", "Age", "Name"]]
trainData.loc[(trainData["Name"].str.contains("Miss\.")) & (trainData["Age"] == -99) & (trainData["Survived"] == 0), 'Age'] = 22.000000
trainData.loc[(trainData["Name"].str.contains("Miss\.")) & (trainData["Age"] == -99) & (trainData["Survived"] == 1), 'Age'] = 22.000000
# 説明変数の分布状況を把握
sns.distplot(trainData["Age"])
# Mrの生存と平均年齢
tmpData = trainData[trainData["Name"].str.contains("Master\.")][["Age", "Survived"]]
#print(tmpData)
tmpData = tmpData[tmpData["Age"] != -99]
tmpData.groupby("Survived").agg("mean")
tmpData.describe()
trainData[trainData["Name"].str.contains("Master\.")][["Survived", "Age", "Name"]]
trainData.loc[(trainData["Name"].str.contains("Master\.")) & (trainData["Age"] == -99) & (trainData["Survived"] == 0), 'Age'] = 3.500000
trainData.loc[(trainData["Name"].str.contains("Master\.")) & (trainData["Age"] == -99) & (trainData["Survived"] == 1), 'Age'] = 3.500000
# 説明変数の分布状況を把握
sns.distplot(trainData["Age"])
trainData[trainData["Age"] == -99]
trainData.loc[(trainData["Name"].str.contains("Dr\.")) & (trainData["Age"] == -99), 'Age'] = 41.000000
# 説明変数の分布状況を把握
sns.distplot(trainData["Age"])
# Pclass別の生存数カウント
sns.countplot(trainData['Age'], hue=trainData['Survived'])
# 年齢別ヒストグラム
trainData['Age'].hist(bins=70)
# 説明変数の分布状況を把握
sns.countplot(x='SibSp', data=trainData)
# 説明変数の分布状況を把握
sns.countplot(x='Parch', data=trainData)
# 説明変数の分布状況を把握
sns.distplot(trainData["Fare"])
# 運賃別ヒストグラム
trainData['Fare'].hist(bins=100)
# 説明変数の値を加工
trainData["Sex_val"] = trainData.apply(lambda x: 1 if x["Sex"] == "male" else  0, axis=1)
trainData["Pclass_1"] = trainData.apply(lambda x: 1 if x["Pclass"] == 1 else 0, axis=1)
trainData["Pclass_2"] = trainData.apply(lambda x: 1 if x["Pclass"] == 2 else 0, axis=1)
trainData["Pclass_3"] = trainData.apply(lambda x: 1 if x["Pclass"] == 3 else 0, axis=1)
trainData.head()
# 欠損値の穴埋め
# 0で埋める
trainData["Cabin"].fillna("ZZ", inplace=True)

# 頭文字を取得
trainData["Cabin_val"] = trainData["Cabin"].str[0:1]

# 説明変数の分布状況を把握
sns.countplot(x='Cabin_val', data=trainData)
# Cabinごとの生存率
trainData[["Cabin_val", "Survived"]].groupby("Cabin_val").agg("mean")
trainData["Cabin_A"] = trainData.apply(lambda x: 1 if x["Cabin_val"] == "A" else 0, axis=1)
trainData["Cabin_B"] = trainData.apply(lambda x: 1 if x["Cabin_val"] == "B" else 0, axis=1)
trainData["Cabin_C"] = trainData.apply(lambda x: 1 if x["Cabin_val"] == "C" else 0, axis=1)
trainData["Cabin_D"] = trainData.apply(lambda x: 1 if x["Cabin_val"] == "D" else 0, axis=1)
trainData["Cabin_E"] = trainData.apply(lambda x: 1 if x["Cabin_val"] == "E" else 0, axis=1)
trainData["Cabin_F"] = trainData.apply(lambda x: 1 if x["Cabin_val"] == "F" else 0, axis=1)
trainData["Cabin_T"] = trainData.apply(lambda x: 1 if x["Cabin_val"] == "T" else 0, axis=1)
trainData["Cabin_Z"] = trainData.apply(lambda x: 1 if x["Cabin_val"] == "Z" else 0, axis=1)
trainData.head()
# 説明変数の値を加工
trainData["Embarked_val"] = trainData.apply(lambda x: 0 if x["Embarked"] == "S" else (1 if x["Embarked"] == "C" else 2), axis=1)
trainData["Embarked_S"] = trainData.apply(lambda x: 1 if x["Embarked"] == "S" else 0, axis=1)
trainData["Embarked_C"] = trainData.apply(lambda x: 1 if x["Embarked"] == "C" else 0, axis=1)
trainData["Embarked_Q"] = trainData.apply(lambda x: 1 if x["Embarked"] == "Q" else 0, axis=1)
trainData.head()
# sibsp をone-hotベクトル化
# 決定木分析では効果がないといわれている
sibsp_enc = trainData["SibSp"].values
sibsp_enc = LabelEncoder().fit_transform(sibsp_enc).reshape(-1,1)
sibsp_enc2 = OneHotEncoder(categories='auto').fit_transform(sibsp_enc).toarray()
trainData["SibSp_0"] = sibsp_enc2[:,0].astype(int)
trainData["SibSp_1"] = sibsp_enc2[:,1].astype(int)
trainData["SibSp_2"] = sibsp_enc2[:,2].astype(int)
trainData["SibSp_3"] = sibsp_enc2[:,3].astype(int)
trainData["SibSp_4"] = sibsp_enc2[:,4].astype(int)
trainData["SibSp_5"] = sibsp_enc2[:,5].astype(int)
trainData["SibSp_6"] = sibsp_enc2[:,6].astype(int)
trainData.head()
# parch をone-hotベクトル化
parch_enc = trainData["Parch"].values
parch_enc = LabelEncoder().fit_transform(parch_enc).reshape(-1,1)
parch_enc2 = OneHotEncoder(categories='auto').fit_transform(parch_enc).toarray()
#parch_enc2 = OneHotEncoder().fit_transform(parch_enc).toarray()
trainData["Parch_0"] = parch_enc2[:,0].astype(int)
trainData["Parch_1"] = parch_enc2[:,1].astype(int)
trainData["Parch_2"] = parch_enc2[:,2].astype(int)
trainData["Parch_3"] = parch_enc2[:,3].astype(int)
trainData["Parch_4"] = parch_enc2[:,4].astype(int)
trainData["Parch_5"] = parch_enc2[:,5].astype(int)
trainData.head()
# 変数を追加【家族】
trainData["Family"] = trainData["SibSp"] + trainData["Parch"] + 1
# 説明変数の分布状況を把握
sns.countplot(x='Family', data=trainData)
# Familyごとの生存率
trainData[["Family", "Survived"]].groupby("Family").agg("mean")
# family をone-hotベクトル化
family_enc = trainData["Family"].values
family_enc = LabelEncoder().fit_transform(family_enc).reshape(-1,1)
family_enc2 = OneHotEncoder().fit_transform(family_enc).toarray()
trainData["Family_0"] = family_enc2[:,0].astype(int)
trainData["Family_1"] = family_enc2[:,1].astype(int)
trainData["Family_2"] = family_enc2[:,2].astype(int)
trainData["Family_3"] = family_enc2[:,3].astype(int)
trainData["Family_4"] = family_enc2[:,4].astype(int)
trainData["Family_5"] = family_enc2[:,5].astype(int)
trainData["Family_6"] = family_enc2[:,6].astype(int)
trainData["Family_7"] = family_enc2[:,7].astype(int)
trainData["Family_8"] = family_enc2[:,8].astype(int)
trainData.head()
# データを訓練用データ・検証用データに分ける（訓練用：検証用＝7:3）
feature_names=[
    "Age",
    "Age_NaN",
    "Pclass_1",
    "Pclass_2",
    "Pclass_3",
    "Sex_val",
    "SibSp",
    "Parch",
    "Parch_0",
    "Parch_1",
    "Parch_2",
    "Parch_3",
    "Parch_4",
    "Parch_5",
    "SibSp_0",
    "SibSp_1",
    "SibSp_2",
    "SibSp_3",
    "SibSp_4",
    "SibSp_5",
    "SibSp_6",
    "Fare",
    "Embarked_S",
    "Embarked_C",
    "Embarked_Q",
    "Family",
    "Family_0",
    "Family_1",
    "Family_2",
    "Family_3",
    "Family_4",
    "Family_5",
    "Family_6",
#    "Family_7",
#    "Family_8",
    "Cabin_A",
    "Cabin_B",
    "Cabin_C",
    "Cabin_D",
    "Cabin_E",
    "Cabin_F",
    "Cabin_T",
    "Cabin_Z",
]
data=trainData[feature_names]
target=trainData["Survived"]
(X_train, X_test ,y_train, y_test) = train_test_split(data, target, test_size = 0.3)
# アンダーサンプリングを行う

# 正例の数を保存
#positive_count_train = y_train.sum()

# 正例が11.1％になるまで負例をダウンサンプリング
#rus = RandomUnderSampler(ratio={0:positive_count_train*8, 1:positive_count_train})
# 正例が10％になるまで負例をダウンサンプリング
#rus = RandomUnderSampler(ratio={0:positive_count_train*9, 1:positive_count_train})

# 学習用データに反映
#X_train_resampled, y_train_resampled = rus.fit_sample(X_train, y_train)

X_train_resampled = X_train
y_train_resampled = y_train
# パラメータの定義
param_test={
    'n_estimators':list(range(40,80,5)),
#    'subsample':[1.0, 0.9],
    'max_depth':list(range(3,5,1)),
#    'min_samples_split':list(range(2,10,1)),
#    'min_samples_leaf':list(range(1,5,1)),
    'learning_rate':[0.01, 0.02, 0.03, 0.04],
#    'loss':['deviance', 'exponential'],
#    'max_features':['sqrt', 'log2'],
}
# 交差検証（クロスバリデーション）
gsearch1 = GridSearchCV(
    estimator = GradientBoostingClassifier(), # 勾配ブースティング決定木
    param_grid = param_test, scoring='roc_auc',n_jobs=5, cv=5)
gsearch1.fit(X_train_resampled, y_train_resampled)

predictor=gsearch1.best_estimator_
# シリアライズして保存
#joblib.dump(predictor,"predictor_gbc.pkl",compress=True)
# 性能の評価結果を出力
liprediction=predictor.predict(X_train_resampled)
table=metrics.confusion_matrix(y_train_resampled, liprediction)
tn,fp,fn,tp=table[0][0],table[0][1],table[1][0],table[1][1]

# True Positive Rate：真陽性率
print("TPR\t{0:.3f}".format(tp/(tp+fn)))

# Specifieity 特異度（偽陽性が少ないほど高い）
print("SPC\t{0:.3f}".format(tn/(tn+fp)))

# Positive Predictive Value：陽性的中率
print("PPV\t{0:.3f}".format(tp/(tp+fp)))

# Accuracy：正確率
print("ACC\t{0:.3f}".format((tp+tn)/(tp+fp+fn+tn)))

# Matthews Correlation Coefficient：マシューズ相関係数
print("MCC\t{0:.3f}".format((tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(1/2)))

# F1 Score：PPV と TPR の調和平均
print("F1\t{0:.3f}".format((2*tp)/(2*tp+fp+fn)))

# パラメーターを出力
print(sorted(predictor.get_params(True).items()))
# 精度検証を行う（訓練用データを確認）
print("ACC\t{0:.3f}".format(accuracy_score(y_train, predictor.predict(X_train))))
conf_mtrx_df = pd.DataFrame(confusion_matrix(y_train, predictor.predict(X_train).reshape(-1,1)))
conf_mtrx_df.rename(columns={0: 'predict(0)',1: 'predict(1)'}, index={0: 'target(0)',1: 'target(1)'})
# 精度検証を行う（検証用データを確認）
print("ACC\t{0:.3f}".format(accuracy_score(y_test, predictor.predict(X_test))))
conf_mtrx_df = pd.DataFrame(confusion_matrix(y_test, predictor.predict(X_test).reshape(-1,1)))
conf_mtrx_df.rename(columns={0: 'predict(0)',1: 'predict(1)'}, index={0: 'target(0)',1: 'target(1)'})
# 説明変数の重要度を可視化する
values, names = zip(*sorted(zip(predictor.feature_importances_, feature_names)))
plt.figure(figsize=(12,12))
plt.barh(range(len(names)), values, align='center')
plt.yticks(range(len(names)), names)
# 精度検証を行う（可視化）

# 訓練用データについて

# FPR、TPR、（閾値）、AUCを算出
y_train_predict = predictor.predict(X_train)
fpr, tpr, thresholds = roc_curve(y_train, y_train_predict)
auc_value_train = auc(fpr, tpr)
# ROC曲線をプロット
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc_value_train)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
print("AUC(train): "+str(auc_value_train))

# 検証用データについて

# FPR、TPR、（閾値）、AUCを算出
y_test_predict = predictor.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_test_predict)
auc_value_test = auc(fpr, tpr)
# ROC曲線をプロット
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc_value_test)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
print("AUC(test): "+str(auc_value_test))
print("AUC ratio(train/test): "+str(auc_value_train/auc_value_test))
# テストデータの読み込み
testData = pd.read_csv('/kaggle/input/titanic/test.csv', delimiter=',')
# 欠損値の有無を把握
testData.isnull().sum()
# データの前処理
# 欠損値の穴埋め
testData["Age"].fillna(-99, inplace=True) # -99で埋める
testData["Fare"].fillna(-99, inplace=True) # -99で埋める
testData["Cabin"].fillna("ZZ", inplace=True) # ZZで埋める
# 説明変数の値を加工
testData["Age_NaN"] = testData.apply(lambda x: 1 if x["Age"] == -99 else  0, axis=1)
testData["Sex_val"] = testData.apply(lambda x: 1 if x["Sex"] == "male" else  0, axis=1)
testData["Pclass_1"] = testData.apply(lambda x: 1 if x["Pclass"] == 1 else 0, axis=1)
testData["Pclass_2"] = testData.apply(lambda x: 1 if x["Pclass"] == 2 else 0, axis=1)
testData["Pclass_3"] = testData.apply(lambda x: 1 if x["Pclass"] == 3 else 0, axis=1)
testData["Embarked_val"] = testData.apply(lambda x: 0 if x["Embarked"] == "S" else (1 if x["Embarked"] == "C" else 2), axis=1)
testData["Embarked_S"] = testData.apply(lambda x: 1 if x["Embarked"] == "S" else 0, axis=1)
testData["Embarked_C"] = testData.apply(lambda x: 1 if x["Embarked"] == "C" else 0, axis=1)
testData["Embarked_Q"] = testData.apply(lambda x: 1 if x["Embarked"] == "Q" else 0, axis=1)
parch_enc = testData["Parch"].values
parch_enc = LabelEncoder().fit_transform(parch_enc).reshape(-1,1)
parch_enc2 = OneHotEncoder(categories='auto').fit_transform(parch_enc).toarray()
testData["Parch_0"] = parch_enc2[:,0].astype(int)
testData["Parch_1"] = parch_enc2[:,1].astype(int)
testData["Parch_2"] = parch_enc2[:,2].astype(int)
testData["Parch_3"] = parch_enc2[:,3].astype(int)
testData["Parch_4"] = parch_enc2[:,4].astype(int)
testData["Parch_5"] = parch_enc2[:,5].astype(int)

testData.loc[
    (testData["Pclass"] == 1) & 
    (testData["Parch"] == 0) & 
    (testData["Name"].str.contains("Mr\.")) & 
    (testData["Age"] == -99),
    'Age'
] = 41.000000
testData.loc[
    (testData["Pclass"] == 1) & 
    (testData["Parch"] == 1) & 
    (testData["Name"].str.contains("Mr\.")) & 
    (testData["Age"] == -99),
    'Age'
] = 38.000000
testData.loc[
    (testData["Pclass"] == 1) & 
    (testData["Parch"] == 2) & 
    (testData["Name"].str.contains("Mr\.")) & 
    (testData["Age"] == -99),
    'Age'
] = 27.000000
testData.loc[
    (testData["Pclass"] == 2) & 
    (testData["Parch"] == 0) & 
    (testData["Name"].str.contains("Mr\.")) & 
    (testData["Age"] == -99),
    'Age'
] = 30.000000
testData.loc[
    (testData["Pclass"] == 2) & 
    (testData["Parch"] == 1) & 
    (testData["Name"].str.contains("Mr\.")) & 
    (testData["Age"] == -99),
    'Age'
] = 31.000000
testData.loc[
    (testData["Pclass"] == 2) & 
    (testData["Parch"] == 2) & 
    (testData["Name"].str.contains("Mr\.")) & 
    (testData["Age"] == -99),
    'Age'
] = 36.00
testData.loc[
    (testData["Pclass"] == 3) & 
    (testData["Parch"] == 0) & 
    (testData["Name"].str.contains("Mr\.")) & 
    (testData["Age"] == -99),
    'Age'
] = 26.000000
testData.loc[
    (testData["Pclass"] == 3) & 
    (testData["Parch"] == 1) & 
    (testData["Name"].str.contains("Mr\.")) & 
    (testData["Age"] == -99),
    'Age'
] = 20.000000
testData.loc[
    (testData["Pclass"] == 3) & 
    (testData["Parch"] == 2) & 
    (testData["Name"].str.contains("Mr\.")) & 
    (testData["Age"] == -99),
    'Age'
] = 19.000000

testData.loc[
    (testData["Pclass"] == 3) & 
    (testData["Parch"] == 9) & 
    (testData["Name"].str.contains("Mr\.")) & 
    (testData["Age"] == -99),
    'Age'
] = 41.000000

testData.loc[(testData["Name"].str.contains("Mrs\.")) & (testData["Age"] == -99), 'Age'] = 36.423077
testData.loc[(testData["Name"].str.contains("Miss\.")) & (testData["Age"] == -99), 'Age'] = 18.636364
testData.loc[(testData["Name"].str.contains("Ms\.")) & (testData["Age"] == -99), 'Age'] = 18.636364
testData.loc[(testData["Name"].str.contains("Master\.")) & (testData["Age"] == -99), 'Age'] = 5.235294

sibsp_enc = testData["SibSp"].values
sibsp_enc = LabelEncoder().fit_transform(sibsp_enc).reshape(-1,1)
sibsp_enc2 = OneHotEncoder(categories='auto').fit_transform(sibsp_enc).toarray()
testData["SibSp_0"] = sibsp_enc2[:,0].astype(int)
testData["SibSp_1"] = sibsp_enc2[:,1].astype(int)
testData["SibSp_2"] = sibsp_enc2[:,2].astype(int)
testData["SibSp_3"] = sibsp_enc2[:,3].astype(int)
testData["SibSp_4"] = sibsp_enc2[:,4].astype(int)
testData["SibSp_5"] = sibsp_enc2[:,5].astype(int)
testData["SibSp_6"] = sibsp_enc2[:,6].astype(int)
testData["Family"] = testData["SibSp"] + testData["Parch"] + 1
family_enc = testData["SibSp"].values
family_enc = LabelEncoder().fit_transform(family_enc).reshape(-1,1)
family_enc2 = OneHotEncoder(categories='auto').fit_transform(family_enc).toarray()
testData["Family_0"] = family_enc2[:,0].astype(int)
testData["Family_1"] = family_enc2[:,1].astype(int)
testData["Family_2"] = family_enc2[:,2].astype(int)
testData["Family_3"] = family_enc2[:,3].astype(int)
testData["Family_4"] = family_enc2[:,4].astype(int)
testData["Family_5"] = family_enc2[:,5].astype(int)
testData["Family_6"] = family_enc2[:,6].astype(int)
#testData["Family_7"] = family_enc2[:,7].astype(int)
#testData["Family_8"] = family_enc2[:,8].astype(int)
testData["Cabin_val"] = testData["Cabin"].str[0:1]
testData["Cabin_A"] = testData.apply(lambda x: 1 if x["Cabin_val"] == "A" else 0, axis=1)
testData["Cabin_B"] = testData.apply(lambda x: 1 if x["Cabin_val"] == "B" else 0, axis=1)
testData["Cabin_C"] = testData.apply(lambda x: 1 if x["Cabin_val"] == "C" else 0, axis=1)
testData["Cabin_D"] = testData.apply(lambda x: 1 if x["Cabin_val"] == "D" else 0, axis=1)
testData["Cabin_E"] = testData.apply(lambda x: 1 if x["Cabin_val"] == "E" else 0, axis=1)
testData["Cabin_F"] = testData.apply(lambda x: 1 if x["Cabin_val"] == "F" else 0, axis=1)
testData["Cabin_T"] = testData.apply(lambda x: 1 if x["Cabin_val"] == "T" else 0, axis=1)
testData["Cabin_Z"] = testData.apply(lambda x: 1 if x["Cabin_val"] == "Z" else 0, axis=1)

testData.head()
# 年齢の割り当て漏れがないか確認
testData[testData["Age"] == -99]
# 検証データからを説明変数データを抽出
x_test = testData[feature_names]
# モデルからを目的変数データを算出
y_test = predictor.predict(x_test)
result = testData[["PassengerId"]].copy()
result["Survived"] = y_test
result.head()
# 結果をCSV出力
result.to_csv(path_or_buf='./submission.csv', sep=',', header=True, index=False, encoding='utf8')