# 必要なライブラリの読み込み
import numpy as np
import pandas as pd
from dateutil.parser import parse
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# グラフをjupyter Notebook内に表示させるための指定
%matplotlib inline
# データの読み込み
df_data = pd.read_csv("../input/mushrooms.csv")
# データの確認
print(df_data.columns)
print(df_data.shape)
display(df_data.head(3))
display(df_data.tail(3))
# 欠測値を確認する
df_data.isnull().sum()
# クロス集計表を作成
for col in df_data.columns:
    if col == "class":
        continue
    print(col)
    df_c = pd.crosstab(index = df_data["class"], columns = df_data[col],
                       margins = True, normalize = True)
    display(df_c)
# ダミー変数への変換
df_str = df_data.copy()
for col in df_data.columns:
    col_str = col+"-str"
    df_str[col_str] = df_data[col].astype(str).map(lambda x: col+'-'+x)
    if col == "class":
        df_en = pd.get_dummies(df_str[col_str])
    else:
        df_en = pd.concat([df_en,pd.get_dummies(df_str[col_str])], axis = 1)
print(df_en.columns)
display(df_en.head(3))
display(df_en.tail(3))
# stalk-root-?が気になるので削除する
df_en_fin = df_en.drop(["stalk-root-?"], axis = 1)
# また、散布図を基にカテゴリ内の選択肢が2つしかないものは片方を削除しておく
df_en_fin = df_en_fin.drop(["class-e","bruises-t",
                            "gill-attachment-f","gill-spacing-w",
                            "gill-size-n","stalk-shape-t"],
                           axis = 1)
display(df_en_fin.head(3))
display(df_en_fin.tail(3))
df_en_fin.corr().style.background_gradient().format('{:.2f}')
# 目的変数、説明変数をセット
y = ((df_en_fin["class-p"] > 0) * 1).values
X = df_en_fin[["bruises-f","odor-n","gill-size-b",
               "gill-color-b","stalk-surface-above-ring-k",
               "stalk-surface-below-ring-k","ring-type-p"]]
# ロジスティック回帰を実施
lr = LogisticRegression()
lr.fit(X,y)
# モデルの精度を確認
print(lr.coef_,lr.intercept_)
y_pred = lr.predict(X)
print(classification_report(y,y_pred))