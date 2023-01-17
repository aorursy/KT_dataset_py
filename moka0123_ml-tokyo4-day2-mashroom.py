# 必要なライブラリの読み込み
# 数値計算用
import numpy as np
import pandas as pd
# 描画用
from IPython.display import display,Image
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import graphviz
import pydotplus
# 機械学習用
from sklearn.externals.six import StringIO
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier,export_graphviz
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
# stalk-root-?が気になるので削除する
df_en_fin = df_en.drop(["stalk-root-?"], axis = 1)
# また、クロス集計表を基にカテゴリ内の選択肢が2つしかないものは片方を削除しておく
df_en_fin = df_en_fin.drop(["class-e","bruises-t",
                            "gill-attachment-f","gill-spacing-w",
                            "gill-size-n","stalk-shape-t"],
                           axis = 1)
# データの表示
print(df_en_fin.columns)
display(df_en_fin.head(3))
display(df_en_fin.tail(3))
# 相関係数を求める
df_en_fin.corr().style.background_gradient().format('{:.2f}')
# 説明変数を抽出
df_exp = df_en_fin[["bruises-f","odor-f","odor-n","gill-size-b",
               "gill-color-b","stalk-surface-above-ring-k",
               "stalk-surface-below-ring-k","ring-type-p"]]
# 多重共線性の確認
for cname in df_exp.columns:  
    y=df_exp[cname]
    X=df_exp.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    # VIFの計算
    VIF = 1/(1-np.power(rsquared,2))
    print(cname,":VIF[", VIF.round(3), "]")
    # 決定係数の確認
    if rsquared >= np.sqrt(0.9):
        print(cname,X.columns[(regr.coef_> 0.5) | (regr.coef_ < -0.5)])    
# 目的変数、説明変数をセット
y = ((df_en_fin["class-p"] > 0) * 1).values
X = df_en_fin[["bruises-f","odor-f","odor-n","gill-size-b",
               "gill-color-b","stalk-surface-above-ring-k",
               "stalk-surface-below-ring-k","ring-type-p"]]
# テストデータと検証データに分割
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=1234
)
# ロジスティック回帰を実施
lr = LogisticRegression()
lr.fit(X_train,y_train)
# モデルの精度を確認
print(lr.coef_,lr.intercept_)
y_pred = lr.predict(X_test)
print(classification_report(y_test,y_pred))
# 目的変数、説明変数をセット
y = ((df_en_fin["class-p"] > 0) * 1).values
X = df_en_fin.drop("class-p", axis=1)
# テストデータを検証データに分割
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=1234
)
# 決定木を実施
clf = DecisionTreeClassifier(criterion="gini", max_depth=3
                             , min_samples_split=3, min_samples_leaf=3, random_state=1234)
clf = clf.fit(X_train, y_train)
# スコアの確認
print("score=", clf.score(X_test,y_test).round(3))
# 決定木の描画
dot_data = StringIO() #dotファイル情報の格納先
export_graphviz(clf, out_file=dot_data,  
                     feature_names=X_train.columns,  
                     class_names=["0","1"],   # 0:class-e 1:class-p
                     filled=True, rounded=True,  
                     special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())
# 木の深さを指定
param_grid = {'max_depth':[3,5,6,7,8,9,11]}
cv = GridSearchCV(DecisionTreeClassifier(),param_grid=param_grid,cv=5)
cv.fit(X_train, y_train)
# 最適なパラメータを確認
cv.best_params_
# CVのスコアを確認
cv.grid_scores_
# 木の深さが6の場合
print(classification_report(y_test,cv.best_estimator_.predict(X_test)))
print(confusion_matrix(y_test,cv.best_estimator_.predict(X_test)))
# 決定木の描画
dot_data = StringIO() #dotファイル情報の格納先
export_graphviz(cv.best_estimator_, out_file=dot_data,  
                     feature_names=X_train.columns,  
                     class_names=["0","1"],   # 0:class-e 1:class-p
                     filled=True, rounded=True,  
                     special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())
