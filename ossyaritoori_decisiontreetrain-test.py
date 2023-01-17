import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
sample_submission = pd.read_csv('../input/titanic/gender_submission.csv')
# 分類する

#maleを0に、femaleを1に変換
train["Sex"] = train["Sex"].map({"male":0,"female":1})
test["Sex"] = test["Sex"].map({"male":0,"female":1})
# EmbarkedのOne-Hotエンコーディング
train = pd.get_dummies(train, columns=['Embarked'])
test = pd.get_dummies(test, columns=['Embarked'])


# 不要な列の削除
train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

# trainの表示
display(train.head())
# NaN の存在確認 と 除去
print(train.isnull().sum())
train2 = train.dropna()

print(test.isnull().sum())
test2 = test.dropna()

# Fill with median 
train3 = train.fillna(train.median())
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X_train = train3.drop(['Survived'], axis=1)  # X_trainはtrainのSurvived列以外
y_train = train3['Survived']  # Y_trainはtrainのSurvived列

# X_trainとY_trainをtrainとvalidに分割
train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.20, random_state=0)
import sklearn.tree as tree

# 分類木だからClassifier，Regressorもある
clf = tree.DecisionTreeClassifier(max_depth=4)

# データを用いて学習
model = clf.fit(train_x, train_y)

# データを用いて予測
predicted = model.predict(valid_x)
print(accuracy_score(predicted,valid_y))
# 3分割交差検証を指定し、インスタンス化
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True, random_state=0)

# スコアとモデルを格納するリスト
score_list = []
models = []

# 各分割ごとに評価
for fold_, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):    
    print(f'fold{fold_ + 1} start')
    train_x = X_train.iloc[train_index]
    valid_x = X_train.iloc[valid_index]
    train_y = y_train.iloc[train_index]
    valid_y = y_train.iloc[valid_index]
    
    ## 分割データで学習・予測・評価
    clf = tree.DecisionTreeClassifier(max_depth=4)
    model = clf.fit(train_x, train_y)
    
    # データを用いて予測，記録
    predicted = model.predict(valid_x)
    score_list.append(accuracy_score(predicted,valid_y))
    models.append(model)
print(score_list, '平均score', round(np.mean(score_list), 3))
# 図示その１
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(30, 30))  # whatever size you want
tree.plot_tree(model, ax=ax)
plt.show()
!pip install dtreeviz pydotplus
# graphvizによる視覚化
import pydotplus as pdp

file_name = "./tree_visualization.png"
dot_data = tree.export_graphviz(model, # 決定木オブジェクトを一つ指定する
                                out_file=None, # ファイルは介さずにGraphvizにdot言語データを渡すのでNone
                                filled=True, # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
                                rounded=True, # Trueにすると、ノードの角を丸く描画する。
                                feature_names=train_x.columns, # これを指定しないとチャート上で特徴量の名前が表示されない
                                class_names=['Survived','Dead'], # これを指定しないとチャート上で分類名が表示されない
                                special_characters=True # 特殊文字を扱えるようにする
                                )
graph = pdp.graph_from_dot_data(dot_data)
graph.write_png(file_name)
# dtreevizによる視覚化
from dtreeviz.trees import dtreeviz

viz = dtreeviz(
    model,
    train_x, 
    train_y,
    target_name='alive',
    feature_names=train_x.columns,
    class_names=['survived','dead']
) 

viz.save("dtreeviz.svg")
from sklearn.metrics import confusion_matrix

# 混同行列の作成
cmatrix = confusion_matrix(valid_y,predicted)

#pandasで表の形に
df = pd.DataFrame(cmatrix,index=["actual_died","actual_survived"],columns=["pred_died","pred_survived"])

print(df)
# 重要度を表示
print(dict(zip(train_x.columns, model.feature_importances_)))

# bar plot
fig, ax = plt.subplots()
plt.grid()
ax.bar(train_x.columns,model.feature_importances_)
fig.autofmt_xdate() # make space for and rotate the x-axis tick labels
plt.show()
# テスト  これを提出
## 訓練データすべて使う
model = clf.fit(X_train, y_train)

## test の中身
print(test.isnull().sum())
## 中央値で埋める
test = test.fillna(test.median())

## 予測結果
test_predicted = model.predict(test)

## 提出用データ
sample_submission['Survived'] = test_predicted
sample_submission.to_csv('submission.csv',index=False)
