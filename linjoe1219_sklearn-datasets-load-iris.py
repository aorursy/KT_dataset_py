from sklearn.datasets import load_iris
iris = load_iris()
iris
import pandas as pd
df = pd.DataFrame(iris["data"], columns=iris["feature_names"])
df["target"] = iris["target"]
df
'''
畫出相關係數圖，用熱度圖表示
相關係數圖在回歸很準，在分類不一定
'''
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.figure(figsize=(8, 8)) # 改圖的大小

# df.astype("float") 單位轉成浮點數
# df.corr() 相關係數矩陣
# cmap改顏色，可以去matplotlib網站查
# annot=True:把係數標上
sns.heatmap(df.astype("float").corr(), cmap="PuBuGn", annot=True)

df.corr()
from sklearn.model_selection import train_test_split
# train_test_split -> (特徵90%, 特徵10%, 目標90%, 目標10%)
# 把target那一列丟掉，axis=1代表列
x_train, x_test, y_train, y_test = train_test_split(df.drop(["target"], axis=1), df["target"], test_size=0.1)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(x_train, y_train)
from sklearn.tree import export_graphviz
import graphviz
g = export_graphviz(clf, out_file=None, feature_names=iris["feature_names"],
                    class_names=iris["target_names"], filled=True, special_characters=True)                
graph = graphviz.Source(g)
graph
pre = clf.predict(x_test)
print("預測結果:", list(pre))
print("真正標籤:", list(y_test))
from sklearn.metrics import accuracy_score
print("預測成功機率:", str(accuracy_score(pre, y_test)*100)+'%')
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pre)
# 矩陣的(0,0), (1,1), (2,2)代表預測正確數量，其他代表預測錯誤數量
pd.DataFrame(confusion_matrix(y_test, pre))