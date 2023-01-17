from sklearn import datasets
from sklearn import svm
# Irisが持っているデータの内容とIrisの形状を確認します
# Irisの測定データの読み込み
iris = datasets.load_iris()

print(iris.data)
print(iris.data.shape) # 形状

print(iris.target)
list(iris.target_names)
# データの長さを調べてみます。
num = len(iris.data)
print(num)
# サポートベクターマシンのアルゴリズムを実装します
# ここで、svm.SVC()とfit()に関して説明します。
# まず、svm.SVC()は、SVM(サポートベクターマシン)というアルゴリズムです。
# Scikit-learnでは分類に関するSVMは3種類(SVC,LinearSVC,NuSVC)用意されています。
# その中のSVC()を利用します。
# 次に、fit()ですが、fit()を使う事で学習(機械学習)が行えます。
# fit()の第1引数に特徴量Xを与え、第2引数にラベルデータYを与え利用します。
clf = svm.SVC(gamma="auto")
clf.fit(iris.data, iris.target)

# 作成したモデルで予測をする
# 【与えたデータ】
# がくの長さが1.4
# がくの幅が1.8
# 花びらの長さが3.9
# 花びらの幅が0.5
print(clf.predict([[1.4, 1.8, 3.9, 0.5]]))