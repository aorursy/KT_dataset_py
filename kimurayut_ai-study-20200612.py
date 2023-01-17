from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# アイリスデータセットを読み込む
# データの読み込み
iris = datasets.load_iris()
x, y = iris.data, iris.target

# トレーニングデータとテストデータに分ける
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

# 機械学習アルゴリズムの実装とハイパーパラメータの設定
model = DecisionTreeClassifier(criterion='entropy',max_depth=1)

# 学習
model.fit(x_train, y_train)
# テストと評価
pred = model.predict(x_test)
print(accuracy_score(y_test, pred))
# 試行するパラメータの羅列
params = {
    'max_depth': list(range(1, 20)),
    'criterion': ['gini', 'entropy'],
}

# 機械学習アルゴリズムの実装
modeli = DecisionTreeClassifier()

# cv=10は10分割の交差検証を実行
grid_search = GridSearchCV(modeli, param_grid=params, cv=10)

# 学習
grid_search.fit(x_train, y_train)

# 最も良かったスコアを出力
print(grid_search.best_score_)

# 最適なモデルを出力
print(grid_search.best_estimator_)

# 上記を記録したパラメータの組み合わせを出力
print(grid_search.best_params_)

# テストと評価
predd = grid_search.predict(x_test)
print(accuracy_score(y_test, predd))