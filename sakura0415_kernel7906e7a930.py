import pandas
import json
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import os

def sumWords(data):
    # 材料名を集約
    words = []
    for i in data:
        words.extend(i['ingredients'])

    # 材料名を一意に
    return list(set(words))

def addColumn(data, mt):
    # 連想配列にカラムを作成。
    # 各材料をキーにしてレコードが材料を持っていたら値を1に設定
    print(len(data))
    print(len(mt))
    for i in data:
        for ii in mt:
            if len(list(filter( lambda x: x == ii, i['ingredients']))) > 0:
                i[ii] = 1
            else:
                i[ii] = 0

def ml_tree(df , depth):
    return tree.DecisionTreeClassifier(max_depth=depth).fit(df.drop('cuisine', axis=1), df.cuisine)

def ml_rf(df , depth):
    return RandomForestClassifier(max_depth=depth).fit(df.drop('cuisine', axis=1), df.cuisine)

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# データ読み込み
f = open('../input/train.json', 'r')
jsonData = json.load(f)[:25000]
mt = sumWords(jsonData)
# テスト
t = open('../input/train.json', 'r')
testData = json.load(t)[30000:35000]
addColumn(testData,mt)
addColumn(jsonData, mt)
df = pandas.DataFrame.from_dict(jsonData)
df = df.drop( ['id','ingredients'], axis=1)

tdf = pandas.DataFrame.from_dict(testData)
tdf = tdf.drop(['id','ingredients'], axis=1)
td = tdf.drop('cuisine', axis=1)
for i in range(10, 50):
    tree_model = ml_tree(df, i)
    rf_model = ml_rf(df, i)
    tr_result = sum(tree_model.predict(td) == tdf.cuisine) / len(tdf.cuisine)
    rf_result = sum(rf_model.predict(td) == tdf.cuisine) / len(tdf.cuisine)
    print('depth:{0},tree:{1},rf:{2}'.format(i,tr_result,rf_result))