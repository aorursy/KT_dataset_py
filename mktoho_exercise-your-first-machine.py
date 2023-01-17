# 前の章でデータを読み込むために使用したコード
import pandas as pd

# ファイルのパス
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# コードチェックを設定する
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *

print("Setup Complete")
# データセット内の列のリストを出力して、予測ターゲットの名前を探します。

#y = _

step_1.check()
# 下の行を実行するとヒントや解決方法が表示されます。
# step_1.hint() 
# step_1.solution()

# 以下のフィーチャーのリストを作成する
# feature_names = ___

# feature_namesのフィーチャに対応するデータを選択する
#X = _

step_2.check()
# step_2.hint()
# step_2.solution()
# データの確認
# Xの概要や統計を出力する
#print(_)

# 先頭の数行を出力する
#print(_)

# from _ import _
# モデルを定義する 
# モデルの再現性を確認するには、モデルを指定するときにrandom_stateの数値を設定します
# iowa_model = _

# モデルにフィットさせる
_

step_3.check()
# step_3.hint()
# step_3.solution()
predictions = _
print(predictions)
step_4.check()
# step_4.hint()
# step_4.solution()