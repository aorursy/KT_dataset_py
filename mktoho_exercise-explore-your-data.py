# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex2 import *
print("Setup Complete")
import pandas as pd

# 読み込むファイルのパス
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# ファイルをhome_data変数に読み込むために、下の行に記入してください
home_data = _

# 下の行を引数なしで呼び出し、データを正しくロードしたことを確認してください
step_1.check()
# 下の行をコメント解除して実行すると、ヒントまたは解決コードを表示します
# step_1.hint()
# step_1.solution()
# 統計の要約を次の行に出力する
_
# 平均ロットサイズ（最も近い整数に丸められた）は何ですか？
avg_lot_size = _

# 今日では、最新の家は築何年が経過していますか（今年 - それが建設された日）
newest_home_age = _

# Checks your answers
step_2.check()
#step_2.hint()
#step_2.solution()