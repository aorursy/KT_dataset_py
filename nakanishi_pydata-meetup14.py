# データ整形と可視化で必要となるライブラリを読み込む

# 数値計算用ライブラリ
import numpy as np

# データ解析用ライブラリ
import pandas as pd

# 基本の描画ライブラリ（２つ）
import matplotlib.pyplot as plt
import seaborn as sns
# 便利な設定

# pandasで全ての列を表示
pd.options.display.max_columns = None

# 図をipython notebook内で表示
%matplotlib inline

# DeplicatedWarningを避けるため
import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
# データの読み込み
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_sub = pd.read_csv("../input/sample_submission.csv")
df_train.rename(columns={'AnimalID': 'ID'}, inplace=True)
df_train.set_index('ID', inplace=True)
df_test.set_index('ID', inplace=True)
# 後でデータを分離しやすいよう、結合前にトレーニングデータとテストデータがわかるようなラベルを振っておく
df_train['_data'] = 'train'
df_test['_data'] = 'test'
# データセットを結合する
df = pd.concat([df_train, df_test], axis=0)
# 一応、データの形をチェック
print(df.shape)
print(df_train.shape)
print(df_test.shape)
outcome_labels = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
outcome2id = dict(zip(*[outcome_labels, np.arange(5)]))
outcome2id
# zip()の使用例
zip(*[['a', 'b'], [1, 2]])
# 最終的に予測するターゲット
df['OutcomeTypeId'] = df['OutcomeType'].map(outcome2id)
# 使わない変数はこのリストに保持
# 後で、df.drop(not_needed, axis=1, inplace=True) とすることで、いらない列を落とせる。
# 予測に使用する変数が増えて来たときに活躍する。
# 今回は簡単のため、OutcomeSubtypeは無視する。
not_needed = ['OutcomeType', 'OutcomeSubtype']
df['AnimalType']
df[['AnimalType']].info()
df['AnimalType'].value_counts()
pt = df.pivot_table(values='_data', columns='AnimalType', index='OutcomeType', aggfunc=lambda x: len(x))
# マジックコマンド（IPythonのなかだけで使えるコマンド）
%matplotlib inline
pt.plot()
pt2 = pt / pt.sum()
pt2.plot()
animal_type2id = {'Dog': 0, 'Cat': 1}
df['AnimalTypeId'] = df['AnimalType'].map(animal_type2id)
not_needed.append('AnimalType')
df
not_needed.append('SexuponOutcome')
output_feature = 'OutcomeTypeId'
input_features = df.columns.difference(not_needed + [output_feature, '_data'])
not_needed