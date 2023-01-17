print('hello world')
import pandas as pd # pandasのインポート



# csvデータの読み込み

df = pd.read_csv('../input/gender_submission.csv')
# dfをsubmission.csvとしてcsvファイルに書き出し

df.to_csv('submission.csv', index=False)