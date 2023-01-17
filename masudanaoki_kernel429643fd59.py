# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 訓練データとテストデータの読み込み
df_train = pd.read_csv('../input/nlp-getting-started/train.csv', dtype={'id': np.int16, 'target': np.int8})
df_test = pd.read_csv('../input/nlp-getting-started/test.csv', dtype={'id': np.int16})

# 訓練データとテストデータの行数・列数を表示する
print('Training Set Shape = {}'.format(df_train.shape))
print('Test Set Shape = {}'.format(df_test.shape))

# 訓練データの中から10行をランダムで抽出
df_train.sample(n=10, random_state=28)
# 訓練データ、テストデータの各カラムの欠損値率を算出
print("missing-value ratio of training data(%)")
print(df_train.isnull().sum()/df_train.shape[0]*100)
print("\nmissing-value ratio of test data(%)")
print(df_test.isnull().sum()/df_test.shape[0]*100)
# ターゲットの要素とその個数をプロット
target_vals = df_train.target.value_counts()
sns.barplot(target_vals.index, target_vals)
plt.gca().set_ylabel('samples')