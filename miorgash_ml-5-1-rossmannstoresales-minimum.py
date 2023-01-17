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
import pandas as pd



train_file = '../input/rossmann-store-sales/train.csv'

test_file = '../input/rossmann-store-sales/test.csv'

output_file = 'predictions.csv'



train = pd.read_csv(train_file, low_memory=False)

test = pd.read_csv(test_file, low_memory=False)



print('train データ\n'

      'shape', train.shape)

display(train.head())



print('test データ：Sales だけ欠けているのでこれを予測する（回帰）．\n'

      'shape', test.shape)

display(test.head())





train_means = train.groupby([ 'Store', 'DayOfWeek', 'Promo' ])['Sales'].mean().reset_index()

print("とりあえず適当にグルーピングした平均値を予測値とすることにする．\n"

      "train データを店，曜日，プロモーション有無で集計\n",

      'shape', train_means.shape)

display(train_means.head())



print("test データに結合")

test = pd.merge(test, train_means,

                on = ['Store','DayOfWeek','Promo'], how='left')

test.fillna(train.Sales.mean(), inplace=True)

display(test.head())



print('提出用に Id, Sales だけ抜き出す')

submission = test[[ 'Id', 'Sales' ]]

display(submission.head())



# 出力

submission.to_csv(output_file, index = False )