# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import statsmodels.formula.api as smf

def train_my_model(train_df):

    #平均・標準偏差・null数を取得する

    Dbox_average = train_df["DefendersInTheBox"].mean() #平均値

    Dbox_std = train_df["DefendersInTheBox"].std()  #標準偏差

    Dbox_nullcount = train_df["DefendersInTheBox"].isnull().sum() #null値の数＝補完する数



    # 正規分布に従うとし、標準偏差の範囲内でランダムに数字を作る

    rand = np.random.randint(Dbox_average - Dbox_std, Dbox_average + Dbox_std , size = Dbox_nullcount)



    #Ageの欠損値

    train_df["DefendersInTheBox"][np.isnan(train_df["DefendersInTheBox"])] = rand



#     train_df.DefendersInTheBox[train_df.DefendersInTheBox.isna()] = train_df.DefendersInTheBox.mean()

    result=smf.ols("Yards ~ A + S + DefendersInTheBox + Distance", data=train_df).fit()

    return result
from kaggle.competitions import nflrush

env = nflrush.make_env()
# Training data is in the competition dataset as usual

train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

Mean_DefendersInTheBox = train_df.DefendersInTheBox.mean()

result_model = train_my_model(train_df)
def make_my_predictions(result_model, test_df, sample_prediction_df):

    test_df = test_df.query("NflId==NflIdRusher")

    test_df.DefendersInTheBox[test_df.DefendersInTheBox.isna()] = Mean_DefendersInTheBox

    out_yard = result_model.predict(test_df)

    num = int(out_yard//1)

    sample_prediction_df.iloc[:,1:] = 0

    sample_prediction_df.loc[:,"Yards%s" % str(num):] = 1

#     output = pd.DataFrame({}, columns=sample_prediction_df.columns)

#     for idx,i in enumerate(num_lst):

#         each_prediction_df = sample_prediction_df.copy()

#         each_prediction_df["PlayId"] = test_df.iloc[idx,0]

#         each_prediction_df.loc[:,"Yards%s" % str(i):] = 1

#         output = pd.concat([output, each_prediction_df])

    return sample_prediction_df
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    predictions_df = make_my_predictions(result_model, test_df, sample_prediction_df)

    env.predict(predictions_df)
env.write_submission_file()

# make_my_predictions(result_model, test_df, sample_prediction_df)
sample_prediction_df
# env.predict(make_my_predictions(result_model, test_df, sample_prediction_df))