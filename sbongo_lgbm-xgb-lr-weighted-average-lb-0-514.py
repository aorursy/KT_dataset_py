# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#OKAY! Best score is 0.600748907457

#OKAY! Best weights are 0.45   0.45   0.1  



lrSubmitY= pd.read_csv('../input/aggr-dataset/lrSubmitY.csv',index_col=False)

lrSubmitY.drop(['Unnamed: 0'], inplace=True,axis=1)
lgbSubmitY= pd.read_csv('../input/dataset2/lgbSubmitY.csv',index_col=False)

lgbSubmitY.drop(['Unnamed: 0'], inplace=True,axis=1)
xgbSubmitY= pd.read_csv('../input/aggr-dataset/xgbSubmitY.csv',index_col=False)

xgbSubmitY.drop(['Unnamed: 0'], inplace=True,axis=1)
combinedPredY = (xgbSubmitY*0.45)+(lgbSubmitY*0.45)+(lrSubmitY*0.10)
combinedPredY.values
#to reproduce the testing IDs

df_train = pd.read_csv(

    '../input/favorita-grocery-sales-forecasting/train.csv', usecols=[1, 2, 3, 4, 5],

    dtype={'onpromotion': bool},

    converters={'unit_sales': lambda u: np.log1p(

        float(u)) if float(u) > 0 else 0},

    parse_dates=["date"],

    skiprows=range(1, 66458909)  # 2016-01-01

)



df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)]

del df_train



df_2017 = df_2017.set_index(

    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(

        level=-1).fillna(0)

df_2017.columns = df_2017.columns.get_level_values(1)
combinedPredY.shape
print("Making submission...")

#y_test = np.array(test_pred).transpose()

#pd.DataFrame(y_test).to_csv('predY_before_stack.csv')

df_preds = pd.DataFrame(

    combinedPredY.values, index=df_2017.index,

    columns=pd.date_range("2017-08-16", periods=16)

).stack().to_frame("unit_sales")

df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)
import gc

gc.collect()
df_test = pd.read_csv(

    "../input/favorita-grocery-sales-forecasting/test.csv", usecols=[0, 1, 2, 3, 4],

    dtype={'onpromotion': bool},

    parse_dates=["date"]  # , date_parser=parser

).set_index(

    ['store_nbr', 'item_nbr', 'date']

)
submission = df_test[["id"]].join(df_preds, how="left").fillna(0)

submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)

submission.to_csv('aggr.csv', float_format='%.4f', index=None)