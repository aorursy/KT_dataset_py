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
import pandas as pd

df = pd.read_csv("../input/train.csv")

colnames = df.columns.values.tolist()

print (colnames)

df.head(n=5)
print(colnames)
cat_var = [x for x in colnames if x[0:3] == 'cat' ]

print(cat_var)
dummy_df = pd.get_dummies(df[cat_var], prefix = cat_var) 

dummy_df.head(n=5)
df_dev = pd.concat([df,dummy_df],axis = 1)

df_dev.drop(cat_var,axis=1,inplace=True)

df_dev.drop('id',axis=1,inplace=True)

df_dev.head(n=4)

dev_colnames = df_dev.columns.values.tolist()

print(dev_colnames)
import xgboost as xgb



dtrain = xgb.DMatrix(df_dev, label=loss)