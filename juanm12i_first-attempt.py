# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


train_path = '../input/train.csv'

test_path = '../input/test.csv'



header = 0

df_train = pd.read_csv(train_path, header = header)

df_test  = pd.read_csv(test_path, header = header)

df_total = df_train.append(df_test, ignore_index = True)



s_y_train = df_train.loc[:,'y']

# Treatment of categorical variables (X1 - X8):

cat_cols = df_train.loc[:,'X1':'X8']

cat_cols = pd.get_dummies(cat_cols)
#Treatment of Binary variables (X10-X385)

binary_cols = df_train.loc[:,'X10':'X385']



corrs = binary_cols.corrwith(df_train.loc[:,'y'], axis=0)

corrs = corrs.sort_values()

# Join data
y = df_train['y']

df_total = df_train.append(df_test, ignore_index = True)
cat_cols