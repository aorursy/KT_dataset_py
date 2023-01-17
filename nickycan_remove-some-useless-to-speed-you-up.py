# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

def U_Get_NC_col_names(data):

    '''Get column names of category and numeric

    

    Parameters

    ----------

    data: dataframe

    

    Return:

    ----------

    numerics_cols: numeric column names

    category_cols: category column names

    

    '''

    numerics_cols = data.select_dtypes(exclude=['O']).columns.tolist()

    category_cols = data.select_dtypes(include=['O']).columns.tolist()

    return numerics_cols, category_cols
train = pd.read_csv("../input/train.csv")

train.columns = map(str.lower, train.columns)

y = train.y

train.drop('y', axis=1, inplace=True)





test = pd.read_csv("../input/test.csv")

test.columns = map(str.lower, test.columns)
plt.plot(y,'.',c='red', alpha=0.2)
numerics_cols, category_cols = U_Get_NC_col_names(train)



train_constant_cols = []

test_constant_cols = []



print("constant col in train")

for col in numerics_cols:

    if(train[col].nunique() == 1):

        print(col)

        train_constant_cols += [col]

print("----------")

print("constant col in test")

for col in numerics_cols:

    if(test[col].nunique() == 1):

        print(col)

        test_constant_cols += [col]



df_constant_cols = test_constant_cols + train_constant_cols

# I choose not to delete x258 because there are ten 1 in train data.

df_constant_cols.remove('x258')