import pandas as pd

import matplotlib.pyplot as plt 
! ls -a
!ls ../input/allstate-claims-severity -a
train = pd.read_csv('../input/allstate-claims-severity/train.csv')

test = pd.read_csv('../input/allstate-claims-severity/test.csv')
print('train.shape:',train.shape)

print('test.shape:',test.shape)
pd.set_option('display.max_columns', len(train.columns))
train.head(3)
train.tail(3)
test.head(3)
test.tail(3)
train.describe()
test.describe()
train.isnull().sum().sum()
test.isnull().sum().sum()
def calc_col_str_ratio(df1, df2):

    

    df1_col = df1.select_dtypes(include=object).columns

    df2_col = df2.select_dtypes(include=object).columns

    

    col_dict = {}

    

    for col1, col2 in zip(df1_col, df2_col):

        df = pd.concat([df1[col1].value_counts(normalize = True), df2[col2].value_counts(normalize = True)], axis = 1)

        

        df.rename(columns = {col1:col1 + '_train', col2:col2 + '_test'}, inplace = True)

        diff_colname = col1 + '_diff'

        df[diff_colname] = df.diff(-1, axis=1).iloc[:,0]

        

        col_dict[col1] = df

        

    return col_dict
col_str_ratio = calc_col_str_ratio(train, test)

print(col_str_ratio)
train.hist(bins = 30, figsize = (15, 10))
test.hist(bins = 30, figsize = (15, 10))