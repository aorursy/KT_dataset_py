import pandas as pd
import numpy as np

df_fb_train = pd.read_csv('../input/train.csv')
df_fb_test = pd.read_csv('../input/test.csv')

print('Size of training data: ' + str(df_fb_train.shape))
print('Size of testing data:  ' + str(df_fb_test.shape))

print('\nColumns:' + str(df_fb_train.columns.values))

print(df_fb_train.describe())


print(df_fb_train.unique())