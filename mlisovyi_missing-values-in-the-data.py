import numpy as np # linear algebra
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
#plt.xkcd()
import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
train.info()
train_null = train.isnull().sum()
train_null_non_zero = train_null[train_null>0] / train.shape[0]
train_null_non_zero
sns.barplot(x=train_null_non_zero, y=train_null_non_zero.index)
_ = plt.title('Fraction of NaN values, %')
test_null = test.isnull().sum()
test_null_non_zero = test_null[test_null>0] / test.shape[0]
sns.barplot(x=test_null_non_zero, y=test_null_non_zero.index)
_ = plt.title('Fraction of NaN values in TEST data, %')
