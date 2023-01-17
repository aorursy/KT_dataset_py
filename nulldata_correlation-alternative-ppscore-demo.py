import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

sns.set(rc={'figure.figsize':(20,20)})
!pip install ppscore
import ppscore as pps
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train.columns
%%time
mat = pps.matrix(train)
sns.heatmap(mat, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
%%time
mat = pps.matrix(train[['OverallQual', 'OverallCond', 'YearBuilt','EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'SalePrice']])
sns.heatmap(mat, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
pps.score(train,'YearBuilt','OverallQual')
%%time
corr = train.corr()
sns.heatmap(corr, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=False)
