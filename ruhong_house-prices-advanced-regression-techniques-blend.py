# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read the test data

target = 'SalePrice'

rf = pd.read_csv('../input/house-prices-advanced-regression-techniques-rf/submission.csv')

lasso = pd.read_csv('../input/house-prices-advanced-regression-techniques-lasso/submission.csv')

gbm = pd.read_csv('../input/house-prices-advanced-regression-techniques-gbm/submission.csv')

gbm_pca = pd.read_csv('../input/house-prices-advanced-regression-techniques-gbmpca/submission.csv')

submission = rf.copy()

submission[target] = (0.2 * rf[target]) + (0.2 * lasso[target]) + (0.4 * gbm[target]) + (0.2 * gbm_pca[target])

submission.head()
submission.to_csv('submission.csv', index=False)

print(os.listdir("."))