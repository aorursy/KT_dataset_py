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

pd.options.display.max_columns =1000
pd.options.display.max_rows=1000
train_data=pd.read_csv("../input/train.csv")

test_data=pd.read_csv("../input/test.csv")
submission_file=pd.read_csv("../input/sample_submission.csv")
submission_file.head()
train_data.SalePrice.mean()
submission_file['SalePrice']=train_data.SalePrice.mean()
submission_file.head()
abc=train_data.groupby('YearBuilt').mean()['SalePrice'].reset_index()
submission_file.to_csv('first_submission_avg.csv', index=False)
#index=False because we do not want a new column of iundex in csv file
abc
abc['new_sale_price']=abc.SalePrice.shift(1)
abc
abc['age']=abc['SalePrice']-abc['new_sale_price']


