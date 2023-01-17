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
import pandas as pd
df = pd.read_csv('../input/train.csv')
df.head()
df
import seaborn as sns
pp = df[['YearBuilt', 'SalePrice']].reset_index()
sns.barplot(x = 'YearBuilt', y = 'SalePrice', data =pp)
pp1 = df.groupby(['YearBuilt']).mean()['SalePrice'].reset_index().sort_values('YearBuilt',ascending=False)
sns.scatterplot(x = 'YearBuilt', y = 'SalePrice', data =pp1);
pp1 = pp1.assign(ShiftedPrice=pp1.SalePrice.values)
# pp1
pp1.ShiftedPrice = pp1.SalePrice.shift(-1)
pp1.head()
pp1['% Change'] = (pp1['SalePrice'] - pp1['ShiftedPrice']) / pp1['ShiftedPrice']
pp1.head()
pp1['% Change'].mean()
pp1['Average Price'] = (pp1['SalePrice'] + (pp1['ShiftedPrice'] * pp1['% Change'])/100)
pp1.head()
# df = pd.read_csv('../input/train.csv')
submission_data = pd.read_csv('../input/sample_submission.csv')
submission_data.head()
submission_data['SalePrice'] =  pp1['Average Price'].values[0]
submission_data.tail()
submission_data.to_csv('submission_data_csv.csv', index = False)
import pandas_profiling
pandas_profiling.ProfileReport(df)
df_MSZoning = df.groupby(['MSZoning']).mean()['SalePrice'].reset_index().sort_values('MSZoning',ascending=False)
sns.scatterplot(x = 'MSZoning', y = 'SalePrice', data =df_MSZoning);

