# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.max_rows = 999

pd.options.display.max_columns = 999

import missingno as msno 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
X_train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv', index_col='id')

X_test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv', index_col='id')



cat_df = X_train.append(X_test, sort=False)

cat_df.drop('target', axis=1, inplace=True)



cat_df.head()
cat_df.info()
cols = list(cat_df.columns)

cols



for col in cols:

    temp = cat_df[col].unique()

    print('Column Name: ', col)

    print('Column Unique Values: ', temp)
cat_df.describe()
for col in cols:

    print('Column Name: ', col, ' ', sum(cat_df[col].isnull()), ' is missing.')

    
msno.matrix(cat_df)
msno.heatmap(cat_df)
msno.dendrogram(cat_df)