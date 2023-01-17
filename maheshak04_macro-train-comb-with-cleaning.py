# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')

macro_df=pd.read_csv('../input/macro.csv')
import matplotlib.pyplot as plt

import seaborn as sns

train_df.columns
plt.scatter(train_df.full_sq, train_df.price_doc, c = "blue", marker = "s")

plt.title("Looking for outliers")

plt.xlabel("full_sq")

plt.ylabel("price_doc")

plt.show()
train_df.loc[train_df[train_df.full_sq>=5000].id.values]
macro_df.columns.T
(macro_df.isnull().sum()/len(macro_df)*100).sort_values(ascending=False)
full_df=pd.concat([train_df,macro_df])
full_df.columns
full_df.shape
train_df.shape
full_df.groupby(['gdp_annual_growth'])['price_doc'].agg([np.median, np.mean, np.std])
full_df['gdp_annual_growth'].isnull().sum()