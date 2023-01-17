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
macro_df=pd.read_csv('../input/macro.csv')

train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train_df.columns
test_df.columns
macro_df.columns
train_df.price_doc.describe()
#histogram

sns.distplot(train_df['price_doc']);
#scatter plot grlivarea/saleprice

var = 'full_sq'

data = pd.concat([train_df['price_doc'], train_df[var]], axis=1)

data.plot.scatter(x=var, y='price_doc', ylim=(0,800000));
#scatter plot grlivarea/saleprice

var = 'life_sq'

data = pd.concat([train_df['price_doc'], train_df[var]], axis=1)

data.plot.scatter(x=var, y='price_doc', ylim=(0,800000));
#scatter plot grlivarea/saleprice

for var in train_df.columns:

    v=var

    data = pd.concat([train_df['price_doc'], train_df[v]], axis=1)

    data.plot.scatter(x=v, y='price_doc', ylim=(0,800000));
#correlation matrix

corrmat = train_df.corr()

f, ax = plt.subplots(figsize=(120, 90))

sns.heatmap(corrmat, vmax=.9, square=True);
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'price_doc')['price_doc'].index

cm = np.corrcoef(train_df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
train_df.corr().price_doc.sort_values(ascending=False)
comb_df=pd.concat([train_df,macro_df],axis=0)
len(comb_df)
comb_df.columns
comb_df.shape
comb_df.corr().price_doc.sort_values(ascending=False)
missing = train_df.isnull().sum()

missing = missing[missing <4000]

missing.sort_values(inplace=True)

missing.plot.bar()
missing_df=pd.DataFrame(missing,columns=missing.name)
missing_df.index