# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/HR_comma_sep.csv')
train_df.head()
y_train = train_df.pop('left')
y_train.head()
train_df.shape
pd.get_dummies(train_df['salary'], prefix='salary').head()
all_dummy_df = pd.get_dummies(train_df)

all_dummy_df.head()
numeric_cols = train_df.columns[train_df.dtypes != 'object']

numeric_cols
numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()

numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()

all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std

dummy_train_df = all_dummy_df.loc[train_df.index]

dummy_test_df = all_dummy_df.loc[y_train.index]
dummy_train_df.shape, dummy_test_df.shape
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score


X_train = dummy_train_df.values

X_test = dummy_test_df.values
alphas = np.logspace(-3, 2, 500)

test_scores = []

for alpha in alphas:

    clf = Ridge(alpha)

    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))

    test_scores.append(np.mean(test_score))
import matplotlib.pyplot as plt

%matplotlib inline

plt.plot(alphas, test_scores)

plt.title("Alpha vs CV Error");