# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_gaussian_quantiles

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import BernoulliNB

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.linear_model import SGDClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_creditcard = pd.read_csv('../input/creditcard.csv')

df_creditcard.info()
print(df_creditcard.Class.unique())

print(len(df_creditcard.loc[df_creditcard['Class'] == 1]))

print(len(df_creditcard.loc[df_creditcard['Class'] == 0]))

print(df_creditcard.head())
df_train = df_creditcard[df_creditcard.columns][1000:]

df_test = df_creditcard[df_creditcard.columns][:1000]

print('# froud count in test data:',len(df_test.loc[df_test['Class'] == 1]))

df_train.head()
c1_filter = df_train.loc[df_train['Class'] == 1]

c0_filter = df_train.loc[df_train['Class'] == 0]
plt.scatter(c0_filter.V1,c0_filter.V15, color = "b")

plt.scatter(c1_filter.V1,c1_filter.V15, color = "r")