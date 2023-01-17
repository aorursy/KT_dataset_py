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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder



pd.set_option('display.max_columns',400)
train_df = pd.read_csv("../input/train.csv")

n_train_df = train_df.copy()

enc = LabelEncoder()
for col in n_train_df.columns:

    if n_train_df[col].dtypes == 'object':

        n_train_df[col] = enc.fit_transform(n_train_df[col])
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
X_train,y =  n_train_df[list(range(0,291))],n_train_df[[291]]
X = imp.fit_transform(X_train)
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(random_state=0,n_estimators=100)
clf.fit(X,y)
f_val = list(clf.feature_importances_)

feat = list(n_train_df.columns[:291])
d = dict(zip(feat,f_val))
df = pd.DataFrame.from_dict(d,orient='index')
df