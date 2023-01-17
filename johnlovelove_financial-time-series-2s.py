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
df = pd.read_hdf('../input/train.h5')
df = df.dropna()

df.describe()
mid = 1081

excl = ['id', 'sample', 'y', 'timestamp']

col = [c for c in df.columns if c not in excl]



df_train = df[df.timestamp <= mid][col]

d_mean= df_train.median(axis=0)



df_all = df[col]



X_train = df_all[df.timestamp <= mid].values

y_train = df.y[df.timestamp <= mid].values

X_valid = df_all[df.timestamp > mid].values

y_valid = df.y[df.timestamp > mid].values

feature_names = df_all.columns

del df_all, df_train, df
X_train.shape
X_valid.shape
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

rf.fit(X_train, y_train)

print("Features sorted by their score:")

print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names), 

             reverse=True))
indicators = ['technical_33', 'technical_24', 'technical_41', 'technical_1', 'technical_3', 'technical_28', 'technical_44', 'technical_31', 'technical_5', 'technical_30']