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
from sklearn.ensemble import RandomForestClassifier
wine = pd.read_csv('../input/winequality-red-train.csv')
wine_val = pd.read_csv('../input/winequality-red-validate-data.csv')
wine.head()
wine.drop(['id'], axis=1, inplace=True)
val = wine_val.drop(['id'], axis=1)
X = wine.drop('quality', axis = 1)
y = wine['quality']
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X, y)
pred = rfc.predict(val)
wine_val['quality'] = pred
wine_val.head()
wine_val.to_csv('submission.csv', columns=('id', 'quality'), index=False)