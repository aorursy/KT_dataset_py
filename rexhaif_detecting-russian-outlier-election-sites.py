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
import sklearn.ensemble as ens
import sklearn.svm as svm
import numpy as np
import seaborn as sns
df = pd.read_csv('../input/uiks-utf8.csv')
df.head(2)
sns.distplot(df['registered_voters'])
df = df[df['registered_voters'] < 3001]
sns.distplot(df['registered_voters'])
numerical_data = df.drop(['region_name', 'tik_name', 'uik_name'], axis=1)
numerical_data.head(2)
numerical_data = numerical_data.values
detector = ens.IsolationForest(n_jobs=-1, n_estimators=250)
numerical_data = np.random.permutation(numerical_data)
train, test = numerical_data[:85000], numerical_data[85000:]
detector.fit(X=numerical_data.values)
res = detector.predict(numerical_data.values)
numerical_data['outlier_score'] = res
numerical_data['outlier_score'].value_counts()
