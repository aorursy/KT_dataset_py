# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd

df = pd.read_csv('/kaggle/input/particle-identification-from-detector-responses/pid-5M.csv')
df.head()
df.describe()
sns.set(style='darkgrid')
corr = df.corr()
sns.heatmap(corr)
df.isnull().sum()
print(df[df.id == -11].shape[0])
print(df[df.id == 211].shape[0])
print(df[df.id == 321].shape[0])
print(df[df.id == 2212].shape[0])
df = df.drop(df[df.beta >= 1].index)
df.describe()
features = df.drop('id', axis=1)
labels = df['id']
features.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, labels, 
                                                    test_size = 0.20, random_state = 13)
import time
from sklearn.metrics import accuracy_score
t0= time.perf_counter()

from sklearn.linear_model import SGDClassifier

model = SGDClassifier()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
print('accuracy score:', accuracy_score(y_test, prediction))

t1 = time.perf_counter() - t0
print('tempo necessario: ', t1, ' s')
t0 = time.perf_counter()

from sklearn.ensemble import AdaBoostClassifier

clf_abc = AdaBoostClassifier()
clf_abc.fit(x_train, y_train)
pred_abc = clf_abc.predict(x_test)
print('accuracy score:', accuracy_score(y_test, pred_abc))

t1 = time.perf_counter() - t0
print('tempo necessario: ', t1, ' s')
t0 = time.perf_counter()

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb = XGBClassifier()
xgb.fit(x_train, y_train)
pred_xgb = xgb.predict(x_test)
print('accuracy score:', accuracy_score(y_test, pred_xgb))

t1 = time.perf_counter() - t0
print('tempo necessario: ', t1, ' s')
t0 = time.perf_counter()

from sklearn.ensemble import RandomForestClassifier
clf_rfc = RandomForestClassifier(n_estimators=50, max_depth=4)
clf_rfc.fit(x_train, y_train)
pred_rfc = clf_rfc.predict(x_test)
print('accuracy score:', accuracy_score(y_test, pred_rfc))

t1 = time.perf_counter() - t0
print('tempo necessario: ', t1, ' s')
from sklearn.tree import export_graphviz

estimator = clf_rfc.estimators_[1]
export_graphviz(estimator, out_file='tree.dot',
                feature_names=features.columns,
                filled=True,
                rounded=True)

import os
os.system('dot -Tpng ./tree.dot -o tree.png')