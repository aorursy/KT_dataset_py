# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_test.head()
df_train.info()
df_train.describe()
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
df_train, df_cv = train_test_split(df_train, test_size = 0)
df_trainy = df_train['target']
df_trainx = df_train.drop(['target', 'ID'], axis = 1)
#df_cvy = df_cv['target']
#df_cvx = df_cv.drop(['target', 'ID'], axis = 1)
from sklearn.decomposition import PCA
pca = PCA(n_components=512)
pca.fit(df_trainx)
plt.plot(pca.explained_variance_ratio_)
plt.show()
plt.plot(pca.singular_values_)
plt.show()
print(np.sum(pca.explained_variance_ratio_))
df_trainx = pd.DataFrame(pca.transform(df_trainx))
#df_cvx = pd.DataFrame(pca.transform(df_cvx))
df_trainx.head()
from sklearn.linear_model import LinearRegression as Lr
from sklearn.ensemble import RandomForestRegressor as Rfr
from sklearn.ensemble import GradientBoostingRegressor as Gbr
from xgboost import XGBRegressor as Xgb
lr = Lr(normalize=True)
lr.fit(df_trainx, df_trainy)
lr.score(df_trainx, df_trainy)
rf = Rfr()
rf.fit(df_trainx, df_trainy)
rf.score(df_trainx, df_trainy)
#rf.score(df_cvx, df_cvy)
gbr = Gbr()

gbr.fit(df_trainx, df_trainy)

gbr.score(df_trainx, df_trainy)
#gbr.score(df_cvx, df_cvy)
xgb = Xgb()
xgb.fit(df_trainx, df_trainy)
xgb.score(df_trainx, df_trainy)
#xgb.score(df_cvx, df_cvy)
submit = pd.DataFrame(df_test['ID'])
df_test = df_test.drop(['ID'], axis = 1)
df_test.shape
df_test = pd.DataFrame(pca.transform(df_test))
sub01 =pd.Series(np.abs((gbr.predict(df_test) + xgb.predict(df_test) + lr.predict(df_test) + rf.predict(df_test)) / 4), name='target')
submit = submit.join(sub01)
submit.to_csv('../working/submit01.csv', index=False)
submit.drop(['target'], axis = 1, inplace=True)
sub02 = rf.predict(df_test)
submit = submit.join(sub02)
submit.to_csv('../working/submit02.csv', index=False)

