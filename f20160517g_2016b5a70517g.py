# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# example to generate the submission file

# generateSubmissionFile("sample.csv", df_train['id'], df_train['rating'])



def generateSubmissionFile(filename, df_id, df_rating):

    tmp_data = pd.DataFrame(data=np.transpose([np.asarray(df_id).flatten(), np.asarray(df_rating)]), columns = ['id', 'rating'])

    df_submission = pd.DataFrame(data=tmp_data, columns = ['id', 'rating'])

    pd.DataFrame.to_csv(df_submission, filename, index = False)
# read the files



df_trainfile = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")

df_testfile = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")
df_trainfile.fillna(df_trainfile.mean(), inplace=True)

df_testfile.fillna(df_testfile.mean(), inplace=True)
df_trainfile['type'] = [0 if x=='new' else 1 for x in df_trainfile['type']]

df_testfile['type'] = [0 if x=='new' else 1 for x in df_testfile['type']]



# changing the encoding

# df_trainfile['type'] = [1 if x=='new' else 0 for x in df_trainfile['type']]

# df_testfile['type'] = [1 if x=='new' else 0 for x in df_testfile['type']]



# 

# df_trainfile = pd.get_dummies(df_trainfile)

# df_testfile = pd.get_dummies(df_testfile)
X = df_trainfile.loc[:, df_trainfile.columns!='rating']

y = df_trainfile[['rating']]
X_test = df_testfile.loc[:, df_testfile.columns!='rating']
X.head()
y.head()
etr = ExtraTreesRegressor(n_estimators=2000, max_depth=None, max_features=None)

etr.fit(X, y.values.ravel())
y_pred1 = np.round(etr.predict(X_test))

generateSubmissionFile("sub1.csv", df_testfile['id'].astype(int), pd.DataFrame(data=y_pred1.astype(int), columns=['rating'])['rating'])
etr = ExtraTreesRegressor(n_estimators=2100, max_depth=None, max_features=None)

etr.fit(X, y.values.ravel())
y_pred1 = np.round(etr.predict(X_test))

generateSubmissionFile("sub2.csv", df_testfile['id'].astype(int), pd.DataFrame(data=y_pred1.astype(int), columns=['rating'])['rating'])

