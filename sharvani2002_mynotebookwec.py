# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import datetime
import os
print((os.listdir('../input/')))
df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')
df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
df_test.head()
df_train.head()
test_index=df_test['Unnamed: 0']

df_train.plot('F2')
df_train["O/P"].nunique()
plt.plot(df_train["F9"], df_train["O/P"])
df_train.loc[:, 'F6':'F9']
df_test.F2.max()
# plt.plot(frame1["Month"], df_train["O/P"])


df_train.drop(['F1', 'F2'], axis = 1, inplace = True)
train_y = df_train.loc[:, 'O/P']
train_X = df_train.loc[:, 'F3':'F17']

# train_X.drop(['O/P'], axis = 1, inplace = True)
from sklearn.feature_selection import SelectFromModel

rf = RandomForestRegressor(n_estimators=500, random_state=43)
# sel = SelectFromModel(RandomForestClassifier(n_estimators = 50))
rf.fit(train_X, train_y)
# mse = sklearn.metrics.mean_squared_error(actual, predicted)
# rmse = math.sqrt(mse)
# print('Accuracy for Random Forest',100*max(0,rmse))
from sklearn import metrics
from sklearn import model_selection
Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(train_X, train_y)
rf.fit(Xtrain, Ytrain)
result = rf.score(Xtest, Ytest)
print("Accuracy: %.3f%%" % (result*100.0))
# pscore_train = metrics.accuracy_score(y_train, pred_train)
# print(pscore_train)
df_test = df_test.loc[:, 'F3':'F17']
pred = rf.predict(df_test)
print(pred)
result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(pred)
result.head()
result.to_csv('output0.csv', index=False)

