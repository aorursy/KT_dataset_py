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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
train_data = pd.read_csv("/kaggle/input/into-the-future/train.csv")
train_data.head()
train_data.info()
X = train_data["feature_1"].values.reshape(-1,1)
y = train_data["feature_2"].values.reshape(-1,1)
plt.scatter(X,y);
sns.distplot(train_data["feature_1"]);
sns.distplot(train_data["feature_2"]);
sns.jointplot(x='feature_1', y='feature_2', data=train_data, kind='hex');
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
Model_LR = LinearRegression()
Model_LR.fit(X,y)
from sklearn.svm import SVR
Model_SVR = SVR(kernel='linear')
Model_SVR.fit(X,y)
from sklearn.ensemble import RandomForestRegressor
Model_RFR = RandomForestRegressor()
Model_RFR.fit(X,y);
test_data = pd.read_csv("/kaggle/input/into-the-future/test.csv")
test_data.head()
test_feature = test_data["feature_1"].values.reshape(-1,1)
test_id = test_data["id"].values.reshape(-1,1)
pred_LR = Model_LR.predict(test_feature)
pred_1 = pred_LR.mean()
pred_1
pred_SVR = Model_SVR.predict(test_feature)
pred_2 = pred_SVR.mean()
pred_2
pred_RFR = Model_RFR.predict(test_feature)
pred_3 = pred_RFR.mean()
pred_3
from sklearn.model_selection import GridSearchCV
import time
np.random.seed(42)
grid = {'n_estimators': [100, 200, 500],
        'max_depth': [None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_split': [6],
        'min_samples_leaf': [1, 2]}

grid_result = GridSearchCV(estimator=Model_RFR,
                       param_grid=grid,
                       cv=5,
                       verbose=2)
start_time = time.time()
grid_result.fit(X,y)
# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')
pred_hypertune = grid_result.predict(test_feature)
pred_4 = pred_hypertune.mean()
pred_4
compare_models = pd.DataFrame({'LinearRegression': [pred_1],
                              'SVR': [pred_2],
                              'RandomForestRegression': [pred_3],
                              'Hypertune RandomForestRegression': [pred_4]})
compare_models.plot.bar(figsize=(10,8));
pred_LR = pd.DataFrame({'id': test_id.flatten(),'feature_2': pred_LR.flatten()})
pred_LR.to_csv('LR.csv')
pred_SVR = pd.DataFrame({'id': test_id.flatten(),'feature_2': pred_SVR.flatten()})
pred_SVR.to_csv('SVR.csv')
pred_RFR = pd.DataFrame({'id': test_id.flatten(),'feature_2': pred_RFR.flatten()})
pred_RFR.to_csv('RFR.csv')
