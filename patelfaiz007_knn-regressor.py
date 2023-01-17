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
data = pd.read_csv('../input/Advertising.csv', index_col=0)
data.head()
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data['Sales']
from sklearn.model_selection import train_test_split

#Split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=10)
from sklearn.neighbors import KNeighborsRegressor

model_KNN=KNeighborsRegressor(n_neighbors=4, metric='euclidean')

model_KNN.fit(X_train,y_train)
y_pred=model_KNN.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error
r2score=r2_score(y_test,y_pred)
print(r2score)

rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(predictions.head())
# KNN Regressor can do a pretty good job sometimes!
