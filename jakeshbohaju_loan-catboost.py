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

df = pd.read_csv("../input/loan-data/Loan-data.csv")

df.columns = df.iloc[0]

df = df[1:]
df.head(5)
df.isnull().sum(axis = 0)
y = df['default payment next month']

X = df.drop(['default payment next month','ID'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.25)
cate_features_index = np.where(X.dtypes != float)[0]
cate_features_index
from catboost import CatBoostRegressor

model = CatBoostRegressor(iterations=1500, learning_rate=0.01, loss_function= 'RMSE', eval_metric='AUC',use_best_model=True,random_seed=42)
model.fit(X_train,y_train,cat_features=cate_features_index,eval_set=(X_test,y_test))