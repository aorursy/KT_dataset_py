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
df = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv')
df.head()
df.columns
df['description'].isna()
df[df['description'].isna()]
df['description'].fillna("missing", inplace = True)
df[df['description'].isna()]
df['description'].dropna(inplace = True)
#select features and target
X = df[['likes', 'dislikes', 'views', 'category_id']]
y = df[
    'comment_count'
]
#convert categorical data
X = pd.get_dummies(X, columns=['category_id'], drop_first=True)
X
                   
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
from sklearn.metrics import mean_squared_error

mean_squared_error (ytest, y_model, squared=False)

from sklearn import svm
model = svm.LinearSVR(dual = False, loss='squared_epsilon_insensitive')
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

mean_squared_error(ytest, y_model, squared=False)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=8)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

mean_squared_error(ytest, y_model, squared=False)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)


from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(15,10,5), alpha=0.0001, max_iter=1000, solver="lbfgs")
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

mean_squared_error(ytest, y_model, squared=False)


