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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/negative_tweet_updated.csv')

df.head(3)
df1 = df[['id','Comments', 'favoriteCount', 'retweetCount', 'photo', 'sentiment'  ]]

df1.head(1)
df1.rename(columns={'sentiment':'sensitivity',},inplace=True)

# using sentiment score as proxy of sensitivity score
from sklearn.preprocessing import StandardScaler, LabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error
categorical_feature = ['photo']

X = df1.drop(categorical_feature, axis= 1)
scaler = StandardScaler()

lb = LabelBinarizer()

X = scaler.fit_transform(X)

X = np.c_[X,lb.fit_transform(df1['photo'])]
X_train,X_test,y_train,y_test = train_test_split(X,df1['sensitivity'])
reg = DecisionTreeRegressor()

reg.fit(X_train,y_train)
reg.score(X_test,y_test)
mean_absolute_error(y_test, reg .predict(X_test))
# XGBRegressor

from xgboost import XGBRegressor

xgb = XGBRegressor()

xgb.fit(X_train,y_train)

xgb.score(X_test,y_test)
mean_absolute_error(y_test, xgb .predict(X_test))
data = df1
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import StratifiedKFold

from yellowbrick.model_selection import LearningCurve

from sklearn.linear_model import RidgeCV
# Specify features of interest and the target

targets = ["sensitivity"]

features = ['id','Comments', 'favoriteCount', 'retweetCount', 'photo',]



# Extract the instances and target

X = data[features]

y = data[targets[0]]
# Create the learning curve visualizer, fit and poof

viz = LearningCurve(RidgeCV(), scoring='r2')

viz.fit(X, y)

viz.poof()