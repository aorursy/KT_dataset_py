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
import matplotlib.pyplot as plt
import seaborn as sn
df = pd.read_csv("../input/youtube-new/DEvideos.csv")
df.head()
df.describe()
plt.scatter(df["views"], df["likes"])
plt.xlabel("Views")
plt.ylabel("Likes")
plt.xlim(0, 1e7)
plt.ylim(0, 1e6)
corrMatrix = df[["views", "likes", "dislikes", "comment_count"]].corr(method='pearson')
sn.heatmap(corrMatrix, annot=True)
features = df[["views", "dislikes", "comment_count"]]
target = df[["likes"]]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
predictions_lr = model_lr.predict(X_test)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions_lr, y_test)))
from xgboost import XGBRegressor
model_base = XGBRegressor()
model_base.fit(X_train, y_train)
predictions = model_base.predict(X_test)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_test)))
model_n5000 = XGBRegressor(n_estimators=5000, learning_rate=0.001)
model_n5000.fit(X_train, y_train)
predictions_n5000 = model_n5000.predict(X_test)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions_n5000, y_test)))