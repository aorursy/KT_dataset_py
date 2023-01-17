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
df = pd.read_csv("/kaggle/input/inspiration/train.csv")
df.head()
y = df["% Silica Concentrate"]

x = df.drop("% Silica Concentrate",axis=1)

x = x.drop("date",axis=1)
x.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2)
from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=2)

neigh.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error



preds = neigh.predict(X_train)



print("Trainings Error")

print(mean_squared_error(preds,y_train))
print(mean_squared_error(preds,y_train))
submission = pd.read_csv("/kaggle/input/inspiration/Submission.csv")

submission.head()
x_test = pd.read_csv("/kaggle/input/inspiration/test.csv")

x_test = x_test.drop("date",axis=1)
X_train.head()

x_test.head()
preds = neigh.predict(x_test)
submission.head()

submission["Expected"].shape
preds.shape
submission.to_csv("submission_base_blend_top_Scale extremes2.csv", index=False) 
sub =  pd.read_csv("/kaggle/input/inspiration/Submission.csv")