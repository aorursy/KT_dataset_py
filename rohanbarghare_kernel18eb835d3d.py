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

import numpy as np

from sklearn.model_selection import train_test_split
df=pd.read_csv('../input/file.csv')
df


# code starts here



X = df.drop('list_price',axis=1)



# store dependent variable

y = df['list_price']



# spliting the dataset

X_train,X_test,y_train,y_test=train_test_split(X,y ,test_size=0.3,random_state=6)

# code ends here
X_train
import matplotlib.pyplot as plt





cols = X_train.columns



fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20,20))



for i in range(0,3):

    for j in range(0,3): 

            col = cols[i*3 + j]

            axes[i,j].set_title(col)

            axes[i,j].scatter(X_train[col],y_train)

            axes[i,j].set_xlabel(col)

            axes[i,j].set_ylabel('list_price')

        



# code ends here

plt.show()
cols = X_train.columns



fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20,20))



for i in range(0,3):

    for j in range(0,3): 

            col = cols[i*3 + j]

            axes[i,j].set_title(col)

            axes[i,j].scatter(X_train[col],y_train)

            axes[i,j].set_xlabel(col)

            axes[i,j].set_ylabel('list_price')

        



# code ends here

plt.show()

corr=X_train.corr()
corr
# Code starts here



X_train.drop(['play_star_rating','val_star_rating'], 1 ,inplace=True) 

X_test.drop(['play_star_rating','val_star_rating'], 1 ,inplace=True)

# Code ends here
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score



# Code starts here

regressor=LinearRegression()

regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

mse=mean_squared_error(y_test,y_pred)

r2=r2_score(y_test,y_pred)
mse
r2
# Code starts here

residual = y_test - y_pred

plt.hist(residual)

plt.show()

# Code ends here