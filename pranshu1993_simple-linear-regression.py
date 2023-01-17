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

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

import seaborn as sns
df_train = pd.read_csv("../input/random-linear-regression/train.csv")

df_test = pd.read_csv("../input/random-linear-regression/test.csv")
# Check for any null Values for train Dataset

df_train.isnull().sum()
# Check for any null values for test Dataset

df_test.isnull().sum()
clean_df_train = df_train.dropna()

clean_df_train.isnull().sum()
X_train = np.array(clean_df_train.iloc[:,:-1])

Y_train = np.array(clean_df_train.iloc[:,1])

print(X_train.shape)

print(Y_train.shape)
X_test = np.array(df_test.iloc[:,:-1])

Y_test = np.array(df_test.iloc[:,1])

print(X_test.shape)

print(Y_test.shape)
LR = LinearRegression()
LR.fit(X_train,Y_train)
y_pred = LR.predict(X_test)
RMSE = np.sqrt(mean_squared_error(Y_test,y_pred))

RMSE
accuracy = LR.score(X_test,Y_test)

accuracy 
sns.jointplot(clean_df_train.x,clean_df_train.y,kind='reg')