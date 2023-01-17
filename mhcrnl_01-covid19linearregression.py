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

import seaborn as sb



train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

train.head(10)
train_x = pd.DataFrame(train.iloc[:,-5])

train_x.head(10)

                       
train_y=pd.DataFrame(train.iloc[:,-1])

train_y.head(10)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size=0.3)
from sklearn.linear_model import LinearRegression

regression=LinearRegression()

regression.fit(X_train,Y_train)

Y_pred_lin=regression.predict(X_test)

y_pred_train=pd.DataFrame(Y_pred_lin,columns=["Predict"])

Y_test.head()
y_pred_train.head()
sub=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/submission.csv")

sub.to_csv('submission.csv',index=False)