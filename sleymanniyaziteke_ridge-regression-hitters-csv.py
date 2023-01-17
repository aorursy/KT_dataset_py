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

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.model_selection import train_test_split,cross_val_score

import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeCV
hit=pd.read_csv("../input/hitters/Hitters.csv")

hit.head()
hit.describe().T
hit.isnull().sum()
hit=hit.dropna()
hit["League"].value_counts()
hit["Division"].value_counts()
hit["NewLeague"].value_counts()
hit=pd.get_dummies(hit,columns=["League","Division","NewLeague"],drop_first=True)

hit.head()
y=hit["Salary"]

X=hit.drop("Salary",axis=1)
X.head()
X_train,X_test,y_train,y_test=train_test_split(X,

                                               y,

                                               test_size=0.20,

                                               random_state=46

                                              )
ridge_model=Ridge().fit(X_train,y_train)

ridge_model.intercept_
ridge_model.coef_
y_pred=ridge_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))
alphas=np.linspace(0,1,1000)
ridge_cv=RidgeCV(alphas,scoring="neg_mean_squared_error",cv=10,normalize=True)

ridge_cv.fit(X_train,y_train)

ridge_cv.alpha_
ridge_tuned=Ridge(alpha=0.3933933933933934).fit(X_train,y_train)

y_pred=ridge_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))