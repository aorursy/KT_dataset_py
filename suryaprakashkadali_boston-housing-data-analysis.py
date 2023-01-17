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
import pandas as pd
dataframe = pd.read_csv('../input/HousingData.csv')
dataframe.info()
dataframe.describe()
dataframe.isnull().sum()
dataframe=dataframe.dropna()
import seaborn as sns
sns.pairplot(data = dataframe)
#we can see that MEDV is strongly correlated to LSTAT, RM
# RAD and TAX are stronly correlated so to avoid multiple multi-colinearity
target = dataframe['MEDV']
features = dataframe[['LSTAT','RM']]
import matplotlib.pyplot as plt
plt.hist(target,bins=10)
import numpy as np
dataframe_normalized = np.log(target)
plt.hist(dataframe_normalized)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.2,train_size=0.8)
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
reg_fit = regressor.fit(X_train,y_train)
reg_pred = reg_fit.predict(X_test)
score_not_norm = r2_score(y_test,reg_pred)
print(score_not_norm)
