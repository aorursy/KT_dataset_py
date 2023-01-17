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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import random
data = pd.read_csv("../input/insurance.csv")
data.shape
data.describe()
data.head()
data.isnull().sum()
#We have no null values, so we can analyse and proceed with Prediction
data.age.unique()
sns.barplot(x=data.sex,y=data.charges,)
#Plot shows, male pay more charges than female, although not by great margin
corr = data.corr()
sns.heatmap(corr,cmap="Blues",mask=np.zeros_like(corr, dtype=np.bool),annot=True)
#Heatmap shows that there is no major corelation between dependent variables
#Let's do a basic Linear  Regressin with all the variables
data_new = pd.get_dummies(data,drop_first=True)
data_new.head()
df = data_new.drop(['charges'],axis=1)
y = data_new['charges']
random.seed(2)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
from sklearn.linear_model import LinearRegression
log_reg=LinearRegression()
log_reg.fit(X_train,y_train)
pred = log_reg.predict(X_test)
from sklearn.metrics import r2_score
print (r2_score(y_test, pred))
print (pred)
#Lets Use Random Forest for Prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error

forest = RandomForestRegressor(n_estimators = 100,
                              criterion = 'mse',
                              random_state = 1,
                              n_jobs = -1)
forest.fit(X_train,y_train)
forest_train_pred = forest.predict(X_train)
forest_test_pred = forest.predict(X_test)

print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,forest_train_pred),
r2_score(y_test,forest_test_pred)))
