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
#Importing necessary libraries

import matplotlib.pyplot as plt

%matplotlib inline
# DataSet as bos

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']



bos = pd.read_csv("/kaggle/input/boston-house-prices/housing.csv" , header=None, delimiter=r"\s+", names=column_names)



bos.columns
bos
bos.shape
x = bos.iloc[:,0:13]

y = bos["MEDV"]
#Establishing correlation in data



import seaborn as sns

names = []

#creating a correlation matrix



correlations = bos.corr()

sns.heatmap(correlations,square = True, cmap = "inferno")

plt.yticks(rotation=0)

plt.xticks(rotation=90)

plt.show()
# Splitting dataset as Test and Train

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.33, random_state = 5)
from sklearn.linear_model import LinearRegression



#Fitting model to train and test data to linear regression model

lr = LinearRegression()



model = lr.fit(x_train, y_train)
pred = lr.predict(x_test)
# Creating a dataframe with the values

pd.DataFrame({"Actual": y_test, "Predict": pred})
# Visualization of the above dataframe in a scatterplot



plt.scatter(y_test, pred)

plt.xlabel('Y test')

plt.ylabel('X test (or) Predicted')
# Mean Squared Error (MAE)

from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import r2_score as score



print("Accuracy :", score(y_test, pred))

print("Mean Squared Error :", mse(y_test, pred))
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor

from xgboost import XGBClassifier,XGBRFRegressor,XGBRegressor

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

from sklearn.linear_model import LogisticRegression,LinearRegression,SGDRegressor

from sklearn.svm import SVC,SVR

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

from sklearn.naive_bayes import GaussianNB

from lightgbm import LGBMRegressor
model_a = XGBRegressor(n_estimators = 1000)

model_a.fit(x_train, y_train)
pred_a = model_a.predict(x_test)





# Comparison on prediction values before and after using XGBoost

pd.DataFrame({"Actual": y_test, "Predict":pred, "Predict_XGBOOST": pred_a})
# Mean Squared error after Boosting



print("Accuracy :", score(y_test, pred_a))

print("Mean Squared Error :", mse(y_test,pred_a))