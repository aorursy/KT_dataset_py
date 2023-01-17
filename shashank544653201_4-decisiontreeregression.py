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
#Importing Dataset
dataset = pd.read_csv("../input/redwinequality/datasets_4458_8204_winequality-red.csv")
#Peek into Dataset
print(dataset.head())     #First Few Rows
print(dataset.info())     #Print a concise summary of a DataFrame
print(dataset.describe()) #Generate descriptive statistics.

#Spliting Dataset into Feature and Dependent Variable
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1)
#Spliting Dataset into Training and Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0 , test_size = 0.3 )
#Fitting the DecisionTree To the Dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train,y_train)
#Predicting on test set
y_pred = regressor.predict(X_test)
#Mean Squared Error
from sklearn.metrics import mean_squared_error 
mean_squared_error(y_test, y_pred)
#Comparing Predicted and Actual Values 
compare = pd.DataFrame({"Actual": y_test.flatten(),"Pedicted": y_pred.flatten()}).head(10)
compare
