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
titanic_file_path = '/kaggle/input/titanic/train.csv'
titanic_data_val = pd.read_csv(titanic_file_path)
titanic_data_val.columns
titanic_data = titanic_data_val.dropna(axis=0)
titanic_data.columns
y = titanic_data.Survived
titanic_features = ['Pclass','Age','Sex']
X = titanic_data[titanic_features]
X.describe
X.head()
#changing datatype from object to float
one_hop_encoded_X = pd.get_dummies(X)
one_hop_encoded_X.head()
#checking null values
X_null_val = one_hop_encoded_X.isnull().sum()
print(X_null_val)
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
titanic_model = DecisionTreeRegressor(random_state = 1)

#fit model
titanic_model.fit(X_with_imputed_val,y)

#Test set

titanic_test_path = '/kaggle/input/titanic/test.csv'
titanic_test_val = pd.read_csv(titanic_test_path)
titanic_test = titanic_test_val.dropna(axis=0)
test_features = ['Pclass','Age','Sex']
test_X = titanic_test[test_features]
test_X.head()
one_hop_encoded_test = pd.get_dummies(test_X)
one_hop_encoded_test.head()
#Prediction
titanic_model.predict(one_hop_encoded_test)