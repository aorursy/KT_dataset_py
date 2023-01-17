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
train_data = pd.read_csv(r'../input/house-prices-advanced-regression-techniques/train.csv')

train_data.head()
test_data = pd.read_csv(r'../input/house-prices-advanced-regression-techniques/test.csv')

test_data.head()
train_data = train_data.fillna(0)

test_data = test_data.fillna(0)
pearson = train_data.corr("pearson")

pearson.sort_values(by = "SalePrice", ascending = False, inplace = True)

pearson.head(30)
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X = sc.fit_transform(X)

X_test = sc.transform(X_test)



y = train_data["SalePrice"]

features = ["OverallQual", "GrLivArea", "TotalBsmtSF", "TotRmsAbvGrd", "GarageCars","GarageArea", "1stFlrSF", "FullBath", "YearBuilt","YearRemodAdd"]





model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions_RF = model.predict(X_test)



output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions_RF})

output.to_csv('house_submission_forest.csv', index=False)

print("Your submission was successfully saved!")
from sklearn.tree import DecisionTreeClassifier



features = ["OverallQual", "GrLivArea", "TotalBsmtSF", "TotRmsAbvGrd", "GarageCars","GarageArea", "1stFlrSF", "FullBath", "YearBuilt","YearRemodAdd"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

HouseTree = DecisionTreeClassifier(criterion="entropy", max_depth = 3)

HouseTree.fit(X,y)

predictions_DT = HouseTree.predict(X_test)



output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions_DT})

output.to_csv('my_submission_tree.csv', index=False)

print("Your submission was successfully saved!")