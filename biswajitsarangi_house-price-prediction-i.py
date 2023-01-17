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
# Loding the data



train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

train_data.tail()
test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

test_data.head()
import matplotlib.pyplot as plt
train_data.corr()
# Comparing MSZoning with the price as, the kind of zone the house is situated also affects the price.



plt.scatter(train_data['MSZoning'],train_data['SalePrice'],c='r')

plt.xlabel("MSZoning")

plt.ylabel("Price")

plt.title("PLOTS")
# Comparision between area avaailable near by..



plt.scatter(train_data['LotFrontage'],train_data['SalePrice'],c='g')

plt.xlabel("LotFrontage")

plt.ylabel("Price")

plt.title("PLOTS")


plt.scatter(train_data['YearBuilt'],train_data['SalePrice'],c='b')

plt.xlabel("YearBuilt")

plt.ylabel("Price")

plt.title("PLOTS")
plt.scatter(train_data['OverallQual'],train_data['SalePrice'],c='b')

plt.xlabel("OverallQual")

plt.ylabel("Price")

plt.title("PLOTS")
plt.scatter(train_data['TotalBsmtSF'],train_data['SalePrice'],c='g')

plt.xlabel("Total basement area aquare feet")

plt.ylabel("SalePrice")
plt.scatter(train_data['GrLivArea'],train_data['SalePrice'],c='y')

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")
train_data.isnull().any()
# separating target variable.



y = train_data['SalePrice']



features = ['OverallQual','YearBuilt','TotalBsmtSF','GrLivArea']



X_train = pd.get_dummies(train_data[features])

X_train.isnull().any()
X_test = pd.get_dummies(test_data[features])

X_test.isnull().any()
X_test['TotalBsmtSF'].fillna(X_test['TotalBsmtSF'].mean(),inplace = True)
X_test.isnull().any()
from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(X_train,y)



prediction = lr.predict(X_test)



output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': prediction})

output.to_csv('my_submission.csv', index=False)

print("Submission saved")