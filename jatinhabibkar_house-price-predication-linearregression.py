# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

# import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline
theredata=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

theredata.head()
theredata.columns
Dataset=theredata[['LotArea','YrSold','OverallQual','OverallCond','YearBuilt','TotalBsmtSF','1stFlrSF','GrLivArea','GarageArea','YrSold','SalePrice']]
Dataset.head()

Dataset.dropna(inplace=True)
from sklearn.model_selection import train_test_split

Dataset.head()

Dataset.dropna()

Dataset.shape
y=Dataset['SalePrice']

X=Dataset.drop('SalePrice',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
Dataset.describe()
import seaborn as sns

sns.distplot(Dataset['SalePrice'])
sns.heatmap(X.corr())
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
test_data_set=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
feature_data_kaggle=test_data_set[['LotArea','YrSold','OverallQual','OverallCond','YearBuilt','TotalBsmtSF','1stFlrSF','GrLivArea','GarageArea','YrSold']]

feature_data_kaggle=feature_data_kaggle.apply(lambda row: row.fillna(row.mean()), axis=1)
predictions=lm.predict(feature_data_kaggle)

data={'Id':test_data_set['Id'],'SalePrice':predictions}

new_prediction=pd.DataFrame(data,columns=['Id','SalePrice']).astype('int')
new_prediction.to_csv("../working/submission.csv",index=False)
new_prediction.shape