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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

train.describe()

train
y_train = train["SalePrice"]

x_train = train.drop(["SalePrice","Id","MSZoning","Street","LotShape","LandContour","Utilities","LotConfig","LandSlope",

                      "Neighborhood","Condition1","Condition2","BldgType","HouseStyle","HouseStyle","RoofStyle","RoofMatl",

                     "Exterior1st","Exterior2nd","Fence","MasVnrType","ExterQual","ExterCond","Foundation","SaleType",

                     "SaleCondition","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1"],axis=1)



x_train.dtypes.sample(10)

plt.hist(np.log(train["LotFrontage"]))

mean_LotFrontage = x_train['LotFrontage'].mean()

std_LotFrontage = x_train['LotFrontage'].std()



x_train.fillna(np.random.randint(mean_LotFrontage - std_LotFrontage, mean_LotFrontage + std_LotFrontage), inplace=True)



x_train["LotFrontage"] = np.log(x_train["LotFrontage"])

x_train["LotArea"] = np.log(x_train["LotArea"])





# one_hot_encoded_training_predictors = pd.get_dummies(x_train)

# one_hot_encoded_training_predictors



# one_hot_encoded_training_predictors

x_train
y_train.describe()