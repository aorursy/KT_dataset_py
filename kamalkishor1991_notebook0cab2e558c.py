# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt

%matplotlib inline

# Any results you write to the current directory are saved as output.
training_data = pd.read_csv('../input/train.csv', encoding="utf8")
training_data.describe()

training_data.shape
test_data = pd.read_csv('../input/test.csv', encoding="utf8")
n, bins, patches = plt.hist(training_data['SalePrice'], 100)

plt.axvline(training_data['SalePrice'].mean(), color='g', linestyle='dashed', linewidth=2)
test_data.shape[0]
training_data['MSSubClass'].unique()
training_data['LotFrontage'] = training_data['LotFrontage'].fillna(training_data['LotFrontage'].mean())
n, bins, patches = plt.hist(training_data['LotFrontage'], 10)
n, bins, patches = plt.hist(training_data['MSSubClass'], 15)
training_data['MSZoning'].value_counts()
from sklearn import tree

from sklearn import preprocessing

from sklearn import datasets, linear_model

from xgboost import XGBRegressor

reg =XGBRegressor()
def labelEncode(value) :

    le = preprocessing.LabelEncoder()

    le.fit(value.unique())

    return le.transform(value)
training_data["Street"].value_counts()
training_data["Alley"].fillna("na").value_counts()
print("Nan count: " + str(training_data["LotShape"].isnull().sum()))

training_data["LotShape"].value_counts()
print("Nan count: " + str(training_data["LandContour"].isnull().sum()))

training_data["LandContour"].value_counts()
print("Nan count: " + str(training_data["Utilities"].isnull().sum()))

training_data["Utilities"].value_counts()




for column in training_data.columns:

    print("---------------------------" + column + "-------------------------------------------------")    

    print("Nan count: " +  str(training_data[column].isnull().sum()))

    print(str(training_data[column].value_counts()))

    print("----------------------------------------------------------------------------")
training_data.columns


def getXFromData(data):

    xr = data.copy(True).iloc[:, 1:79]

    

    xr = xr.loc[:, ["MSSubClass", "LotFrontage", "LotArea", "MSZoning", "LotShape", "LandContour", "YrSold", 

                    "BsmtHalfBath",  "FullBath", "HalfBath", "BedroomAbvGr", 

                    "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", 

                    "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", 

                    "ScreenPorch", "PoolArea", "MiscVal", "MoSold",  

                     ]]

    xr["LotFrontage"] = xr["LotFrontage"].fillna(training_data["LotFrontage"].mean())

    #stringCols = [ "CentralAir", "SaleType"]

    xr["MSZoning"] = labelEncode(xr["MSZoning"])

    xr["LotShape"] = labelEncode(xr["LotShape"])

    xr["LandContour"] = labelEncode(xr["LandContour"])

    #for col in stringCols:

    #   print("column:" + col)

    #   xr[col] = labelEncode(xr[col])

    return xr;

x = getXFromData(training_data)
x
y = training_data.iloc[:, 80:81]
reg.fit(x, y)
np.mean((np.array(reg.predict(x)) - np.array(y)) ** 2)
test_data["MSZoning"] = test_data["MSZoning"].fillna("RL")
tx = getXFromData(test_data)

tx
output = reg.predict(tx)

output.reshape(-1)
reshape_output = output.reshape(-1)

test_output = pd.DataFrame({'Id': test_data["Id"], 'SalePrice': reshape_output})
test_output.to_csv('./output.csv', index=False)