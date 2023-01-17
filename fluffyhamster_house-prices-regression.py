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
data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

data.head()
y_train = data.SalePrice
X_train = pd.DataFrame()
print(data.columns.size)
columns = list(data.columns)
numerical_data = ["MSSubClass", "LotFrontage", "LotArea", "OverallQual", "OverallCond", 

                  "MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", 

                  "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", 

                  "HalfBath", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",  "GarageCars", 

                  "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", 

                  "ScreenPorch", "PoolArea", "MiscVal", "BsmtFinSF1", "BsmtFinSF2", 

                  "BedroomAbvGr"]

for col in numerical_data:

    print(col, data[col].isnull().sum())

    X_train[col] = data[col]
X_train.LotFrontage = X_train.LotFrontage.fillna(0)

X_train.LotFrontage.isnull().sum()
X_train.MasVnrArea = X_train.MasVnrArea.fillna(0)

X_train.MasVnrArea.isnull().sum()
other_columns = []

for col in columns:

    if numerical_data.count(col) < 1:

        other_columns.append(col)

print(other_columns)
import math
def Sturges_interval(data, column):

    

    x_max = data[column].max()

    x_min = data[column].min()

    

    n = data[column].size # count elements



    m = 1 + math.log(n, 2) # count intervals

    h = math.ceil((x_max - x_min) / m) # Sturges's formula 

    x_start = round(x_min - h / 2)

    

    intervals = []

    for i in range(round(m)):

        

        interval = [x_start, x_start + h]

        intervals.append(interval)

        x_start = interval[1]

    

    return intervals 
def col_new(data, column, intervals):

    m = 0

    for inter in intervals:

        m = m + 1

    i = 0

    for interval in intervals:

        i = i + 1

        value = []

        if i == m:

            for value_col in data[column]:

                if value_col <= int(interval[1]) and value_col >= int(interval[0]):

                    val = 1

                else:

                    val = 0

                value.append(val)

        else:

            for value_col in data[column]:

                if value_col < int(interval[1]) and value_col >= int(interval[0]):

                    val = 1

                else:

                    val = 0

                value.append(val)

    

        X_train["["+str(interval[0])+" ,"+str(interval[1]) + ")"] = value
X_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
date_data = ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]



for col in date_data:

    print(col, " train :", data[col].min(), data[col].max())

    print(col, " test :", X_test[col].min(), X_test[col].max())
X_train.shape
date_data = ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]



for col in date_data:

    intervals = Sturges_interval(data, col)

    print(col, intervals)

    col_new(data, col, intervals)



X_train = X_train.drop("[2013 ,2019)", axis=1)
X_train.shape
Month = [i + 1 for i in range(12)] 

Month
# date data Sold: "MoSold", "YrSold"



months = [i + 1 for i in range(12)] 



for month in months:

    X_train["month_" + str(month)] = [int(val == month) for val in data.MoSold]
X_train.shape
YrSold = list(data.YrSold.unique())



for year in YrSold:

    X_train["year_sold_" + str(year)] = [int(val == year) for val in data.YrSold]
X_train.shape
date_data = ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold", "MoSold"]
cotegorical_columns = []

for col in other_columns:

    if date_data.count(col) < 1:

        cotegorical_columns.append(col)

print(cotegorical_columns)
for col in cotegorical_columns:

    print(col, " nan % = ", 100 * (data[col].size - data[col].value_counts().sum()) / data[col].size)
drop_columns = ["SalePrice", "Id", "Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]



cotegorical_nim_nan_columns = []

for col in cotegorical_columns:

    if drop_columns.count(col) < 1:

        cotegorical_nim_nan_columns.append(col)

print(cotegorical_nim_nan_columns)
cotegorical_have_nan_columns = []

cotegorical_columns = []

for col in cotegorical_nim_nan_columns:

    percent = 100 * (data[col].size - data[col].value_counts().sum()) / data[col].size

    if percent > 0:

        cotegorical_have_nan_columns.append(col)

    else:

        cotegorical_columns.append(col)

print(cotegorical_columns)
data.MSZoning.unique()
new_msz = {"RL": 4, "RM": 3, "RH": 2, "FV": 1, "C (all)": 0} #categorical variable



X_train["MSZoning"] = [new_msz[val] for val in data.MSZoning]
data.Street.unique()
new_Street = {"Pave": 0, "Grvl": 1} #binary variable



X_train["Street"] = [new_Street[val] for val in data.Street]
data.Utilities.unique()
new_Utilities= {"NoSeWa": 0, "AllPub": 1} #binary variable



X_train["Utilities"] = [new_Utilities[val] for val in data.Utilities]
new_LandSlope = {'Gtl': 2, 'Mod' : 1, 'Sev' : 0}



X_train["LandSlope"] = [new_LandSlope[val] for val in data.LandSlope]
Condition = {'Norm': 6, 'Feedr': 1, 'PosN': 8, 'Artery': 0, 'RRAe': 3, 'RRNn': 5, 'RRAn': 2, 'PosA': 7, 'RRNe': 4}



X_train["Condition1"] = [Condition[val] for val in data.Condition1]

X_train["Condition2"] = [Condition[val] for val in data.Condition2]

BldgType = {'1Fam': 4, '2fmCon': 0, 'Duplex': 3, 'TwnhsE': 2, 'Twnhs': 0}



X_train["BldgType"] = [BldgType[val] for val in data.BldgType]
HouseStyle = {'2Story':5, '1Story':0, '1.5Fin':2, '1.5Unf':1, 'SFoyer':4, 'SLvl':3, '2.5Unf': 6,'2.5Fin': 7}



X_train["HouseStyle"] = [HouseStyle[val] for val in data.HouseStyle]

RoofStyle = {'Gable':4, 'Hip':3, 'Gambrel':2, 'Mansard':5, 'Flat': 0, 'Shed':1}

X_train["RoofStyle"] = [RoofStyle[val] for val in data.RoofStyle]

roof = pd.DataFrame()

roof["material"] = ['CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran', 'Tar&Grv', 'Roll', 'ClyTile']

roof["min_price"] = [9, 30, 6, 4.50, 3, 2.50, 1.50, 10]

roof["max_price"] = [12, 75, 12, 9, 4, 5, 2.50, 18]

roof["year"] = [50, 75, 30, 20, 20, 20, 10, 100]

roof["$_year"] = ((roof.max_price + roof.min_price) / 2) /roof.year

roof.sort_values(by=["$_year", "year"])
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

sns.barplot(x="material" , y="$_year", data=roof)
plt.figure(figsize=(10,5))

sns.barplot(x="material" , y="year", data=roof)
roof.sort_values(by=["$_year", "year"])
RoofMatl = {'CompShg': 4, 'WdShngl': 0, 'Metal': 2, 'WdShake':1, 'Membran': 6, 'Tar&Grv': 5, 'Roll': 3, 'ClyTile': 7}



X_train["RoofMatl"] = [RoofMatl[val] for val in data.RoofMatl]

Foundation = {'PConc':5, 'CBlock':2, 'BrkTil':1, 'Wood':0, 'Slab':4, 'Stone':3}



X_train["Foundation"] = [Foundation[val] for val in data.Foundation]

Heating = {'GasA':4, 'GasW':5, 'Grav':3, 'Wall':0, 'OthW':2, 'Floor':1}

X_train["Heating"] = [Heating[val] for val in data.Heating]

meaning = {'Ex': 4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0}

X_train["HeatingQC"] = [meaning[val] for val in data.HeatingQC]

X_train["KitchenQual"] = [meaning[val] for val in data.KitchenQual]

CentralAir = {'Y':1, 'N':0}

X_train["CentralAir"] = [CentralAir[val] for val in data.CentralAir]

PavedDrive = {'Y':2, 'N':0, 'P':1}

X_train["PavedDrive"] = [PavedDrive[val] for val in data.PavedDrive]

cotegorical_have_nan_columns
meaning = {'Ex': 5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, "Na":0}



X_train["BsmtQual"] = data.BsmtQual.fillna("Na")

X_train["BsmtQual"] = [meaning[val] for val in X_train.BsmtQual]



X_train["BsmtCond"] = data.BsmtCond.fillna("Na")

X_train["BsmtCond"] = [meaning[val] for val in X_train.BsmtCond]



X_train["GarageQual"] = data.GarageQual.fillna("Na")

X_train["GarageQual"] = [meaning[val] for val in X_train.GarageQual]



X_train["GarageCond"] = data.GarageCond.fillna("Na")

X_train["GarageCond"] = [meaning[val] for val in X_train.GarageCond]
meaning = {'No':1, 'Gd':4, 'Mn':2, 'Av':3, "Na":0}



X_train["BsmtExposure"] = data.BsmtExposure.fillna("Na")

X_train["BsmtExposure"] = [meaning[val] for val in X_train.BsmtExposure]
meaning = {'GLQ':6, 'ALQ':5, 'Unf':1, 'Rec':3, 'BLQ':4, "Na":0, 'LwQ':2}



X_train["BsmtFinType1"] = data.BsmtFinType1.fillna("Na")

X_train["BsmtFinType1"] = [meaning[val] for val in X_train.BsmtFinType1]



X_train["BsmtFinType2"] = data.BsmtFinType2.fillna("Na")

X_train["BsmtFinType2"] = [meaning[val] for val in X_train.BsmtFinType2]
X_train["MasVnrType"] = data.MasVnrType.fillna("None")

X_train["Electrical"] = data.Electrical.fillna("Mix")

X_train["GarageType"] = data.GarageType.fillna("Na")



one_hot_columns = ["MasVnrType", "Electrical", "GarageType"]

for col in one_hot_columns:

    uniques = list(X_train[col].unique())

    for unique in uniques:

        X_train[col + '_' + str(unique)] = [int(val == unique) for val in X_train[col]]

    X_train = X_train.drop(col, axis=1)
meaning = {'RFn':2, 'Unf':1, 'Fin':3, "Na":0}



X_train["GarageFinish"] = data.GarageFinish.fillna("Na")

X_train["GarageFinish"] = [meaning[val] for val in X_train.GarageFinish]
one_hot_columns = ["LotShape", "LandContour", "LotConfig", "Neighborhood", 'Exterior1st', 

                   'Exterior2nd', "Functional", "SaleType", "SaleCondition"]



for col in one_hot_columns:

    uniques = list(data[col].unique())

    for unique in uniques:

        X_train[col + '_' + str(unique)] = [int(val == unique) for val in data[col]]
def X_standardization(data, col):

    mean = data[col].mean()

    std = data[col].std()

    data[col] -= mean

    data[col] /= std



def y_standardization(y):

    mean = y.mean()

    std = y.std()

    y -= mean

    y /= std
def z_normalization_X(data, col):

    N = data[col].size

    x_mean = data[col].mean()

    s_x = 1 / N * sum(abs(data[col]-x_mean))

    data[col] = (data[col] - x_mean) / s_x



def z_normalization_y(y):

    N = y.size

    y_mean = y.mean()

    s_y = 1 / N * sum(abs(y-y_mean))

    y = (y - y_mean) / s_y
def normalization_X(data, col):

    x_min = data[col].min()

    x_max = data[col].max()

    data[col] = (data[col] - x_min) / (x_max - x_min)



def normalization_y(y):

    y_min = y.min()

    y_max = y.max()

    y = (y - y_min) / (y_max - y_min)
X_train_stand = X_train

y_train_stand = y_train



X_train_z_norm = X_train

y_train_z_norm = y_train



X_train_norm = X_train

y_train_norm = y_train



for col in list(X_train.columns):

    X_standardization(X_train_stand, col)

    z_normalization_X(X_train_z_norm, col)

    normalization_X(X_train_norm, col)



y_standardization(y_train_stand)

z_normalization_y(y_train_z_norm)

normalization_y(y_train_norm)
X_train.head()
for col in list(X_train.columns):

    if X_train[col].isnull().sum() > 0:

        print(col, X_train[col].isnull().sum())
X_train.shape
y_train.shape
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
np.random.seed(42)



model = Sequential()



model.add(Dense(500, input_dim=214, activation="relu"))

model.add(Dense(700, activation="relu"))

model.add(Dense(1000, activation="relu"))

model.add(Dense(1))



model.compile(loss="mse", optimizer="adam", metrics=['mae'])



print(model.summary())
history_stand = model.fit(X_train_stand, y_train_stand, batch_size=1, epochs=10, validation_split=0.2, verbose=1)
model_1 = Sequential()



model_1.add(Dense(500, input_dim=214, activation="relu"))

model_1.add(Dense(700, activation="relu"))

model_1.add(Dense(1000, activation="relu"))

model_1.add(Dense(1))



model_1.compile(loss="mse", optimizer="adam", metrics=['mae'])

history_z_norm = model_1.fit(X_train_z_norm, y_train_z_norm, batch_size=1, epochs=10, validation_split=0.2, verbose=1)
model_2 = Sequential()



model_2.add(Dense(500, input_dim=214, activation="relu"))

model_2.add(Dense(700, activation="relu"))

model_2.add(Dense(1000, activation="relu"))

model_2.add(Dense(1))



model_2.compile(loss="mse", optimizer="adam", metrics=['mae'])

history_norm = model_2.fit(X_train_norm, y_train_norm, batch_size=1, epochs=10, validation_split=0.2, verbose=1)
print(history_stand.history.keys())
history_data_stand = pd.DataFrame([history_stand.history["loss"],

                                   history_stand.history["val_loss"],

                                   history_stand.history["mae"],

                                   history_stand.history["val_mae"]]).T

history_data_stand = history_data_stand.rename(columns={0:"loss", 1: "val_loss", 2:"mae", 3:"val_mae"})

history_data_stand.head()
history_data_z_norm = pd.DataFrame([history_z_norm.history["loss"],

                                   history_z_norm.history["val_loss"],

                                   history_z_norm.history["mae"],

                                   history_z_norm.history["val_mae"]]).T

history_data_z_norm = history_data_z_norm.rename(columns={0:"loss", 1: "val_loss", 2:"mae", 3:"val_mae"})

history_data_z_norm.head()
history_data_norm = pd.DataFrame([history_norm.history["loss"],

                                   history_norm.history["val_loss"],

                                   history_norm.history["mae"],

                                   history_norm.history["val_mae"]]).T

history_data_norm = history_data_norm.rename(columns={0:"loss", 1: "val_loss", 2:"mae", 3:"val_mae"})

history_data_norm.head()
import seaborn as sns

import matplotlib.pyplot as plt



plt.figure(figsize=(10, 5))

sns.lineplot(data=history_data_stand)
plt.figure(figsize=(10, 5))

sns.lineplot(data=history_data_z_norm)
plt.figure(figsize=(10, 5))

sns.lineplot(data=history_data_norm)