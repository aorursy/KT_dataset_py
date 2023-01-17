import pandas as pd 

import numpy as np

import xgboost as xgb

import matplotlib.pyplot as plt
train_dataset = pd.read_csv("../input/train.csv")

test_dataset = pd.read_csv("../input/test.csv")
train_dataset.head()
train_dataset.shape # 1460x81
test_dataset.drop_duplicates().shape # no duplicates
train_dataset.columns
print(set(train_dataset.columns) - set(test_dataset.columns))

# we need to split SalePrice
train_Y = train_dataset["SalePrice"]

train_dataset = train_dataset.drop(["SalePrice"], axis=1) 
# There is also no need of Id column

train_dataset = train_dataset.drop(["Id"], axis=1)

test_dataset = test_dataset.drop(["Id"], axis=1)
# Next step: to combine these two datasets for processing

dataset = pd.concat([train_dataset, test_dataset])

dataset.shape
# function-helper for getting null columns

def getNullsOf(dataframe):

    df = dataframe.isnull().sum().sort_values(ascending=False)

    df = df[df > 0] # keeping values that > 0

    return df
null_cols = getNullsOf(dataset)

# we'll remove all cols with more than 50% null. Because there are cannot be helpful due to missing majority of data

(null_cols/len(dataset) * 100)
# we'll drop these values:

to_drop = ["PoolQC", "MiscFeature", "Alley", "Fence"]

dataset = dataset.drop(to_drop, axis=1) 

# we droped 4 unnecessary columns

dataset.shape
numeric_cells = dataset._get_numeric_data()

getNullsOf(numeric_cells)
# filling numeric columns with mean() data

print("Before:\n", dataset["LotFrontage"].head(20))

def fillWithMean(name):

    numeric_cells[name] = numeric_cells[name].fillna(numeric_cells[name].mean())

    dataset[name] = numeric_cells[name]

for name in getNullsOf(numeric_cells).index.tolist():

    fillWithMean(name)

print("After:\n", dataset["LotFrontage"].head(20))
# categorical cells

categorical_cells = list(set(dataset.columns) - set(numeric_cells.columns))

categorical_cells[:5] #printing first 5 categorical columns
categorical_df = dataset[categorical_cells] # dataframe of cat. values

null_cat_cells = getNullsOf(categorical_df)

null_cat_cells
null_cat_cells = null_cat_cells.index.tolist()

dataset[null_cat_cells].describe()

# here we can see top values with which we'll replace nulls
for cell in null_cat_cells:

    dataset[cell] = dataset[cell].fillna(dataset[cell].mode()[0])
getNullsOf(dataset) # all data is cleaned!
train_dataset = dataset.iloc[:train_dataset.shape[0], :]

test_dataset = dataset.iloc[train_dataset.shape[0]:, :]
print(train_dataset.shape, test_dataset.shape)
train_Y.describe()
train_Y.plot(kind="hist")

plt.show()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
for category in categorical_cells:

    train_dataset[category] = encoder.fit_transform(train_dataset[category])

train_dataset.head()
xgb_regressor = xgb.XGBRegressor()

xgb_regressor.fit(train_dataset,  train_Y)
from sklearn.metrics import mean_squared_error

y_pred = xgb_regressor.predict(train_dataset)

np.sqrt(mean_squared_error(train_Y, y_pred))