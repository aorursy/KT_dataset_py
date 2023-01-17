import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
dataset = pd.read_csv(r"/kaggle/input/expenses.csv")
# Making a list of missing value types
# missing_values = ["n/a", "na", "--"]
# df = pd.read_csv(r"/kaggle/input/expenses.csv", na_values = missing_values)
dataset.head()
dataset.info()
dataset.describe()
dataset.shape
dataset.count()
dataset.isna().sum()
dataset.isna().sum().sum()
dataset.isna().sum().any()
dataset.notna().sum()
dataset[dataset.isna().any(axis=1)]
dataset[dataset["bmi"].isna()]
dataset[dataset["bmi"].isna()].index.values.tolist()
datasetcopy = dataset.copy()
# Replace missing values with a number
datasetcopy["bmi"].fillna(5)
datasetcopy.isna().sum()
# Replace missing values with a number
datasetcopy["bmi"].fillna(5, inplace=True)
datasetcopy.isna().sum()
datasetcopy = dataset.copy()
# Replace using median 
median = datasetcopy["bmi"].median()
print(median)
datasetcopy["bmi"].fillna(median, inplace=True)
datasetcopy = dataset.copy()
from sklearn.impute import SimpleImputer
# define the imputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# transform the dataset
transformed_values = imputer.fit_transform(datasetcopy["bmi"].values.reshape(-1, 1) )
# count the number of NaN values in each column

datasetcopy["bmi"] = transformed_values
datasetcopy.isna().sum()
datasetcopy.loc[16:17,:]
datasetcopy.loc[[17]]
datasetcopy.loc[17]
datasetcopy.iloc[16:18,:]
datasetcopy.iloc[[16,18,35],[1,2]]
datasetcopy.loc[16:17,"bmi"]
datasetcopy.loc[[16,17,35],"bmi"]
datasetcopy.loc[[16,17,35],:]
datasetcopy.loc[datasetcopy["sex"]=="male", ["charges", "bmi"]]
datasetcopy = dataset.copy()
datasetcopy.loc[16:17,:]
imputer = KNNImputer(n_neighbors=5)
datasetcopy["bmi"] = imputer.fit_transform(datasetcopy["bmi"].values.reshape(-1, 1))
datasetcopy.loc[16:17,:]
