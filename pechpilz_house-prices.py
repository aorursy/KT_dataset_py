# Data manipulation and wrangling libaries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import norm, skew



# Plotting libaries

import matplotlib.pyplot as plt

import seaborn as sns



# Models

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import ElasticNet



# Supporting actors

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import RobustScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Training data

train = pd.read_csv("../input/train.csv").drop(labels="Id", axis = "columns")

y_train = train["SalePrice"] # True labels

train.drop(labels="SalePrice", axis = "columns", inplace = True)

n_train = train.shape[0] # Number of training samples



# Testing data

test = pd.read_csv("../input/test.csv").drop(labels="Id", axis = "columns")

n_test = test.shape[0] # Number of testing samples



# Concatenating for easier data cleaning

df = pd.concat((train, test)).reset_index(drop=True)
# NaN Values

df.isna().sum()[df.isna().sum()!=0]
columns_fill = ["Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", 

                "BsmtFinType2", "FireplaceQu","GarageType", "GarageQual", "GarageCond", "PoolQC",

                "Fence","MiscFeature","GarageFinish"]

for col in columns_fill:

    df[col].fillna("None", inplace = True)

    

# NaN Values still left

df.isna().sum()[df.isna().sum()!=0]
columns_fill = ["MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","BsmtFullBath","BsmtHalfBath",

               "GarageCars","GarageArea"]



for col in columns_fill:

    df[col].fillna(0, inplace = True)

    

# NaN Values still left

df.isna().sum()[df.isna().sum()!=0]
columns_fill = ["Exterior1st","Exterior2nd","MSZoning","Electrical","KitchenQual","Functional","SaleType","Utilities"]



for col in columns_fill:

    most_common = df[col].mode()[0]

    percentage = round(df[col].value_counts(normalize=True)[most_common]*100,2) # Percentage of values that equal the mode

    print("{}% of values for {} have the value {}".format(percentage,col.ljust(12),most_common)) 

    

    df[col].fillna(most_common, inplace = True)

    

# NaN Values still left

df.isna().sum()[df.isna().sum()!=0]
# Calculate percentage of samples where the YearBuilt == GarageYrBlt

print((df["YearBuilt"] == df["GarageYrBlt"]).value_counts(normalize=True))



# For the missing values, insert the YearBuilt

df["GarageYrBlt"][df["GarageYrBlt"].isna()] = df["YearBuilt"][df["GarageYrBlt"].isna()]
train["LFisNa"] = train["LotFrontage"].isna()

train["SalePrice"] = y_train



sns.set(style="whitegrid")

ax = sns.boxplot(x = "LFisNa", y = "SalePrice", data = train, showfliers = False)
# Create scatterplot with linear regression between LotFronage and root of LotArea

df["LotAreaRoot"] = df["LotArea"].apply(np.sqrt)

ax = sns.regplot(x = "LotAreaRoot", y = "LotFrontage",data=df)

ax.set(xlim=(25, 200))

ax.set(ylim=(0, 200))



# Use a linear regression to guess the missing LotFrontage values

X = df[df["LotFrontage"].notna()]["LotAreaRoot"].values.reshape(-1, 1)

y = df[df["LotFrontage"].notna()]["LotFrontage"]

reg = LinearRegression().fit(X, y)

print("Regression with R^2 = {}".format(reg.score(X, y))) 



X_missing = df[df["LotFrontage"].isna()]["LotAreaRoot"].values.reshape(-1, 1)

df["LotFrontage"][df["LotFrontage"].isna()] = reg.predict(X_missing)



# Drop the LotAreaRoot since we don't need it anymore

df.drop(labels = "LotAreaRoot", axis = "columns", inplace = True)
# NaN Values still left

df.isna().sum()[df.isna().sum()!=0]
# Numerical features

num_features = ["LotFrontage","LotArea","YearBuilt","YearRemodAdd","MasVnrArea",

                "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF",

                "2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath",

                "FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd",

                "Fireplaces","GarageYrBlt","GarageCars","GarageArea","WoodDeckSF",

                "OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea",

                "MiscVal","MoSold","YrSold"]

# Ordinal features

ord_features = ["OverallQual","OverallCond","ExterQual","ExterCond","BsmtQual",

                "BsmtCond","BsmtFinType1","BsmtFinType2","HeatingQC","KitchenQual",

                "Functional","FireplaceQu","GarageQual","GarageFinish","GarageCond",

                "Fence","PoolQC"]

# Categorical features

cat_features = list((set(df.columns.values)-set(ord_features))-set(num_features))
# Make some sexy plots of the distributon of saleprice and log(saleprice)

plot_dims = (11.7, 5)

fig, axes = plt.subplots(1,2, figsize=plot_dims)

ax1 = sns.distplot(y_train, fit = norm, ax = axes[0])

ax2 = sns.distplot(y_train.apply(np.log), fit = norm, ax = axes[1])

ax2.set(xlabel='log(SalePrice)')

fig.show()



# Take the log of saleprice and use these values to train the model

y_train = y_train.apply(np.log)
df[ord_features].head()
# Most ordinal features are rated from Poor to Excellent

rating_to_int = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}

columns_to_fill = ["ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC",

                   "KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"]

for col in columns_to_fill:

    df[col] = df[col].map(rating_to_int)

    

# BmstFinType1 & 2

rating_to_int = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}

df["BsmtFinType1"] = df["BsmtFinType1"].map(rating_to_int)

df["BsmtFinType2"] = df["BsmtFinType2"].map(rating_to_int)



# Functional 

rating_to_int = {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7}

df["Functional"] = df["Functional"].map(rating_to_int)



# GarageFinish

rating_to_int = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}

df["GarageFinish"] = df["GarageFinish"].map(rating_to_int)



# Fence

rating_to_int = {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}

df["Fence"] = df["Fence"].map(rating_to_int)
df["MSSubClass"] = df["MSSubClass"].astype("object")

print(df.shape)

df = pd.get_dummies(df, drop_first = True)

df.head()
X_train = df.iloc[0:n_train]

X_test  = df.iloc[-n_test:]

parameters = {'alpha': [0.1,0.5,1,2],'l1_ratio': [0.1,0.3,0.5,0.7,0.9],'normalize': [True, False]}

elnet_clf = ElasticNet(random_state=0)

grid_search_cv = GridSearchCV(elnet_clf, parameters, cv=5)

grid_search_cv.fit(X_train, y_train)

grid_search_cv.best_estimator_
# Get mean squared error for the training set

y_pred = grid_search_cv.predict(X_train)

mean_squared_error(y_train, y_pred)
parameters = {'elnet_clf__alpha': [0.1,0.5,1,2], 'elnet_clf__l1_ratio': [0.1,0.3,0.5,0.7,0.9], 'elnet_clf__normalize': [True, False]}

pipe = Pipeline([

            ("scaler", RobustScaler()),

            ("elnet_clf", ElasticNet())

        ])



grid_search_cv_pipe = GridSearchCV(pipe, parameters, cv=5)

grid_search_cv_pipe.fit(X_train, y_train)
# Get mean squared error for the training set

y_pred = grid_search_cv_pipe.predict(X_train)

mean_squared_error(y_train, y_pred)
y_pred = np.exp(grid_search_cv.predict(X_test)) # Convert log(SalePrice) to SalePrice
results = pd.DataFrame(

    {'Id': (X_test.index.to_series() + 1),

     'SalePrice': y_pred})



results.to_csv("submission.csv",index=False)
results