import numpy as np
import sklearn as sk
print (np.array(range(10)))
print (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
import time

n = 10**6

start = time.time()
foo = list(range(n))
for i in range(len(foo)):
    foo[i] = foo[i] + 1
end = time.time()
print ("For loop:", end - start)
# slightly better: list comprehensions
start = time.time()
foo = [elem + 1 for elem in foo]
end = time.time()
print(end - start)
foo = np.array(range(n))
start = time.time()
### vectorized code goes here
end = time.time()
print(end - start) 
inds = foo < 10
print (foo[inds])
print (foo[~inds])
print (np.random.uniform(0, 1, size=5)) #10 draws from Uniform(0, 1)
print (np.random.normal(0, 100, size=5)) #10 draws from Normal(0, 100)
print (np.random.choice(["A", "B", "C", "D"], size=5)) #10 draws with replacement
import pandas as pd

df = pd.DataFrame({
        "A":list(range(10)),
        "B":np.random.randn(10),
        "C":7,
        "D":["this is a string" for i in range(10)]
    })

df
df.iloc[3:7] #iloc for "integer location". Can use this for selecting rows of a dataframe.
df["B"][[3, 4, 8]]
df["B"][3]
# Kaggle Competition: Predicting housing prices from data about the house
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques
data = pd.read_csv("../input/train.csv") 
data.head()
# Features we'll look at:
# OverallQual: Rates the overall material and finish of the house
# OverallCond: Rates the overall condition of the house
# GarageArea: Size of garage in square feet
# YrSold: Year Sold (YYYY)
# LotFrontage: Linear feet of street connected to property
# LotArea: Lot size in square feet
# YearBuilt: Original construction date

data_subset = data[["SalePrice", "OverallQual", "OverallCond", "GarageArea", 
              "YrSold", "LotArea", "LotFrontage", "YearBuilt"]].copy()
data_subset.head()
import matplotlib.pyplot as plt
%matplotlib inline 
#^this is a magic line for IPython. Don't worry about it, it's not really that interesting.
plt.hist(data_subset["SalePrice"], bins=100)
plt.title("Marginal Distribution of Sale Price")
plt.xlabel("Sale Price (Dollars)")
plt.ylabel("Count")
plt.show()
x = np.log(data_subset["SalePrice"])
mu = np.mean(x)
sigma = np.std(x)
n = len(x)
plt.hist(np.random.normal(loc=mu, scale=sigma, size=n), 
         alpha=0.5, 
         color = "blue",
         bins=100)
plt.hist(x, 
         alpha=0.5, 
         bins=100,
        color="#229911")
plt.show()
data_subset["LogSalePrice"] = np.log(data_subset["SalePrice"])
data_subset.head()
plt.scatter(data_subset["GarageArea"], data_subset["LogSalePrice"])
plt.title("Garage Area X Sale Price")
plt.xlabel("Garage Area")
plt.show()
# Quiz: 
# Add a column HasGarage which is True IFF the house has a garage

for col in data_subset.columns:
    numnan = np.isnan(data_subset[col]).sum()
    if numnan > 0:
        print (col, numnan)
# What to do with missing values?

# 1. Ditch the feature entirely
data_feature_ditched = data_subset[["SalePrice", "OverallQual", "OverallCond", "GarageArea", 
              "YrSold", "LotArea", "YearBuilt"]].copy()
print (data_feature_ditched.shape)
data_feature_ditched.head()
# 1 (Cont'd, Improved) 

data_ditched = data_subset.drop("LotFrontage", axis=1)
print (data_ditched.shape)
data_ditched.head()
# 2. Ditch rows where values are missing

# np.isnan(data): returns a new dataframe of bools
# np.sum(data, axis=1): returns the sum of ROWS of the input dataframe (axis = 0 would be columns)
data_clean = data_subset[~np.isnan(data_subset["LotFrontage"])].copy()
print (data_clean.shape)
data_clean.head()
# 2 (cont'd)
# More general and pandorable. Maybe I want to check NaNs for lots of rows at once!

# data.dropna(how='any')    #to drop if any value in the row has a nan
# data.dropna(how='all')    #to drop if all values in the row are nan
data_clean = data_subset.dropna(how="any") #dropna returns a new dataframe
print (data_clean.shape)
data_clean.head()
# 3. Impute missing values
# Doing this manually is pretty awful tbh. 
data_imputed = data_subset.copy()
data_imputed = data_imputed.fillna(data_imputed.mean())
data_imputed.head()
pd.isnull(data_subset["LotFrontage"]).sum()
pd.isnull(data_imputed["LotFrontage"]).sum()
df = pd.DataFrame(np.random.randn(10,3))
df
df2 = df
df2[0][0] = df[0][0] + 1
print (df2[0][0] == df[0][0]) #Modifying df2 modifies df
df2 = df.copy()
df2[0][0] = df[0][0] + 1
print (df2[0][0] == df[0][0]) #Fixed
df3 = df[1]
df3[0] = df[1][0] + 1
df3[0] == df[1][0] #Oh noes
df3 = df[1].copy()
df3[0] = df[1][0] + 1
df3[0] == df[1][0] #fixed
from sklearn.model_selection import train_test_split

features = ["OverallQual", "OverallCond", "GarageArea", 
              "YrSold", "LotArea", "LotFrontage", "YearBuilt"]
XTrain, XTest, yTrain, yTest = train_test_split(
    data_imputed[features], 
    data_imputed["LogSalePrice"], 
    test_size=100)
# uppercase X denotes that X is a matrix, and lowercase y denotes a vector
#XTest, XVal, yTest, yVal = train_test_split(XTest, yTest, test_size=50)
from sklearn import linear_model 

# Initialize linear regression object
model = linear_model.LinearRegression(fit_intercept=True)

# fit your model to training data
model.fit(XTrain, yTrain)

# Make predictions using the testing set
yPreds = model.predict(XTest)
from sklearn.metrics import r2_score

print ("Test R-Squared:", r2_score(y_true = yTest, y_pred = yPreds))
print ("Training R-Squared:", r2_score(y_true = yTrain, y_pred = model.predict(XTrain)))
# Root Mean Squared Error
np.sqrt(np.mean((yTest - yPreds)**2))
# Original basis
np.sqrt(np.mean((np.exp(yTest) - np.exp(yPreds))**2))
for i in range(len(features)):
    print (features[i], model.coef_[i])
    
print ("Intercept", model.intercept_)
plt.scatter(yPreds, yPreds - yTest)
plt.xlabel("Predicted Log of Sale Price")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()