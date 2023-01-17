import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

sns.set()
sns.set_style("darkgrid")
# We read not only train data but also test data
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df.columns
# What we have to do is analyze target
train_df["SalePrice"].describe()
sns.distplot(train_df["SalePrice"])
# Skewness and kurtosis
print("Skewness: %f" % train_df["SalePrice"].skew())
print("Kurtosis: %f" % train_df["SalePrice"].kurt())
# Relationship with numerical variables
var = "GrLivArea"
data = pd.concat([train_df["SalePrice"], train_df[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice", ylim=(0, 800000))
# scatter plot totalbsmtsf / saleprice
var = "TotalBsmtSF"
data = pd.concat([train_df["SalePrice"], train_df[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice", ylim=(0, 800000))
# Relationship with categorical features
# box plot ovarallqual / saleprice
var = "OverallQual"
data = pd.concat([train_df["SalePrice"], train_df[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
var = "YearBuilt"
data = pd.concat([train_df["SalePrice"], train_df[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
# correlation matrix (heatmap style)
corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
# "SalePrice correlation matrix (zoomed heatmap style)
# saleprice correlation matrix
k = 10 # number of variables for heatmap
cols = corrmat.nlargest(k, "SalePrice")["SalePrice"].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True,
                 fmt=".2f", annot_kws={"size": 10}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.show()
# scatterplot
sns.set()
cols = ["SalePrice", "OverallQual", "GrLivArea", "GarageCars", 
        "TotalBsmtSF", "FullBath", "YearBuilt"]
sns.pairplot(train_df[cols], size=2.5)
plt.show()
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum() / train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
# dealing with missing data
train_df = train_df.drop((missing_data[missing_data["Total"] > 1]).index, 1)
train_df = train_df.drop(train_df.loc[train_df["Electrical"].isnull()].index)
print(train_df.isnull().sum().max()) # just check there's no missing data
# What we drop from train_df
(missing_data[missing_data["Total"] > 1])
# Standardizing data
saleprice_scaled = StandardScaler().fit_transform(train_df["SalePrice"][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()[:10]]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()[-10:]]
print("outer range (low) of the distribution")
print(low_range)
print("\nouter range (high) of the distribution")
print(high_range)
# bivariate analysis saleprice / grlivarea
var = "GrLivArea"
data = pd.concat([train_df["SalePrice"], train_df[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice", ylim=(0, 800000))
# deleting points
print(train_df.sort_values(by="GrLivArea", ascending=False)[:2])
train_df = train_df.drop(train_df[train_df["Id"] == 1299].index)
train_df = train_df.drop(train_df[train_df["Id"] == 524].index)
# Observe scatter plot again
var = "GrLivArea"
data = pd.concat([train_df["SalePrice"], train_df[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice", ylim=(0, 800000))
# bivariate analysis saleprice / TotalbsmtSF
var = "TotalBsmtSF"
data = pd.concat([train_df["SalePrice"], train_df[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice", ylim=(0, 800000))
# bistogram and normal probability plot
sns.distplot(train_df["SalePrice"], fit=norm)
fig = plt.figure()
res = stats.probplot(train_df["SalePrice"], plot=plt)

# applying log transformation
train_df["SalePrice"] = np.log(train_df["SalePrice"])
# transformed bistogram and normal probability plot
sns.distplot(train_df["SalePrice"], fit=norm)
fig = plt.figure()
res = stats.probplot(train_df["SalePrice"], plot=plt)
# bistogram and normal probability plot
sns.distplot(train_df["GrLivArea"], fit=norm)
fig = plt.figure()
res = stats.probplot(train_df["GrLivArea"], plot=plt)
# data transformation
train_df["GrLivArea"] = np.log(train_df["GrLivArea"])
# transformed bistogram and normal probability plot
sns.distplot(train_df["GrLivArea"], fit=norm)
fig = plt.figure()
res = stats.probplot(train_df["GrLivArea"], plot=plt)
# histogram and normal probability plot
sns.distplot(train_df["TotalBsmtSF"], fit=norm)
fig = plt.figure()
res = stats.probplot(train_df["TotalBsmtSF"], plot=plt)
# create column fot new variable (one is enough because it is a binary categorical feature)
# if area > 0  it gets 1, for area == 0 it gets 0
train_df["HasBsmt"] = pd.Series(len(train_df["TotalBsmtSF"]), index=train_df.index)
train_df["HasBsmt"] = 0
train_df.loc[train_df["TotalBsmtSF"] > 0, "HasBsmt"] = 1
# transform data
train_df.loc[train_df["HasBsmt"] == 1, "TotalBsmtSF"] = np.log(train_df["TotalBsmtSF"])
# histogram and normal probability plot
sns.distplot(train_df[train_df["TotalBsmtSF"] > 0]["TotalBsmtSF"], fit=norm)
fig = plt.figure()
res = stats.probplot(train_df[train_df["TotalBsmtSF"] > 0]["TotalBsmtSF"], plot=plt)
# scatter plot
plt.scatter(train_df["GrLivArea"], train_df["SalePrice"])
# scatter plot
plt.scatter(train_df[train_df["TotalBsmtSF"] > 0]["TotalBsmtSF"], train_df[train_df["TotalBsmtSF"] > 0]["SalePrice"])
# First of all, We have to check test DataFrame information
test_df.info()
# Of course, we have to drop the same columns of test DataFrame as well as train DataFrame
test_df = test_df.drop((missing_data[missing_data["Total"] > 1]).index, 1)
test_df = test_df.drop(test_df.loc[test_df["Electrical"].isnull()].index)
# After dropping some columns, what changes occured to the test DataFrame
test_df.info()
total = test_df.isnull().sum().sort_values(ascending=False)
percent = (test_df.isnull().sum() / test_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(16)
# There are more null objects in test DataFrame than in train DataFrame
# I don't know whether this method is correct. 
# So if you think there are better ways, please share! 
# Categorical Feature --> fillna() by mode()
categorical_features = ["Utilities", "Functional", "KitchenQual", "Exterior1st",
                       "Exterior2nd", "SaleType"]

for feature in categorical_features:
    freq_port = test_df[feature].mode()[0]
    test_df[feature] = test_df[feature].fillna(freq_port)
# Numerical Feature --> fillna() by median()
numerical_features = ["BsmtHalfBath", "BsmtFullBath", "GarageCars", "TotalBsmtSF",
                      "BsmtUnfSF", "BsmtFinSF1", "BsmtFinSF2", "GarageArea"]

for feature in numerical_features:
    freq_port = test_df[feature].median()
    test_df[feature] = test_df[feature].fillna(freq_port)
# data transformation, like train DataFrame
test_df["GrLivArea"] = np.log(test_df["GrLivArea"])
# create column fot new variable (one is enough because it is a binary categorical feature)
# if area > 0  it gets 1, for area == 0 it gets 0
test_df["HasBsmt"] = pd.Series(len(test_df["TotalBsmtSF"]), index=test_df.index)
test_df["HasBsmt"] = 0
test_df.loc[test_df["TotalBsmtSF"] > 0, "HasBsmt"] = 1
# transform data
test_df.loc[test_df["HasBsmt"] == 1, "TotalBsmtSF"] = np.log(test_df["TotalBsmtSF"])
# convert categorical variables into dummy
# If we do pd.get_dummies to train DataFrame and test DataFrame respectively,
# the columns are different from each other. Therefore, we can't construct models.
# Thus, we must use pd.get_dummies to the whole dataset as Pratik Singh said at the comment.
# I greatly appreciate his comment.
# This cell is the same as his comment.
train_len = len(train_df)
dataset = pd.concat([train_df, test_df])
dataset = pd.get_dummies(dataset)

train_df = dataset[:train_len]
test_df = dataset[train_len:]
# We have to check whether the number of columns are the same.
print(train_df.head(5))
print("---------------------------------------")
print(test_df.head(5))
# OK. now let's move to make model
# First of all, I want to try RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# X = All values except price, y = price
X_train, y_train = train_df.drop("SalePrice", axis=1), train_df.SalePrice

# make model
RFR = RandomForestRegressor(n_estimators=100)
RFR.fit(X_train, y_train)

# show score
RFR.score(X_train, y_train)
# Support Vector machine
from sklearn.svm import SVR

svc = SVR(gamma=10)
svc.fit(X_train, y_train)

svc.score(X_train, y_train)
# GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

grbt = GradientBoostingRegressor()
grbt.fit(X_train, y_train)

grbt.score(X_train, y_train)
# DecisionTreeRegressor
# I don't think it is correct.
# So please make a comment if you notice some mistakes
"""
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)

print(tree.score(X_train, y_train))
"""
# We have to drop target of test_df
test_df = test_df.drop("SalePrice", axis=1)
# Let's make submission file
# I used GradientBoostingRegressor to make my model
# Regarding this cell, I refered to ...
preds = grbt.predict(test_df)
 
# we have to change log -> normal "SalePrice" 
np.exp(preds)
 
# Numpy array -> pandas.Series
preds = pd.Series(np.exp(preds))
 
# Make submission dataframe
submit = pd.concat([test_df.Id, preds], axis=1)
 
# change columns' name
submit.columns = ['Id', 'SalePrice']
 
# write to csv_file
submit.to_csv('submit_grbt.csv', index=False)
