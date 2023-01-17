import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import itertools

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
homePrices = pd.read_csv("../input/train.csv")

df = pd.DataFrame(homePrices)



%matplotlib inline



# Histogram of home prices

plt.figure(figsize=(15,5))

sns.distplot(df["SalePrice"], kde=True, rug=True);

#plt.hist(df["SalePrice"], bins=18)

plt.title("Distribution of all home sale prices in the dataset")

plt.xlabel("Sale Price")

plt.ylabel("Number of Homes")

plt.show()
fig, (ax1,ax2) = plt.subplots(2, figsize=(16,10))

sns.set_style("whitegrid")

sns.despine(left=False)



sns.boxplot("Neighborhood","SalePrice",data=homePrices, ax=ax1)

sns.countplot(x="Neighborhood", data=homePrices,ax=ax2)

for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)



plt.show()
sns.set(context="notebook", font="monospace")



# Correlation matrix of numerical features

corrmat=homePrices[["SalePrice", "LotFrontage", "LotArea", "OverallQual", "OverallCond","YearBuilt","YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "WoodDeckSF", "OpenPorchSF",

                  "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MoSold", "YrSold", "GrLivArea",

                  "GarageArea","TotalBsmtSF","1stFlrSF","FullBath", "TotRmsAbvGrd"]].corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15, 10))



# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=.8, square=True, cmap='coolwarm')

plt.show()

fig, axs = plt.subplots(ncols=3, figsize=(16,5))



sns.regplot(x='OverallQual', y='SalePrice', data=homePrices, ax=axs[0])

sns.regplot(x='GrLivArea', y='SalePrice', data=homePrices, ax=axs[1])

sns.regplot(x='TotalBsmtSF',y='SalePrice', data=homePrices, ax=axs[2])



plt.show()
f, ax = plt.subplots(figsize=(10, 5))

sns.regplot(x='EnclosedPorch', y='SalePrice', data=homePrices)

plt.show()
fig, axs = plt.subplots(ncols=3, figsize=(16,5))

plt.xticks(rotation=45)

sns.stripplot(x=homePrices["CentralAir"], y=homePrices["SalePrice"],jitter=True, ax=axs[0])

sns.stripplot(x=homePrices["Foundation"], y=homePrices["SalePrice"],jitter=True, ax=axs[1])

sns.stripplot(x=homePrices["HouseStyle"], y=homePrices["SalePrice"],jitter=True, ax=axs[2])



plt.title("Sale Price vs Streets");
fig, axs = plt.subplots(ncols=2, figsize=(16,5))

sns.violinplot("OverallQual","SalePrice",data=homePrices, ax = axs[0])

sns.violinplot("OverallCond","SalePrice",data=homePrices, ax = axs[1])



sns.despine(trim=True)
sns.stripplot(x=homePrices["MSSubClass"], y=homePrices["SalePrice"],jitter=True)

plt.title("Sale Price vs Streets");
nans=pd.isnull(homePrices).sum()

nans[nans>(len(homePrices)*.1)]
homePrices = homePrices.drop("Alley", 1)

homePrices = homePrices.drop("FireplaceQu", 1)

homePrices = homePrices.drop("PoolQC", 1)

homePrices = homePrices.drop("Fence", 1)

homePrices = homePrices.drop("MiscFeature", 1)

homePrices = homePrices.drop("LotFrontage", 1)
nans = pd.isnull(homePrices).sum()

nans[nans > 0] 
fig, (ax1,ax2) = plt.subplots(2, figsize=(7,5))

sns.set_style("whitegrid")



sns.violinplot("MasVnrType","SalePrice",data=homePrices, ax = ax1)

sns.countplot(x="MasVnrType", data=homePrices,ax=ax2)

for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)

sns.despine(trim=True)

plt.show()



homePrices["Electrical"].describe()
fig, (ax1,ax2) = plt.subplots(2, figsize=(7,5))

sns.set_style("whitegrid")



sns.violinplot("Electrical","SalePrice",data=homePrices, ax = ax1)

sns.countplot(x="Electrical", data=homePrices,ax=ax2)

for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)

sns.despine(trim=True)



plt.show()
# Drop bsmt and garage variables as discussed above

homePrices = homePrices.drop("GarageType", 1)

homePrices = homePrices.drop("GarageYrBlt", 1)

homePrices = homePrices.drop("GarageFinish", 1)

homePrices = homePrices.drop("GarageQual", 1)

homePrices = homePrices.drop("GarageCond", 1)

homePrices = homePrices.drop("BsmtQual", 1)

homePrices = homePrices.drop("BsmtCond", 1)

homePrices = homePrices.drop("BsmtExposure", 1)

homePrices = homePrices.drop("BsmtFinType1", 1)

homePrices = homePrices.drop("BsmtFinType2", 1)
# Replaces NAs in scalar fields with the mean value

homePrices = homePrices.fillna(homePrices.mean())



# Replaces NAs in categorical fields with the mode value

homePrices.loc[(homePrices["Electrical"].isnull()), "Electrical"] = 'SBrkr'

homePrices.loc[(homePrices["MasVnrType"].isnull()), "MasVnrType"] = 'None'
nans = pd.isnull(homePrices).sum()

nans[nans>0]
# Create a list of all categorical variables

categories_to_numeric = list()

for columns in homePrices.columns:

    if homePrices[columns].dtype == "O":

        categories_to_numeric.append(columns)

        

# Turn variable string values into integer values

homePrices_with_dummies = pd.get_dummies(data = homePrices)



# Let "SalesPrice" be it's own variable

label = homePrices_with_dummies["SalePrice"]

noSale = homePrices_with_dummies.drop("SalePrice",1)



# Log transform

logged_homePrices_with_dummies = np.log(noSale)

logged_label = np.log(label)



# Remove -infs in the dataset

logged_homePrices_with_dummies[logged_homePrices_with_dummies==-np.inf] = 0
from sklearn.decomposition import PCA



pca = PCA(n_components=20, whiten=True)

a = pca.fit(logged_homePrices_with_dummies)

variance = pd.DataFrame(pca.explained_variance_ratio_)

print(variance)
plt.figure(1, figsize=(9, 6))

plt.clf()

plt.axes([.2, .2, .7, .7])



dat = range(1, 21)

plt.plot(dat, variance, marker='o')



plt.axis('tight')

plt.xlabel('Number of components')

plt.ylabel('Percent variance explained')

plt.title("Scree Plot Showing Percent Variability Explained by Each Eigenvalue")



plt.show()

label.describe()
class1 = label[label > 214000]

class3 = label[label <= 129275]

print(len(class1))

print(len(class3))
pca = PCA(n_components=10, whiten=True)

transf = pca.fit_transform(logged_homePrices_with_dummies)

transf_label = label.T



# Transpose the class labels

class1T = class1.T

class3T = class3.T

transf.shape
plt.plot(transf[class1T.index,0],transf[class1T.index,1], 'o', markersize=7, color='blue', alpha=0.5, label='Upper Quartile Sales Price')

plt.plot(transf[class3T.index,0],transf[class3T.index,1], '^', markersize=7, color='green', alpha=0.5, label='Lower Quartile Sales Price')



plt.xlabel('PCA 1')

plt.ylabel('PCA 2')

plt.xlim([-3,3])

plt.ylim([-3,3])

plt.legend()

plt.title("PCA Plot of Homes Prices Data")



plt.show()
homePrices["PricePerSqFt"] = homePrices["SalePrice"]/homePrices["GrLivArea"]
grouped = homePrices['PricePerSqFt'].groupby(df['Neighborhood']).median()

grouped = grouped.sort_values(ascending=False)

names = list(grouped.index)



plt.figure(figsize=(15,5))

neighborhoodMedians = list(grouped)



N = len(neighborhoodMedians)

ind = np.arange(N)

sns.barplot(ind, neighborhoodMedians)

plt.xlabel("Neighborhood")

plt.ylabel("Median Sale Price")

plt.title("\'Most-Desired\' Neighborhoods Ranked By Median Sale Price")

plt.xticks(ind, names, rotation="25")

plt.show()

homePrices["Price / (Overall Quality * Square Feet)"] = homePrices["PricePerSqFt"]/homePrices["OverallQual"]
fig, (ax1,ax2) = plt.subplots(2, figsize=(16,20))

sns.set_style("whitegrid")

sns.despine(left=False)



sns.boxplot("PricePerSqFt", "Neighborhood", data=homePrices, ax=ax1)

sns.boxplot("Price / (Overall Quality * Square Feet)", "Neighborhood", data=homePrices,ax=ax2)

for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=0)



plt.show()

plt.figure(figsize=(15,5))

sns.swarmplot(x="OverallQual", y="PricePerSqFt", hue="Neighborhood", data=homePrices)

plt.legend()

plt.show()