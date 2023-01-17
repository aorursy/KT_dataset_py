# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm 

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error



pd.options.display.max_columns = None



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/house-prices-advanced-regression-techniques/housetrain.csv")

dataset = dataset.set_index('Id')
dataset.describe(include='all')
plot = plt.figure(figsize=(19, 15))

plt.matshow(dataset.corr(), fignum=plot.number)

plt.xticks(range(dataset.describe().shape[1]), dataset.describe().columns, fontsize=14, rotation=90)

plt.yticks(range(dataset.describe().shape[1]), dataset.describe().columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title("Correlation Matrix", fontsize=16, pad=100)
dataset.corr()
#Columns which have good correlation with SalesPrice (target variable)

cols = ['OverallQual', 'YearBuilt', 'TotalBsmtSF', 

        '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'SalePrice']
newCorrMat = np.corrcoef(dataset[cols].values.T)

size = plt.figure(figsize=(19,15))

sns.set(font_scale=1.25)

heatM = sns.heatmap(newCorrMat, annot=True, cbar=True, square=True, fmt='0.2f', 

                    xticklabels=cols, yticklabels=cols)

plt.show()
newCols = ['TotalBsmtSF', 'GrLivArea', 'GarageCars', 'YearBuilt', 'OverallQual', 'FullBath', 'SalePrice']

newDf = dataset[newCols]

sns.pairplot(newDf)
newDf.isna().count()
def scatterPlot(x, y, xlabel, ylabel, title):

    plt.scatter(x=x, y=y)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title(title)

    plt.show()
scatterPlot(newDf.TotalBsmtSF, newDf.SalePrice, 'Total Basement Surface area', 'Sales Price', 

           'Total Basement surface area vs Sales Price')
newDf = newDf.drop(newDf[newDf.TotalBsmtSF > 6000].index)
scatterPlot(newDf.GrLivArea, newDf.SalePrice, 'Ground living area', 'Sales Price', 

           'Ground living area vs Sales Price')
newDf = newDf.drop(newDf[(newDf.GrLivArea>4000) & (newDf.SalePrice<200000)].index)
scatterPlot(newDf.GarageCars, newDf.SalePrice, 'Garage Cars', 'Sales Price', 

           'Garage cars vs Sales Price')
scatterPlot(newDf.YearBuilt, newDf.SalePrice, 'Year built', 'Sales Price', 

           'Year build vs Sales Price')
scatterPlot(newDf.OverallQual, newDf.SalePrice, 'Overall Quality', 'Sales Price', 

           'Overall quality vs Sales Price')
scatterPlot(newDf.FullBath, newDf.SalePrice, 'Full Bath', 'Sales Price', 

           'Full bath vs Sales Price')
sns.distplot(newDf.SalePrice, fit=norm)
newDf['SalePrice'] = np.log(newDf.SalePrice)
sns.distplot(newDf.GrLivArea, fit=norm)
newDf['GrLivArea'] = np.log(newDf.GrLivArea)
sns.distplot(newDf.TotalBsmtSF, fit=norm)
newDf['TotalBsmtSF'] = newDf['TotalBsmtSF'].apply(lambda x: 1 if x==0 else x)

newDf['TotalBsmtSF'] = np.log(newDf['TotalBsmtSF'])
X = newDf.iloc[:, :-1]

y = newDf.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8, random_state=10)
#Random Forest Regression

RF = RandomForestRegressor(n_estimators=10000)

RF.fit(X_train, y_train)

y_predict = RF.predict(X_test)
mean_squared_error(y_test, y_predict)
boosting = GradientBoostingRegressor(n_estimators=10000, learning_rate=0.01)

boosting.fit(X_train, y_train)

y_predict1 = boosting.predict(X_test)
mean_squared_error(y_test, y_predict1)