# Importing Libraries

import numpy as np

import pandas as pd

import xgboost

import math

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from scipy.stats import pearsonr

from time import time

from sklearn.metrics import r2_score

import warnings

warnings.filterwarnings('ignore')

from scipy.stats import norm 

from sklearn.metrics import mean_absolute_error
data = pd.read_csv('../input/kc-house-data/kc_house_data.csv')
# Copying data to another dataframe df_train for our convinience so that original dataframe remain intact.



df_train=data.copy()

df_train.rename(columns ={'price': 'SalePrice'}, inplace =True)

# Now lets see the first five rows of the data

data.head()
print(data.shape) # shape 

# Check the data types of each column

print(data.dtypes)
data.nunique()
# Check any number of columns with NaN or missing values 

print(data.isnull().any().sum(), ' / ', len(data.columns))

# Check any number of data points with NaN

print(data.isnull().any(axis=1).sum(), ' / ', len(data))
# As id and date columns are not important to predict price so we are discarding it for finding correlation

features = data.iloc[:,3:].columns.tolist()

target = data.iloc[:,2].name
# Finding Correlation of price woth other variables to see how many variables are strongly correlated with price

correlations = {}

for f in features:

    data_temp = data[[f,target]]

    x1 = data_temp[f].values

    x2 = data_temp[target].values

    key = f + ' vs ' + target

    correlations[key] = pearsonr(x1,x2)[0]
# Printing all the correlated features value with respect to price which is target variable

data_correlations = pd.DataFrame(correlations, index=['Value']).T

data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]
plt.figure(figsize = (12,8))

plot_p = sns.lmplot(x="sqft_living", y="price", aspect=1.8,data=data, hue="floors", fit_reg=False)

plot_p.set_titles("Floors by sqft_living and price", fontsize=15)

plot_p.set_xlabels("sqft Living")

plot_p.set_ylabels("Price(US)")

plt.show()
#Pairplots to visualize strong correlation

sns.set()

cols = [ 'price','sqft_living', 'grade', 'sqft_above', 'view', 'bathrooms','bedrooms','sqft_basement']

sns.pairplot(data[cols], height = 3.5)

plt.show();
saleprice = data["price"].values

plt.scatter(x = range(saleprice.shape[0]), y = np.sort(saleprice))

plt.title('saleprice Vs observations')

plt.xlabel('Observations')

plt.ylabel('saleprice')
print(data["bedrooms"].unique().shape[0])

data["bedrooms"].value_counts(sort = False).plot.bar()

plt.title("bedrooms")

plt.xlabel("Number Of Bedrooms")

plt.ylabel("Houses Count")
print(data["grade"].unique().shape[0])

data["grade"].value_counts(sort = False).plot.bar()

plt.title("Grades")

plt.xlabel("Number Of Grades")

plt.ylabel("Houses Count")


print(data["condition"].unique().shape[0])

data["condition"].value_counts(sort = False).plot.bar()

plt.title("Condition")

plt.xlabel("Condition")

plt.ylabel("Houses Count")
new_data = df_train[['sqft_living','grade', 'sqft_above', 

                     'sqft_living15','bathrooms','view','sqft_basement','waterfront','yr_built','lat','bedrooms','long']]

X = new_data.values

y = df_train.SalePrice.values
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error as MSE

#splitting data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=568)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
rand_regr = RandomForestRegressor(n_estimators=400,random_state=0)

rand_regr.fit(X_train, y_train)

print(rand_regr.score(X_test,y_test))

predictions = rand_regr.predict(X_test)
from sklearn.metrics import explained_variance_score

explained_variance_score(predictions,y_test)
decision=DecisionTreeRegressor()

decision.fit(X_train, y_train)

print(decision.score(X_test,y_test))

prediction2 = decision.predict(X_test)
explained_variance_score(prediction2,y_test)
ada=AdaBoostRegressor(n_estimators=50, learning_rate=0.2,loss='exponential').fit(X_train, y_train)

print(ada.score(X_test,y_test))

prediction3=ada.predict(X_test)
explained_variance_score(prediction3,y_test)
Gdt=GradientBoostingRegressor(n_estimators=400, max_depth=5, loss='ls',min_samples_split=2,

                              learning_rate=0.1).fit(X_train, y_train)

print(Gdt.score(X_test,y_test))



prediction4 = Gdt.predict(X_test)
explained_variance_score(prediction4,y_test)
XGB = XGBRegressor(n_estimators=100, learning_rate=0.09, gamma=0).fit(X_train, y_train)

print(XGB.score(X_test,y_test))

prediction5 = XGB.predict(X_test)
explained_variance_score(prediction5,y_test)