# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





##visual imports

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



##Missing data

from sklearn.impute import SimpleImputer



##Categorical Encoding

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder



##Feature Scaling

from sklearn.preprocessing import StandardScaler



##Splitting data

from sklearn.model_selection import train_test_split



#Splitting Data

from sklearn.model_selection import train_test_split



# Feature Scaling

from sklearn.preprocessing import StandardScaler





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
diamond_data = pd.read_csv("/kaggle/input/diamonds/diamonds.csv")
diamond_data.head()
diamond_data.info()
diamond_data.describe()
print("Cut Breakdown\n")

print(diamond_data["cut"].value_counts())

print("_"*20)

print("Color Breakdown\n")

print(diamond_data["color"].value_counts())

print("_"*20)

print("Clarity Breakdown\n")

print(diamond_data["clarity"].value_counts())
diamond_data.drop(["Unnamed: 0"], axis = 1, inplace = True)
sns.pairplot(diamond_data)
plt.figure(figsize=(12, 7))

correl = diamond_data.corr()

sns.heatmap(correl, annot = True)
print("0 value x: {}".format(diamond_data['x'].isin([0]).sum()))

print("0 value y: {}".format(diamond_data['y'].isin([0]).sum()))

print("0 value z: {}".format(diamond_data['z'].isin([0]).sum()))
diamond_data[["x","y","z"]] = diamond_data[["x","y","z"]].replace(0,np.NaN)

diamond_data.isnull().sum()
diamond_data.dropna(inplace=True)

diamond_data.shape
diamond_data.describe()
plt.title('X Distribution Plot')

sns.distplot(diamond_data["x"], bins = 50)
plt.title('Y Distribution Plot')

sns.distplot(diamond_data["y"], bins = 50)
plt.title('Z Distribution Plot')

sns.distplot(diamond_data["z"], bins = 50)
x_rep =diamond_data['x'] < 9.5

y_rep =diamond_data['y'] < 20

z_rep =diamond_data['z'] < 20

diamond_data['x'].where(x_rep,np.NaN, inplace = True)

diamond_data['y'].where(y_rep,np.NaN, inplace = True)

diamond_data['z'].where(z_rep,np.NaN, inplace = True)

diamond_data.isnull().sum()
diamond_data.dropna(inplace=True)

diamond_data.shape
sns.pairplot(diamond_data)
diamond_data['vol'] = diamond_data['x']*diamond_data['y']*diamond_data['z']

diamond_data.head()
diamond_data.drop(['x','y','z'],axis =1, inplace=True)
sns.pairplot(diamond_data)
print('The mean volume in the set is: {:.2f}'.format(diamond_data['vol'].mean()))

print('The maximum volume in the set is: {:.2f}'.format(diamond_data['vol'].max()))
plt.figure(figsize=(12, 7))

correl = diamond_data.corr()

sns.heatmap(correl, annot = True)
from scipy.stats import kurtosis

from scipy.stats import skew

print('excess kurtosis of normal distribution (should be 0): {}'.format(skew(diamond_data['price'])))

print('skewness of normal distribution (should be 0): {}'.format(kurtosis(diamond_data['price'])))
sns.boxplot(x = "price", y = "cut", data = diamond_data)
sns.boxplot(x = "price", y = "color", data = diamond_data)
sns.boxplot(x = "price", y = "clarity", data = diamond_data)
sns.jointplot(x = "price", y = "carat", data = diamond_data)
plt.figure(figsize=(12, 7))

sns.distplot(diamond_data["price"], bins = 50)
diamond_data["price"] = diamond_data["price"].apply(np.log)

diamond_data["carat"] = diamond_data["carat"].apply(np.log)

diamond_data["vol"] = diamond_data["vol"].apply(np.log)
plt.figure(figsize=(12, 7))

plt.title('Price Distribution')

sns.distplot(diamond_data["price"])

print('excess kurtosis of normal distribution (should be 0): {}'.format(skew(diamond_data['price'])))

print('skewness of normal distribution (should be 0): {}'.format(kurtosis(diamond_data['price'])))
plt.figure(figsize=(12, 7))

plt.title('Volume Distribution')

sns.distplot(diamond_data["vol"])

print('excess kurtosis of normal distribution (should be 0): {}'.format(skew(diamond_data['vol'])))

print('skewness of normal distribution (should be 0): {}'.format(kurtosis(diamond_data['vol'])))
plt.figure(figsize=(12, 7))

plt.title('Carat Distribution')

sns.distplot(diamond_data["carat"])

print('excess kurtosis of normal distribution (should be 0): {}'.format(skew(diamond_data['carat'])))

print('skewness of normal distribution (should be 0): {}'.format(kurtosis(diamond_data['carat'])))
sns.pairplot(diamond_data)
cut_mapping = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}

color_mapping = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E":6, "D":7}

clarity_mapping = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2":6,"VVS1":7,"IF":8}



diamond_data['cut'] = diamond_data['cut'].map(cut_mapping)

diamond_data['color'] = diamond_data['color'].map(color_mapping)

diamond_data['clarity'] = diamond_data['clarity'].map(clarity_mapping)



diamond_data.head()
# Simple & Multi Linear Regression

from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor



# Polynomial Regression 

from sklearn.preprocessing import PolynomialFeatures



#Support Vector Regression

from sklearn.svm import SVR



#CART Regression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



#K-Nearest Neighbours

from sklearn.neighbors import KNeighborsRegressor



#XG Boost

from xgboost import XGBRegressor

X_train, X_test, y_train, y_test = train_test_split(diamond_data.drop('price',axis=1), 

                                                    diamond_data['price'], test_size=0.25, 

                                                    random_state=101)
from sklearn.metrics import r2_score



LR = LinearRegression()

LR.fit(X_train,y_train)

y_pred = LR.predict(X_test)



R2 = r2_score(y_test, y_pred)



n=diamond_data.shape[0]

p=diamond_data.shape[1] - 1



adj_rsquared = 1 - (1 - R2) * ((n - 1)/(n-p-1))





from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

print('MSE:', metrics.mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("r2: ", r2_score(y_test, y_pred))

print("Adjusted r2:", adj_rsquared )



from sklearn.model_selection import cross_val_score



accuracies = cross_val_score(estimator = LR, X = X_train, y = y_train, cv = 10)

R2accuracies = cross_val_score(estimator = LR, X = X_train, y = y_train, cv = 10, scoring = 'r2')

MSEaccuracies = cross_val_score(estimator = LR, X = X_train, y = y_train, cv = 10, scoring = 'neg_mean_squared_error')



n=diamond_data.shape[0]

p=diamond_data.shape[1] - 1



MSE = MSEaccuracies.mean()*-1

R2 = R2accuracies.mean()*100

adj_rsquared = 1 - (1 - R2) * ((n - 1)/(n-p-1))



print("MSE: {:.2f}".format(MSE))

print("RMSE: {:.2f}".format((MSE**0.5)))

print("R2: {:.2f}".format((R2)))

print("Adjusted R2: {:.2f}".format(adj_rsquared))
from sklearn.model_selection import ShuffleSplit

cv_split = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )
MLA = [LinearRegression(), DecisionTreeRegressor(),KNeighborsRegressor(), XGBRegressor(), RandomForestRegressor()]

MLA_columns = ["MLA Name","Mean Price","MAE","MSE", "RMSE", "R2","Adjusted R2"]

MLA_compare = pd.DataFrame(columns = MLA_columns)

n=diamond_data.shape[0]

p=diamond_data.shape[1] - 1



row_index = 0

for alg in MLA:

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    

    alg.fit(X_train, y_train)

    pred = alg.predict(X_test)

    

    

    R2accuracies = cross_val_score(estimator = alg, X = X_train, y = y_train, cv = cv_split)

    MSEaccuracies = cross_val_score(estimator = alg, X = X_train, y = y_train, cv = cv_split, scoring = 'neg_mean_squared_error')

    MAEaccuracies = cross_val_score(estimator = alg, X = X_train, y = y_train, cv = cv_split, scoring = 'neg_mean_absolute_error')

    MSE = MSEaccuracies.mean()*-1

    R2 = R2accuracies.mean()*100

    MAE = MAEaccuracies.mean()*-1

    adj_rsquared = 1 - (1 - R2) * ((n - 1)/(n-p-1))

    

    MLA_compare.loc[row_index, "Mean Price"] = pred.mean()

    MLA_compare.loc[row_index, "MAE"] = MAE

    MLA_compare.loc[row_index, "MSE"] = int(MSE)

    MLA_compare.loc[row_index, "RMSE"] = MSE**0.5

    MLA_compare.loc[row_index, "R2"] = R2

    MLA_compare.loc[row_index, "Adjusted R2"] = adj_rsquared



    row_index +=1

                                                       



MLA_compare.sort_values(by = ["R2"], ascending = False, inplace = True)

MLA_compare
plt.title("MLA Accuracy Rank")

sns.barplot(x = "R2", y = "MLA Name", data = MLA_compare)