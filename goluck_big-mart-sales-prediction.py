"""

The project predicts the sales at Big Mart.

The demo code below shows predictions based on Scikit-Learn Linear Regression function.

...Improve the model accuracy by including extra hyperparameters on the model below, engineering new features,

using alternate regression models such as Lasso or Ridge Regression.

...May the best coder win.

"""

# linear algebra operations

import numpy as np 

# data handling processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd 

from pandas import Series, DataFrame

#Data visualization

import matplotlib.pyplot as plt

#Correlation analysis and data visualization toolbox.

import seaborn as sns

#Popular ML framework and library.

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import data

train = pd.read_csv("../input/Train.csv")

test = pd.read_csv("../input/Test.csv")

train.head()
#View the relationships between individual features as well as between the features and the dependent variable.

corr = train.corr()

#Heatmaps show the strenght of relationships between the data features.

sns.heatmap(corr, annot=True)
#Get an overall feel of the dataset.

desc = train.describe() 

desc
#visualizing the data to detect missing values

sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis')
#Impute the missing values in the Out_let size column.

train.Outlet_Size = train.Outlet_Size.fillna('High')
#Impute Missing values in the Item_Weight column by the mean value.

train.Item_Weight = train.Item_Weight.fillna(12.86)
#visualizing the data to detect/Review presence of missing values

sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis')
#Changing categorical data(Outlet_Location_Type) to numerical data.

train["Outlet_Location_Type"] = pd.Categorical(train["Outlet_Location_Type"])

Outlet_Location_Type_categories = train.Outlet_Location_Type.cat.categories

train["Outlet_Location_Type"] = train.Outlet_Location_Type.cat.codes
#Changing categorical data(Outlet_Size) to numerical data.

train["Outlet_Size"] = pd.Categorical(train["Outlet_Size"])

Outlet_Size_categories = train.Outlet_Size.cat.categories

train["Outlet_Size"] = train.Outlet_Size.cat.codes
#Changing categorical data(Item_Fat_Content) to numerical data.

train["Item_Fat_Content"] = pd.Categorical(train["Item_Fat_Content"])

Item_Fat_Content_categories = train.Item_Fat_Content.cat.categories

train["Item_Fat_Content"] = train.Item_Fat_Content.cat.codes
#Changing categorical data(Item_Type) to numerical data.

train["Item_Type"] = pd.Categorical(train["Item_Type"])

Item_Type_categories = train.Item_Type.cat.categories

train["Item_Type"] = train.Item_Type.cat.codes
#Changing categorical data(Item_Type) to numerical data.

train["Outlet_Type"] = pd.Categorical(train["Outlet_Type"])

Outlet_Type_categories = train.Outlet_Type.cat.categories

train["Outlet_Type"] = train.Outlet_Type.cat.codes
#Changing categorical data(Outlet_Identifier) to numerical data.

train["Outlet_Identifier"] = pd.Categorical(train["Outlet_Identifier"])

Outlet_Identifier_categories = train.Outlet_Identifier.cat.categories

train["Outlet_Identifier"] = train.Outlet_Identifier.cat.codes

#Changing categorical data(Outlet_Establishment_Year) to numerical data.

train["Outlet_Establishment_Year"] = pd.Categorical(train["Outlet_Establishment_Year"])

Outlet_Establishment_Year_categories = train.Outlet_Establishment_Year.cat.categories

train["Outlet_Establishment_Year"] = train.Outlet_Establishment_Year.cat.codes
#Training data after categorical to numerical conversions.

train.head()
#A correlation analysis will indicate additional relationships in the dataset

corr = train.corr()

sns.heatmap(corr, annot=True)
from sklearn.model_selection import train_test_split

#Introducing Polynomial regression.

poly_features= PolynomialFeatures(degree=3)
X = train.drop(['Item_Identifier','Item_Outlet_Sales'], axis = 1)

x_poly = poly_features.fit_transform(X)

Y = train.Item_Outlet_Sales

x_train, x_cv, y_train, y_cv = train_test_split(x_poly,Y, test_size =0.2)
#Give the regression function an easy name to use.

lr = LinearRegression()
#Ridge regression function with hyperparameters.

#rg = Ridge(alpha=0.025, normalize=True)
#Lasso regression function with hyperparameters.

#ls = Lasso(alpha=0.03, normalize=True)
#Train the model based on the training dataset.

lr.fit(x_train,y_train)
#View the model performance using the validation dataset.

lr.score(x_cv,y_cv)
#Evaluating the prediction error.

poly_pred = lr.predict(x_cv)

rmse = np.sqrt(mean_squared_error(y_cv,poly_pred))

rmse
#the square error

r2 = r2_score(y_cv,poly_pred)

r2
#calculating mse

mse = np.mean((poly_pred - y_cv)**2)

mse
#Testing the model performance on Unseen data.

test.head()
#Changing categorical data to numerical data in test data.

test["Outlet_Location_Type"] = pd.Categorical(test["Outlet_Location_Type"])

Outlet_Location_Type_categories = test.Outlet_Location_Type.cat.categories

test["Outlet_Location_Type"] = test.Outlet_Location_Type.cat.codes

###

test["Outlet_Size"] = pd.Categorical(test["Outlet_Size"])

Outlet_Size_categories = test.Outlet_Size.cat.categories

test["Outlet_Size"] = test.Outlet_Size.cat.codes

###

test["Item_Fat_Content"] = pd.Categorical(test["Item_Fat_Content"])

Item_Fat_Content_categories = test.Item_Fat_Content.cat.categories

test["Item_Fat_Content"] = test.Item_Fat_Content.cat.codes

###

test["Item_Type"] = pd.Categorical(test["Item_Type"])

Item_Type_categories = test.Item_Type.cat.categories

test["Item_Type"] = test.Item_Type.cat.codes

###

test["Outlet_Type"] = pd.Categorical(test["Outlet_Type"])

Outlet_Type_categories = test.Outlet_Type.cat.categories

test["Outlet_Type"] = test.Outlet_Type.cat.codes

###

test["Outlet_Identifier"] = pd.Categorical(test["Outlet_Identifier"])

Outlet_Identifier_categories = test.Outlet_Identifier.cat.categories

test["Outlet_Identifier"] = test.Outlet_Identifier.cat.codes

###

test["Outlet_Establishment_Year"] = pd.Categorical(test["Outlet_Establishment_Year"])

Outlet_Establishment_Year_categories = test.Outlet_Establishment_Year.cat.categories

test["Outlet_Establishment_Year"] = test.Outlet_Establishment_Year.cat.codes
#View test data after transformantions.

test.head()
test_en = test.drop('Item_Identifier', axis=1)
#visualizing the data to detect/Review presence of missing values

sns.heatmap(test_en.isnull(),yticklabels=False,cmap='viridis')
#Impute Missing values in the Item_Weight column by the mean value.

test_en.Item_Weight = test_en.Item_Weight.fillna(12.86)

test_en.head()
#Tranform the test features to align with the prediction polynomial.

test_poly = poly_features.fit_transform(test_en)
#Generate a prediction list based on the model.

pred2 = lr.predict(test_poly)

pred2
#Prepare and add the prediction to the test dataframe.

predictions = pd.Series(pred2)

test = test.assign(Predicted_Sales = predictions.values)

test.head(30)