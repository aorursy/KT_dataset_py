import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
estate_df = pd.read_csv('../input/real-estate-price-prediction/Real estate.csv')
estate_df.head()
#setting the 'No' column as index column
estate_df.index = estate_df['No']
estate_df = estate_df.drop('No',axis = 1)  #axis = 1 for column
estate_df.head()
#renaming the variables
new_colm_names = {'X1 transaction date' : 'x1tdate','X2 house age': 'x2hage',
                  'X3 distance to the nearest MRT station' : 'x3MRT',
                  'X4 number of convenience stores' : 'x4stores',
                  'X5 latitude' : 'x5lat',
                  'X6 longitude' : 'x6long','Y house price of unit area' : 'yprice'}
estate_df = estate_df.rename(columns = new_colm_names)
estate_df.head()
estate_df.isnull().sum() #cheking for null values
estate_df = estate_df.astype(float)  #coverting all the values to float
estate_df.head()
#X = estate_df.iloc[:,:-1] #step 1: taking all the independent features except the target value
X = estate_df.iloc[:,:-2] #step 2: after checking the p value below, removing x6long as it's P value is greater than 0.05 which show that it is multicollinear
X.head()
Y = estate_df.yprice #target value
#splitting the dataset into training and testing dataset
xtrain, xtest, ytrain ,ytest = train_test_split(X, Y, random_state = 0,test_size = .20)
#ytrain.head()
#defining the model
model = LinearRegression()  

#fitting the model
model.fit(xtrain,ytrain)
estate_df.corr()
#range of correaltion coef is -1 to 1
#relation between x3MRT and x6 long is very high i.e they might be collinear
#therefore further checking, so that we can drop one of the columns
model_cor = estate_df.drop(['x6long'], axis = 1) 
model_cor.corr()
#all the corr values are within a proper range
#using OLS model
X = sm.add_constant(X)
X.head()
model_ols = sm.OLS(Y,X).fit() #fitting the model
model_ols.summary()
#before dropping x6long the p value of x6long was 0.7 which is greater than 0.05(actual range of p)
#therefore removing the x6long column from X above
#using linearregression model
y_predict = model.predict(xtest)
res = r2_score(ytest,y_predict) #checking the goodness of fit using r2 method
res*100
#using variance_inflation_factor to check multicollinearity

model_before = estate_df
model_after = estate_df.drop(['x6long'], axis = 1) #as we can see the p value of 'x6long' is greater than 0.05 and it has high collinearity  with x3MRT


x1 = sm.add_constant(model_before)
x2 = sm.add_constant(model_after)

series_before = pd.Series([variance_inflation_factor(x1.values, i) for i in range(x1.shape[1])], index = x1.columns)
series_after = pd.Series([variance_inflation_factor(x2.values, i) for i in range(x2.shape[1])], index = x2.columns)

print('DATA BEFORE')
print('-'*100)
print(series_before)

print('DATA AFTER REMOVING COLLINEAR VARIABLES')
print('-'*100)
print(series_after)

#notice that x3MRT value is reduces to ~2 as it was ~5 before which isn't expected
#Therefore there's no multicollinearity present in the dataset