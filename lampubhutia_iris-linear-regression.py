# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats, integrate

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

%matplotlib inline

pd.options.display.float_format = '{:.2f}'.format

plt.rcParams['figure.figsize'] = (8, 6)

plt.rcParams['font.size'] = 14
iris=pd.read_csv("../input/Iris.csv")

iris.head()
# Copy of iris dataset

iris_data= iris.drop(['Id'],axis=1)

iris_data.head()
# Descriptive statistics 

df_summary=iris_data.describe()

df_summary
boxplot=iris_data.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False, colormap='bwr', figsize=(15,10))



print('From the boxplot, it is visible that only sepal width is having the outlier, whereas sepallength, petallength, petalwidth having datapoints within min and maxrange.')

print('It is visible that values in sepallength and sepalwidth are tightly distributed and IQ range is small, whereas in petallength and petalwidth the values are distibuted widely,so IQ range are high.')
# Correlation Matrix for Iris_dataset

iris_data.corr()
# Scatterplot with best fit line to explore relation b/w sepallength(dependent)& sepalwidth,petalwidth,petallengh(Independent)



a=sns.lmplot(x='SepalWidthCm', y='SepalLengthCm', data=iris_data, aspect=1.5, scatter_kws={'alpha':0.2})

b=sns.lmplot(x='PetalWidthCm', y='SepalLengthCm', data=iris_data, aspect=1.5, scatter_kws={'alpha':0.2})

c=sns.lmplot(x='PetalLengthCm', y='SepalLengthCm', data=iris_data, aspect=1.5, scatter_kws={'alpha':0.2})  



#Combine scatterplot to explore relation b/w sepallength, sepalwidth, petalwidth, petallength



scatter=pd.plotting.scatter_matrix(iris_data,figsize=(15,10))

#Scatterplot EDA b/w sepallength, sepalwidth, petalwidth, petallength and how species are reacting to it.



scatterplot=sns.pairplot(iris_data,hue="Species")
#Checking the relationship between Sepal width and Sepal length

input_cols = ['SepalWidthCm']

output_variable = ['SepalLengthCm']

X = iris_data[input_cols]

Y = iris_data[output_variable]

#Creating the Linear Regression Model

linreg = LinearRegression()

linreg.fit(X,Y)

print (linreg.intercept_)

print (linreg.coef_)

print('\n')

print('Sepal.length = -0.2088*Sepal.Width + 6.481')

print('\n')

print('Holding constant fixed, a 1 centimeter increase in sepalwidth lead to a decrease in Sepalength by 0.208centimeter')

#Checking the relationship between Petal length and Sepal length

input_cols = ['PetalLengthCm']

output_variable = ['SepalLengthCm']

X = iris_data[input_cols]

Y = iris_data[output_variable]

#Creating the Linear Regression Model

linreg = LinearRegression()

linreg.fit(X,Y)

print (linreg.intercept_)

print (linreg.coef_)

print('\n')

print('Sepal.length = 0.409*Petal.length + 4.305')

print('\n')

print('Holding constant fixed, a 1 centimeter increase in petallength lead to a increase in Sepalength by 0.409 centimeter')
#Checking the relationship between Petal width and Sepal length

input_cols = ['PetalWidthCm']

output_variable = ['SepalLengthCm']

X = iris_data[input_cols]

Y = iris_data[output_variable]

#Creating the Linear Regression Model

linreg = LinearRegression()

linreg.fit(X,Y)

print (linreg.intercept_)

print (linreg.coef_)

print('\n')

print('Sepal.length = 0.887*Petal.Width + 4.77')

print('\n')

print('Holding constant fixed, a 1 centimeter increase in petalwidth lead to a increase in Sepalength by 0.887 centimeter')
#Checking the relationship between Petal width,Petal Length and Sepal length (Multivariate Model)

input_cols = ['PetalWidthCm','PetalLengthCm','SepalWidthCm']

output_variable = ['SepalLengthCm']

X = iris_data[input_cols]

Y = iris_data[output_variable]

#Creating the Linear Regression Model

linreg = LinearRegression()

linreg.fit(X,Y)

print (linreg.intercept_)

print (linreg.coef_)



print('\n')



print('Sepal.length = -0.56*Petal.Width + 0.71*Petal.Length + 0.65*Sepal.Width + 1.84')



print('\n')



print('Holding all other independent variable fixed, a 1 centimeter decrease in petalwidth lead to a increase in Sepalength by -0.56cm')

print('Holding all other independent variable fixed, a 1 centimeter increase in petallength lead to a increase in Sepalength by 0.71cm')

print('Holding all other independent variable fixed, a 1 centimeter increase in sepalwidth lead to a increase in Sepalength by 0.65cm')

print('\n')

print('In given mutivariate model, the petalwidth is showing a negative relation with sepal.length, which proves a level of correlation or dependency between petallength and petalwidth.')

print('This is proved by the correlation matrix as well. Therefore, we have a multicollinearity issue with this model and need to drop one of them.')
# Multivariate model with Sepallength, Sepalwidth, Petallength



input_cols = ['PetalLengthCm','SepalWidthCm']

output_variable = ['SepalLengthCm']

X = iris_data[input_cols]

Y = iris_data[output_variable]

#Creating the Linear Regression Model

linreg = LinearRegression()

linreg.fit(X,Y)

print (linreg.intercept_)

print (linreg.coef_)

print('\n')

print('Sepal.length = 0.47*Petal.Length + 0.59*Sepal.Width + 2.25')

print('\n')

print('Holding all other independent variable fixed, a 1 centimeter increase in petallength lead to a increase in Sepalength by 0.47 centimeter')

print('Holding all other independent variable fixed, a 1 centimeter increase in sepalwidth lead to a increase in Sepalength by 0.59 centimeter')
# Multivariate model with Sepallength, Sepalwidth, Petalwidth



input_cols = ['PetalWidthCm','SepalWidthCm']

output_variable = ['SepalLengthCm']

X = iris_data[input_cols]

Y = iris_data[output_variable]

#Creating the Linear Regression Model

linreg = LinearRegression()

linreg.fit(X,Y)

print (linreg.intercept_)

print (linreg.coef_)

print('\n')

print('Sepal.length = 0.96*Petal.Width + 0.39*Sepal.Width + 3.46')

print('\n')

print('Holding all other independent variable fixed, a 1 centimeter increase in petalwidth lead to a increase in Sepalength by 0.96 centimeter')

print('Holding all other independent variable fixed, a 1 centimeter increase in sepalwidth lead to a increase in Sepalength by 0.39 centimeter')
#Check for multicollinearity thrrough Determinant value

import numpy as np

input_cols = ['PetalWidthCm','PetalLengthCm', 'SepalWidthCm']

output_variable = ['SepalLengthCm']

X = iris_data[input_cols]

Y = iris_data[output_variable]

corr = np.corrcoef(X, rowvar=0)

print(corr)

print('\n')

print (np.linalg.det(corr))



print('\n')



print('petallength and petalwidth are highly correlated=96% and lead to multicollinearity.')

print('\n')

print('Run the input variable in different combination to find the determinant value.The deteminant of correlation matrix is 0<=D<=1.')

print('D=0, then it indicates exact interdependence of expalanatory variable. D=1, then expalanatory variable independent to each other and have no multicollinearity issue.')

print('\n')

print('1.Determinant value for petalwidth, petallength, sepalwidth =0.057.')

print('2.Determinant value for petallength, sepalwidth =0.816.')

print('3.Determinant value for petalwidth, sepalwidth =0.865.')

print('4.Determinant value for petalwidth, petallength =0.072.')

print('\n')

print('We will avoid the model 1 & 4, as the value of D is close to 0, which indicates the multicollinearity issue. Whereas model 2 & 3 are acceptable, as the value of D is close to 1 and independent variables are not dependent to eachother.')

print('In general, when threshold level is D>0.7, then we can take all the input variables in the model.If D<0.4, then we can say that there is lot of interdependency between variables and we need to drop those variables which are highly correlated and causing multicollinearity.')
input_cols = ['PetalLengthCm', 'SepalWidthCm']

output_variable = ['SepalLengthCm']

X = iris_data[input_cols]

y = iris_data[output_variable]

y=iris_data['SepalLengthCm']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=12)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
# Buliding the Linear model with the algorithm

lin_reg=LinearRegression()

model=lin_reg.fit(X_train,y_train)
# Coefficient of determination or R squared value  # input_cols = ['Petal.Length', 'Sepal.Width']



print ('R-Squared for training dataset model:', model.score(X_train,y_train))



print('\n')



print('The high R-Squared value (0.834) from petallength & sepalwidth, implies that petallength & sepalwidth can be relied to explain 83.4% of the variations in sepallength.')
# input_cols = ['Petal.Length', 'Sepal.Width']

print(model.intercept_)

print (model.coef_)
## Predicting the x_test with the model

predicted=model.predict(X_test)

# Input variable petallength and Sepal width



print ('MAE:', metrics.mean_absolute_error(y_test, predicted))

print ('MSE:', metrics.mean_squared_error(y_test, predicted))

print ('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predicted)))
## R Squared value or coefficient of determination

print(metrics.r2_score(y_test,predicted))



#Compute null RMSE

# split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=12)



# create a NumPy array with the same shape as y_test

y_null = np.zeros_like(y_test, dtype=float)



# fill the array with the mean value of y_test

y_null.fill(y_test.mean())

y_null

print(y_test.shape)

print(y_null.shape)
# compute null RMSE

np.sqrt(metrics.mean_squared_error(y_test, y_null))
feature_cols=['PetalWidthCm','PetalLengthCm', 'SepalWidthCm']
# define a function that accepts a list of features and returns testing RMSE

def train_test_rmse(feature_cols):

    X = iris_data[feature_cols]

    y=iris_data['SepalLengthCm']

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=12)

    linreg = LinearRegression()

    linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)

    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# compare different sets of features

print (train_test_rmse(['PetalWidthCm','PetalLengthCm', 'SepalWidthCm']))

print (train_test_rmse(['PetalWidthCm', 'SepalWidthCm']))

print (train_test_rmse(['PetalLengthCm', 'SepalWidthCm']))
# define a function that accepts a list of features and returns testing MSE

def train_test_mse(feature_cols):

    X = iris_data[feature_cols]

    y=iris_data['SepalLengthCm']

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=12)

    linreg = LinearRegression()

    linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)

    return metrics.mean_squared_error(y_test, y_pred)
# compare different sets of features

print (train_test_mse(['PetalWidthCm','PetalLengthCm', 'SepalWidthCm']))

print (train_test_mse(['PetalWidthCm', 'SepalWidthCm']))

print (train_test_mse(['PetalLengthCm', 'SepalWidthCm']))
# define a function that accepts a list of features and returns testing MAE

def train_test_mae(feature_cols):

    X = iris_data[feature_cols]

    y=iris_data['SepalLengthCm']

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=12)

    linreg = LinearRegression()

    linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)

    return metrics.mean_absolute_error(y_test, y_pred)
# compare different sets of features

print (train_test_mae(['PetalWidthCm','PetalLengthCm', 'SepalWidthCm']))

print (train_test_mae(['PetalWidthCm', 'SepalWidthCm']))

print (train_test_mae(['PetalLengthCm', 'SepalWidthCm']))
iris_data.head(5)
# create dummy variables

Species_dummies = pd.get_dummies(iris_data.Species, prefix='Species')



# print 5 random rows from seed value 12

Species_dummies.sample(n=5, random_state=12)
Species_dummies.drop(Species_dummies .columns[0], axis=1, inplace=True)

iris_data = pd.concat([iris_data, Species_dummies], axis=1)

iris_data.head()

#iris_data.sample(n=5, random_state=12)
iris_data.corr()
feature_dummies1 =['PetalLengthCm', 'SepalWidthCm', 'Species_Iris-versicolor','Species_Iris-virginica']



# create X and y

X = iris_data[feature_dummies1]

y = iris_data['SepalLengthCm']



# instantiate and fit



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=12)

linreg = LinearRegression()

linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)







# print the coefficients

print (linreg.intercept_)

print (list(zip(feature_dummies1, linreg.coef_)))

#print (linreg.coef_)



print('\n')



# the predicted value of Sepallength



print('Predicted value of Sepal length:', y_pred)
feature_dummies2 =['PetalWidthCm','SepalWidthCm', 'Species_Iris-versicolor','Species_Iris-virginica']



# create X and y

X = iris_data[feature_dummies2]

y = iris_data['SepalLengthCm']



# instantiate and fit



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=12)

linreg = LinearRegression()

linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)







# print the coefficients

print (linreg.intercept_)

print (list(zip(feature_dummies2, linreg.coef_)))

#print (linreg.coef_)



print('\n')



# the predicted value of Sepallength



print('Predicted value:', y_pred)

feature_dummies3 =['PetalLengthCm', 'PetalWidthCm','SepalWidthCm', 'Species_Iris-versicolor','Species_Iris-virginica']



# create X and y

X = iris_data[feature_dummies3]

y = iris_data['SepalLengthCm']



# instantiate and fit



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=12)

linreg = LinearRegression()

linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)







# print the coefficients

print (linreg.intercept_)

print (list(zip(feature_dummies3, linreg.coef_)))

#print (linreg.coef_)



print('\n')



# the predicted value of Sepallength



print('Predicted value:', y_pred)



print('\n')



print('This model is not effective due to multicollinearity b/w the independent variables, petal length and width.')
print (train_test_rmse(['PetalLengthCm', 'Species_Iris-versicolor','Species_Iris-virginica','SepalWidthCm']))

print (train_test_rmse(iris_data.columns[iris_data.columns.str.startswith('Species_')]))

print (train_test_rmse(['PetalWidthCm', 'Species_Iris-versicolor','Species_Iris-virginica','SepalWidthCm']))



print('\n')



print('We can see among the calculated rmse value, the least rmse value is for the petal length, species_ and it is better than other rmse value. So, model#1 it is a best model to consider.')
print('END')