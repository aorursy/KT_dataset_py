import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
from sklearn import datasets
boston = datasets.load_boston()
boston.keys()
boston.feature_names
boston.data.shape
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['MEDV'] = boston.target #Adding the price column also to the original df
df.head()
df.info()
df.describe()
#sns.pairplot(df)
#Looking at the distribution of the target variable

sns.distplot(df['MEDV'],bins=30)
plt.figure(figsize=(12,10))

sns.heatmap(df.corr(),cmap='coolwarm',annot=True)

plt.xticks(rotation=0)

plt.show()
plt.figure(figsize=(12,10))

sns.clustermap(df.corr(),cmap='coolwarm',annot=True)

plt.xticks(rotation=0)

plt.show()
#Distribution of the dependent variable

plt.figure(figsize=(4,6))

sns.boxplot('MEDV',data=df,orient='v')
#Visualising simple regression output of LSTAT v/s MEDV (based on correlation inference)  

sns.lmplot(x='LSTAT',y='MEDV',data=df)
#Visualising simple regression output of RM v/s MEDV (based on correlation inference)  

sns.lmplot(x='RM',y='MEDV',data=df)
#Separating the feature and predictor df

X = df.iloc[:,:-1]

y = df.iloc[:,-1]
#Splitting X and y in train and test observations 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=100)
#Checking the shape of the new split datasets

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#Creating an instance object of linear regression

from sklearn.linear_model import LinearRegression

ln = LinearRegression()
#Fitting the linear model based on training data

ln.fit(X=X_train,y=y_train)
#Using the linear regression output to predict values of y on the x_test observations

y_pred = ln.predict(X=X_test)
intercept = ln.intercept_

coefficients = ln.coef_



print(intercept)

print(coefficients)
X_train.columns
coeff = pd.DataFrame(data=coefficients,index=('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',

       'PTRATIO', 'B', 'LSTAT'),columns=['Coefficients'])
coeff['Description'] = ['Per capita crime rate by town', 

                        'Proportion of residential land zoned for lots over 25,000 sq. ft', 

                        'Proportion of non-retail business acres per town',

                        'Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)',

                        'Nitric oxide concentration (parts per 10 million)',

                        'Average number of rooms per dwelling',

                        'Proportion of owner-occupied units built prior to 1940',

                        'Weighted distances to five Boston employment centers',

                        'Index of accessibility to radial highways',

                        'Full-value property tax rate per USD 10,000',

                        'Pupil-teacher ratio by town',

                        '1000(Bk — 0.63)², where Bk is the proportion of people of African American descent by town',

                        'Percentage of lower status of the population']
#Printing the new data frame which as coefficient value for each feature variable along with it's description

pd.options.display.max_colwidth = 100

coeff
#Adding the Pearson's correlation coefficients also for comparison

corr = df.corr()['MEDV'].values

coeff['Correlation'] = corr[:-1]

pd.options.display.max_colwidth = 40

coeff
sns.set_style('darkgrid')

plt.scatter(x=y_test,y=y_pred)

plt.xlabel('y_test')

plt.ylabel('y_pred')
#Plotting a comparative plot of predicted y values and actual test y values 

from numpy.polynomial.polynomial import polyfit

b, m = polyfit(y_test, y_pred,1)

plt.scatter(x=y_test,y=y_pred)

plt.plot(y_test, b + m * y_test,color='red')

plt.xlabel('y_test')

plt.ylabel('y_pred')

plt.show()
#Plotting the residuals 

sns.distplot(y_test - y_pred,label='Residual Plot')
from sklearn import metrics as mt
print('MAE',mt.mean_absolute_error(y_test,y_pred))

print('MSE',mt.mean_squared_error(y_test,y_pred))

print('RMSE',np.sqrt(mt.mean_squared_error(y_test,y_pred)))

print('R squared',mt.r2_score(y_test,y_pred))

RMSE = np.sqrt(mt.mean_squared_error(y_test,y_pred))
#Revisiting original mean and median values of the predictor to gauge the extent of error

print(df['MEDV'].mean()) 

pricemean = df['MEDV'].mean()

print(RMSE/pricemean * 100)