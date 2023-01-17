#your_local_path="C:/Users/tejks/Desktop/ML/practice/"
import pandas as pd   

import numpy as np

from sklearn.model_selection  import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore') 
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (8, 6)

plt.rcParams['font.size'] = 14
df=pd.read_csv("../input/Iris.csv")

df.head(10)
print("dimensions of data: ", df.shape,'\n' )
df.info()
df.describe()
rows, col = df.shape

print("Rows : %s, column : %s" % (rows, col))
sns.set()

df.hist(figsize=(12,10), color='red')

plt.show()
snsdata = df.drop(['Id'], axis=1)

g = sns.pairplot(snsdata, hue='Species', markers='x')

g = g.map_upper(plt.scatter)

g = g.map_lower(sns.kdeplot)
sns.violinplot(x='SepalLengthCm', y='Species', data=df, inner='stick', palette='autumn')

plt.show()

sns.violinplot(x='SepalWidthCm', y='Species', data=df, inner='stick', palette='autumn')

plt.show()

sns.violinplot(x='PetalLengthCm', y='Species', data=df, inner='stick', palette='autumn')

plt.show()

sns.violinplot(x='PetalWidthCm', y='Species', data=df, inner='stick', palette='autumn')

plt.show()
#Explore the relationship between Sepal Length and other independent variables

sns.lmplot(x='SepalWidthCm', y='SepalLengthCm', data=df, aspect=1.5, scatter_kws={'alpha':0.2})

df.plot(kind='scatter', x='SepalWidthCm', y='SepalLengthCm', alpha=0.2)
sns.boxplot(df[['SepalLengthCm']],data=df,linewidth=1)

plt.show()
#Checking the relationship between Sepal width and Sepal length

input_cols = ['SepalWidthCm']

output_variable = ['SepalLengthCm']

X = df[input_cols]

Y = df[output_variable]

#Creating the Linear Regression Model

linreg = LinearRegression()

linreg.fit(X,Y)

print (linreg.intercept_)

print (linreg.coef_)
#Checking the relationship between Petal width and Sepal length

input_cols = ['PetalWidthCm']

output_variable = ['SepalLengthCm']

X = df[input_cols]

Y = df[output_variable]

#Creating the Linear Regression Model

linreg = LinearRegression()

linreg.fit(X,Y)

print (linreg.intercept_)

print (linreg.coef_)
#Checking the relationship between Petal length and Sepal length

input_cols = ['PetalLengthCm']

output_variable = ['SepalLengthCm']

X = df[input_cols]

Y = df[output_variable]

#Creating the Linear Regression Model

linreg = LinearRegression()

linreg.fit(X,Y)

print (linreg.intercept_)

print (linreg.coef_)

#Checking the relationship between Petal width,Petal Length and Sepal length

input_cols = ['SepalWidthCm','PetalLengthCm','PetalWidthCm']

output_variable = ['SepalLengthCm']

X = df[input_cols]

Y = df[output_variable]

#Creating the Linear Regression Model

linreg = LinearRegression()

linreg.fit(X,Y)

print (linreg.intercept_)

print (linreg.coef_)

print(model.score(X,Y))
#Check for multicollinearity

import numpy as np

corr = np.corrcoef(X, rowvar=0)

corr
print (np.linalg.det(corr))
y=df['SepalLengthCm']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=False)
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
## Buliding the Linear model with the algorithm

lin_reg=LinearRegression()

model=lin_reg.fit(x_train,y_train)
## Coefficient of determination or R squared value

model.score(x_train,y_train)
print(model.intercept_)

print (model.coef_)
## Predicting the x_test with the model

predicted=model.predict(x_test)
## RMSE(Root Mean Squared Error)

print(np.sqrt(metrics.mean_squared_error(y_test,predicted)))
## R Squared value or coefficient of determination

print(metrics.r2_score(y_test,predicted))
## Mean Absolute Error

print(metrics.mean_absolute_error(y_test,predicted))
#Compute null RMSE

# split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)



# create a NumPy array with the same shape as y_test

y_null = np.zeros_like(y_test, dtype=float)



# fill the array with the mean value of y_test

y_null.fill(y_test.mean())

y_null
# compute null RMSE

np.sqrt(metrics.mean_squared_error(y_test, y_null))