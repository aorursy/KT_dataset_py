import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import sklearn
print(sklearn.__version__)

print(np.__version__)

print(pd.__version__)
### importing the automobile data

df = pd.read_csv("../input/data-for-reg/auto-mpg.csv")
df.head()
df.sample(10)
df.shape
df.isnull().sum()
##replace the missing "?" with the np.nan value

df = df.replace('?',np.nan)

df.isna().sum()
df.isnull().sum()

df = df.dropna()
df.head()
df.isna().sum()
## origin and car name has no predictive powed on mpg

df.columns
df = df[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',

       'acceleration', 'model year']]
df.head()
## we make the model year with the age of the car

## prepending 19 so we got the full year

df['model year'] = '19'+df['model year'].astype(str)
df.head()
## now making  an age column based on the model year

current_year         = datetime.datetime.now().year

car_manufacture_year = pd.to_numeric(df['model year'])

df['age'] = current_year - car_manufacture_year
df.head()
df = df[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',

       'acceleration', 'age']]
df.head()
df.dtypes
## so there is one object field make it numerical

df['horsepower'] = pd.to_numeric(df['horsepower'])
df.dtypes
df.head()
## now we got all the neumeric value

## check the relationship between the age and mpg

plt.scatter(df['age'],df['mpg'])

plt.xlabel('age')

plt.ylabel('mpg')
## so we can see a downward trend that means more the year less 

## mgg and thats ovious during this time engine become advance
plt.scatter(df['horsepower'],df['mpg'])

plt.xlabel('howrsepower')

plt.ylabel('mpg')
plt.scatter(df['weight'],df['mpg'])

plt.xlabel('weight')

plt.ylabel('mpg')
plt.scatter(df['acceleration'],df['mpg'])

plt.xlabel('acceleration')

plt.ylabel('mpg')
df.corr()['mpg'].plot(kind='bar')
df.corr()['mpg'].plot()
df.corr()
sns.heatmap(df.corr(),annot=True)
df = df.sample(frac=1).reset_index(drop=True)
df.head()
### risk in multiple  regression

## building simple linear regression

data = df.copy()
from sklearn.model_selection import train_test_split

X = data[['horsepower']]

y = data[['mpg']]
x_train,x_test,y_train,y_test = train_test_split(X,y)
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression(normalize=True).fit(np.array(x_train),np.array(y_train))
print("Train Score : {}".format(linear_model.score(x_train,y_train)))

print("Test  Score : {}".format(linear_model.score(x_test,y_test)))
y_pred = linear_model.predict(x_test)
from sklearn.metrics import r2_score

print("Testing Score : {}".format(r2_score(y_test,y_pred)))
plt.scatter(x_test,y_test)

plt.plot(x_test,y_pred,color='r')

plt.xlabel("Horsepower")

plt.ylabel('mpg')

from sklearn.model_selection import train_test_split

X = data[['age']]

y = data[['mpg']]

x_train,x_test,y_train,y_test = train_test_split(X,y)

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression(normalize=True).fit(np.array(x_train),np.array(y_train))

print("Train Score : {}".format(linear_model.score(x_train,y_train)))

print("Test  Score : {}".format(linear_model.score(x_test,y_test)))

y_pred = linear_model.predict(x_test)

from sklearn.metrics import r2_score

print("Testing Score : {}".format(r2_score(y_test,y_pred)))

plt.scatter(x_test,y_test)

plt.plot(x_test,y_pred,color='r')

plt.xlabel("age")

plt.ylabel('mpg')

from sklearn.model_selection import train_test_split

X = data[['acceleration']]

y = data[['mpg']]

x_train,x_test,y_train,y_test = train_test_split(X,y)

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression(normalize=True).fit(np.array(x_train),np.array(y_train))

print("Train Score : {}".format(linear_model.score(x_train,y_train)))

print("Test  Score : {}".format(linear_model.score(x_test,y_test)))

y_pred = linear_model.predict(x_test)

from sklearn.metrics import r2_score

print("Testing Score : {}".format(r2_score(y_test,y_pred)))

plt.scatter(x_test,y_test)

plt.plot(x_test,y_pred,color='r')

plt.xlabel("age")

plt.ylabel('mpg')

from sklearn.model_selection import train_test_split

X = data[['displacement','horsepower','weight']]

y = data[['mpg']]

x_train,x_test,y_train,y_test = train_test_split(X,y)

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression(normalize=True).fit(np.array(x_train),np.array(y_train))

print("Train Score : {}".format(linear_model.score(x_train,y_train)))

print("Test  Score : {}".format(linear_model.score(x_test,y_test)))

y_pred = linear_model.predict(x_test)

from sklearn.metrics import r2_score

print("Testing Score : {}".format(r2_score(y_test,y_pred)))

# plt.scatter(x_test,y_test)

# plt.plot(x_test,y_pred,color='r')

# plt.xlabel("age")

# plt.ylabel('mpg')
plt.scatter(x_test['displacement'],y_test)

plt.scatter(x_test['horsepower'],y_test)

plt.scatter(x_test['weight'],y_test)

plt.plot(x_test['displacement'],y_pred,color='r')

plt.plot(x_test['horsepower'],y_pred,color='b')

plt.plot(x_test['weight'],y_pred,color='g')

## find the coefficient of the attributes we use

predictors = x_train.columns
coff = pd.Series(linear_model.coef_[0],predictors).sort_values()
coff
y_pred = linear_model.predict(x_test)
actual = []

for item in y_pred:

    actual.append(item[0])
y_test['predicted'] = np.array(actual)
r2_score(y_test['mpg'],y_test['predicted'])
df = pd.read_csv("../input/data-for-reg/exams.csv")
df.head()
df['gender'].unique()
d1= {'male':0,'female':1}

df['gender'] = df['gender'].map(d1)

df.head()
df['race/ethnicity'].unique()
d2 = {'group E':0, 'group C':1, 'group B':2, 'group D':3, 'group A':4}
df['race/ethnicity'] = df['race/ethnicity'].map(d2)
df.head()
df['parental level of education'].unique()
d3 = {"associate's degree":0, 'some college':1, 'high school':2,

       "bachelor's degree":3, 'some high school':4, "master's degree":5}
df['parental level of education'] = df['parental level of education'].map(d3)
df.head()
print(df['lunch'].unique())

print(df['test preparation course'].unique())
d4 = {'standard':0 ,'free/reduced':1}

d5 = {'none':0, 'completed':1}

df['lunch'] = df['lunch'].map(d4)

df['test preparation course'] = df['test preparation course'].map(d5)
df.dtypes
df.sample(10)
X = df.drop('math score',axis=1)

y = df[['math score']]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
linear_model = LinearRegression(fit_intercept=True).fit(x_train,y_train)
linear_model.score(x_test,y_test)
from sklearn.metrics import r2_score
r2_score(y_test,linear_model.predict(x_test))
plt.plot(range(len(y_test)),y_test)

plt.plot(range(len(y_test)),linear_model.predict(x_test))
### import all kinds of linear regression

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet,Lars,SGDRegressor
from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
def model_fc(model,X_train,y_train):

    model.fit(X,y)

    return model



def build_model(model,X_test,y_test):

    model = model

    print("MODEL ACCURACY : {}".format(model.score(X_test,y_test)));

    y_pred = model.predict(X_test)

    plt.plot(range(len(y_pred)),y_pred)

    plt.plot(range(len(y_test)),y_test)

    plt.show()

    
lr = LinearRegression()

las = Lasso()

ri = Ridge()

el = ElasticNet()

la = Lars()

sg = SGDRegressor()

sv = SVR()

kn = KNeighborsRegressor()

dc = DecisionTreeRegressor()

ra = RandomForestRegressor()
models  = { lr : 'LinearRegression',

las : 'Lasso',

ri : 'Ridge',

el : 'ElasticNet',

la : 'Lars',

sg : 'SGDRegressor',

sv : 'SVR',

kn : 'KNeighborsRegressor',

dc : 'DecisionTreeRegressor',

ra : 'RandomForestRegressor'} 
models
for model in models:

    model = model_fc(model,x_train,y_train)

    print(str(model))

    build_model(model,x_test,y_test)
## Hyparameter tuning

from sklearn.model_selection  import GridSearchCV

parameters = {'alpha':[.2,.4,.6,.7,.8,.9,1]}

grid_search = GridSearchCV(las,parameters,cv=5,return_train_score=True)
grid_search.fit(x_train,y_train)
grid_search.best_params_
d = grid_search.best_estimator_
d.fit(x_train,y_train)
d.score(x_test,y_test)