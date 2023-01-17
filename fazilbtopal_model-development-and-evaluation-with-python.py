import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from IPython.display import display

from IPython.html import widgets

from ipywidgets import interact, interactive, fixed, interact_manual
df = pd.read_csv('../input/auto_clean.csv')

df.head()
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm
X = df[['highway-mpg']]

Y = df['price']
lm.fit(X,Y)
Yhat = lm.predict(X)

Yhat[0:5]   
lm.intercept_
lm.coef_
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])
lm.intercept_
lm.coef_
import seaborn as sns

%matplotlib inline 
width = 12

height = 10

plt.figure(figsize=(width, height))

sns.regplot(x="highway-mpg", y="price", data=df)

plt.ylim(0,)
plt.figure(figsize=(width, height))

sns.regplot(x="peak-rpm", y="price", data=df)

plt.ylim(0,)
#The variable "peak-rpm" has a stronger correlation with "price", 

# it is approximate -0.704692  compared to   "highway-mpg" which is approximate -0.101616.



df[["peak-rpm","highway-mpg","price"]].corr()
width = 12

height = 10

plt.figure(figsize=(width, height))

sns.residplot(df['highway-mpg'], df['price'])

plt.show()
Y_hat = lm.predict(Z)
plt.figure(figsize=(width, height))





ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")

sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)





plt.title('Actual vs Fitted Values for Price')

plt.xlabel('Price (in dollars)')

plt.ylabel('Proportion of Cars')



plt.show()

plt.close()
def PlotPolly(model, independent_variable, dependent_variabble, Name):

    x_new = np.linspace(15, 55, 100)

    y_new = model(x_new)



    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')

    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')

    ax = plt.gca()

    ax.set_facecolor((0.898, 0.898, 0.898))

    fig = plt.gcf()

    plt.xlabel(Name)

    plt.ylabel('Price of Cars')



    plt.show()

    plt.close()
x = df['highway-mpg']

y = df['price']
# Here we use a polynomial of the 3rd order (cubic) 

f = np.polyfit(x, y, 3)

p = np.poly1d(f)

print(p)
PlotPolly(p, x, y, 'highway-mpg')
np.polyfit(x, y, 3)
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=2)

pr
Z_pr = pr.fit_transform(Z)
Z.shape
Z_pr.shape
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
Input=[('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), 

       ('model', LinearRegression())]
pipe = Pipeline(Input)

pipe
pipe.fit(Z,y)
ypipe=pipe.predict(Z)

ypipe[0:4]
# highway_mpg_fit

lm.fit(X, Y)



# Find the R^2

print('The R-square is: ', lm.score(X, Y))
Yhat = lm.predict(X)

print('The output of the first four predicted value is: ', Yhat[0:4])
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df['price'], Yhat)

print('The mean square error of price and predicted value is: ', mse)
# fit the model 

lm.fit(Z, df['price'])



# Find the R^2

print('The R-square is: ', lm.score(Z, df['price']))
Y_predict_multifit = lm.predict(Z)
print('The mean square error of price and predicted value using multifit is: ', \

      mean_squared_error(df['price'], Y_predict_multifit))
from sklearn.metrics import r2_score
r_squared = r2_score(y, p(x))

print('The R-square value is: ', r_squared)
mean_squared_error(df['price'], p(x))
new_input = np.arange(1, 100, 1).reshape(-1, 1)
lm.fit(X, Y)

lm
yhat=lm.predict(new_input)

yhat[0:5]
plt.plot(new_input, yhat)

plt.show()
df._get_numeric_data().head()
# First lets only use numeric data

df = df._get_numeric_data()
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):

    width = 12

    height = 10

    plt.figure(figsize=(width, height))



    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)

    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)



    plt.title(Title)

    plt.xlabel('Price (in dollars)')

    plt.ylabel('Proportion of Cars')



    plt.show()

    plt.close()
def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):

    width = 12

    height = 10

    plt.figure(figsize=(width, height))



    #training data 

    #testing data 

    # lr:  linear regression object 

    #poly_transform:  polynomial transformation object 

    

    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)

    

    plt.plot(xtrain, y_train, 'ro', label='Training Data')

    plt.plot(xtest, y_test, 'go', label='Test Data')

    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), 

             label='Predicted Function')

    plt.ylim([-10000, 60000])

    plt.ylabel('Price')

    plt.legend()
y_data = df['price']
x_data=df.drop('price', axis=1)
from sklearn.model_selection import train_test_split





x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, 

                                                    random_state=1)





print("number of test samples :", x_test.shape[0])

print("number of training samples:", x_train.shape[0])

lre = LinearRegression()
lre.fit(x_train[['horsepower']], y_train)
lre.score(x_test[['horsepower']], y_test)
lre.score(x_train[['horsepower']], y_train)
from sklearn.model_selection import cross_val_score
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
Rcross
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())
-1 * cross_val_score(lre,x_data[['horsepower']], y_data, cv=4, scoring='neg_mean_squared_error')
from sklearn.model_selection import cross_val_predict
yhat = cross_val_predict(lre, x_data[['horsepower']], y_data,cv=4)

yhat[0:5]
lr = LinearRegression()

lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

yhat_train[0:5]
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

yhat_test[0:5]
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'

DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'

DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
pr = PolynomialFeatures(degree=5)

x_train_pr = pr.fit_transform(x_train[['horsepower']])

x_test_pr = pr.fit_transform(x_test[['horsepower']])

pr
poly = LinearRegression()

poly.fit(x_train_pr, y_train)
yhat = poly.predict(x_test_pr)

yhat[0:5]
print("Predicted values:", yhat[0:4])

print("True values:", y_test[0:4].values)
PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly,pr)
poly.score(x_train_pr, y_train)
poly.score(x_test_pr, y_test)
Rsqu_test = []



order = [1, 2, 3, 4]

for n in order:

    pr = PolynomialFeatures(degree=n)

    

    x_train_pr = pr.fit_transform(x_train[['horsepower']])

    x_test_pr = pr.fit_transform(x_test[['horsepower']])    

    

    lr.fit(x_train_pr, y_train)

    Rsqu_test.append(lr.score(x_test_pr, y_test))



plt.plot(order, Rsqu_test)

plt.xlabel('order')

plt.ylabel('R^2')

plt.title('R^2 Using Test Data')

plt.text(3, 0.75, 'Maximum R^2 ')    
def f(order, test_data):

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)

    pr = PolynomialFeatures(degree=order)

    x_train_pr = pr.fit_transform(x_train[['horsepower']])

    x_test_pr = pr.fit_transform(x_test[['horsepower']])

    poly = LinearRegression()

    poly.fit(x_train_pr,y_train)

    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train,y_test, poly, pr)
interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))
pr = PolynomialFeatures(degree=2)

x_train_pr = pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 

                                       'highway-mpg','normalized-losses','symboling']])

x_test_pr = pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 

                                     'highway-mpg','normalized-losses','symboling']])
from sklearn.linear_model import Ridge
RigeModel = Ridge(alpha=0.1)
RigeModel.fit(x_train_pr, y_train)
yhat = RigeModel.predict(x_test_pr)
print('predicted:', yhat[0:4])

print('test set :', y_test[0:4].values)
Rsqu_test = []

Rsqu_train = []

dummy1 = []

ALFA = 10 * np.array(range(0,1000))

for alfa in ALFA:

    RigeModel = Ridge(alpha=alfa) 

    RigeModel.fit(x_train_pr, y_train)

    Rsqu_test.append(RigeModel.score(x_test_pr, y_test))

    Rsqu_train.append(RigeModel.score(x_train_pr, y_train))
width = 12

height = 10

plt.figure(figsize=(width, height))



plt.plot(ALFA,Rsqu_test, label='validation data  ')

plt.plot(ALFA,Rsqu_train, 'r', label='training Data ')

plt.xlabel('alpha')

plt.ylabel('R^2')

plt.legend()
from sklearn.model_selection import GridSearchCV
parameters1 = [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]

parameters1
RR = Ridge()

RR
Grid1 = GridSearchCV(RR, parameters1, cv=4)
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
BestRR = Grid1.best_estimator_

BestRR
BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)
parameters2 = [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000],'normalize':[True,False]} ]

Grid2 = GridSearchCV(Ridge(), parameters2, cv=4)

Grid2.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_data)

Grid2.best_estimator_
Grid2.best_estimator_.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)