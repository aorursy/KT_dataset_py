import pandas as pd

import numpy as np



# Import clean data 

path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/module_5_auto.csv'

df = pd.read_csv(path)
df.to_csv('module_5_auto.csv')
df=df._get_numeric_data()

df.head()
%%capture

! pip install ipywidgets
from IPython.display import display

from IPython.html import widgets 

from IPython.display import display

from ipywidgets import interact, interactive, fixed, interact_manual
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
def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):

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

    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')

    plt.ylim([-10000, 60000])

    plt.ylabel('Price')

    plt.legend()
y_data = df['price']
x_data=df.drop('price',axis=1)
from sklearn.model_selection import train_test_split





x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)





print("number of test samples :", x_test.shape[0])

print("number of training samples:",x_train.shape[0])

from sklearn.linear_model import LinearRegression
lre=LinearRegression()
lre.fit(x_train[['horsepower']], y_train)
lre.score(x_test[['horsepower']], y_test)
lre.score(x_train[['horsepower']], y_train)
from sklearn.model_selection import cross_val_score
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
Rcross
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())
-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')
from sklearn.model_selection import cross_val_predict
yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)

yhat[0:5]
lr = LinearRegression()

lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

yhat_train[0:5]
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

yhat_test[0:5]
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'

DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'

DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
from sklearn.preprocessing import PolynomialFeatures
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
pr=PolynomialFeatures(degree=2)

x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
from sklearn.linear_model import Ridge
RigeModel=Ridge(alpha=0.1)
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
parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]

parameters1
RR=Ridge()

RR
Grid1 = GridSearchCV(RR, parameters1,cv=4)
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
BestRR=Grid1.best_estimator_

BestRR
BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)