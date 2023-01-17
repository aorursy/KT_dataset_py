### This is notebook following with this kernel https://www.kaggle.com/sasha18/lasso-ridge-regularization. 
### But I dig deeper in the how to choose the best alpha (lambda penalties) for both RIDGE and LASSO REGRESSION and perform it with graph
#Import numerical libraries
import numpy as np
import pandas as pd

#Import graphically plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Import Linear Regression ML Libraries
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
data = pd.read_csv("../input/carmpg/car-mpg (1).csv")
data.head()
## Drop "Car_name" columns
data = data.drop(["car_name"], axis=1)
data.head()
## Replace Origin 1,2,3 into "America", "Europe", "Asia" and then get dummies
data["origin"] = data["origin"].replace({1:"america", 2:"europe", 3:"aisa"})
data.head()
data = pd.get_dummies(data,columns=["origin"])
data.head()
## Replace "?" with NAN value
## Replace NAN with median
data = data.replace("?", np.nan)
data = data.apply(lambda x: x.fillna(x.median()), axis=0)
X = data.drop(["mpg"], axis = 1) ## independent variables
y = data[["mpg"]] ## Dependent variable
## Scaling data

X_s = preprocessing.scale(X)
X_s = pd.DataFrame(X_s, columns= X.columns)
X_s.head()
y_s = preprocessing.scale(y)
y_s = pd.DataFrame(y_s, columns=y.columns)
y_s.head()
## Split into train, test set
X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.3, random_state = 1)
X_train.shape
regression_model = LinearRegression().fit(X_train,y_train)
for inx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][inx]))
intercept = regression_model.intercept_[0]
print("The intercept is {}".format(intercept))
### REGULARIZED RIDGE REGRESSION
#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of coeff
ridge_model = Ridge(alpha = 0.1).fit(X_train, y_train)
print("Ridge model coef {}".format(ridge_model.coef_))
## Ridge model with different alpha (lambda penalties)
ridge_model1 = Ridge(alpha=1).fit(X_train,y_train)
ridge_model5 = Ridge(alpha=5).fit(X_train,y_train)
ridge_model10 = Ridge(alpha = 10).fit(X_train,y_train)

plt.plot(ridge_model.coef_[0], "s", label="Ridge alpha 0.3")
plt.plot(ridge_model1.coef_[0], "^", label="Ridge alpha 1")
plt.plot(ridge_model5.coef_[0], "v", label="Ridge alpha 5")
plt.plot(ridge_model10.coef_[0], "o", label="Ridge alpha 10")
plt.plot(regression_model.coef_[0], "o", label="Linear")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0,0,len(regression_model.coef_[0]))
plt.ylim(-1, 1)
plt.legend()
print("LINEAR REGRESSION")
print(regression_model.score(X_train,y_train))
print(regression_model.score(X_test,y_test))
print("***********************")
print("RIDGE SCORE ALPHA 0.1")
print(ridge_model.score(X_train,y_train))
print(ridge_model.score(X_test,y_test))
print("***********************")
print("RIDGE SCORE ALPHA 1")
print(ridge_model1.score(X_train,y_train))
print(ridge_model1.score(X_test,y_test))
print("***********************")
print("RIDGE SCORE ALPHA 5")
print(ridge_model5.score(X_train,y_train))
print(ridge_model5.score(X_test,y_test))
print("***********************")
print("RIDGE SCORE ALPHA 10")
print(ridge_model10.score(X_train,y_train))
print(ridge_model10.score(X_test,y_test))
## Ridge regression performs better than linear regression, especailly when ALPHA is higher
## We can choose the model Ridge Regression with Alpha 10
### REGULARIZED LASSO REGRESSION
lasso_model = Lasso(alpha = 0.01).fit(X_train, y_train)
print("Lasso model coef {}".format(lasso_model.coef_))
### We can see there are some coeffcients that become zero.
lasso_model005 = Lasso(alpha = 0.005).fit(X_train,y_train)
lasso_model1 = Lasso(alpha = 1).fit(X_train,y_train)
lasso_model5 = Lasso(alpha = 5).fit(X_train,y_train)
lasso_model10 = Lasso(alpha = 10).fit(X_train,y_train)

plt.plot(lasso_model.coef_, "s", label="Lasso alpha 0.01")
plt.plot(lasso_model005.coef_, "^", label="Lasso alpha 0.005")
plt.plot(lasso_model1.coef_, "v", label="Lasso alpha 1")
plt.plot(lasso_model5.coef_, "o", label="Lasso alpha 5")
plt.plot(lasso_model10.coef_, "o", label="Lasso alpha 10")
plt.plot(ridge_model10.coef_[0], "o", label="Ridge alpha 10")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0,0,len(regression_model.coef_[0]))
plt.ylim(-1, 1)
plt.legend()
print("RIDGE SCORE ALPHA 10")
print(ridge_model10.score(X_train,y_train))
print(ridge_model10.score(X_test,y_test))
print("***********************")
print("LASSO SCORE ALPHA 0.01")
print(lasso_model.score(X_train,y_train))
print(lasso_model.score(X_test, y_test))
print("***********************")
print("LASSO SCORE ALPHA 0.005")
print(lasso_model005.score(X_train,y_train))
print(lasso_model005.score(X_test, y_test))
print("***********************")
print("LASSO SCORE ALPHA 1")
print(lasso_model1.score(X_train,y_train))
print(lasso_model1.score(X_test, y_test))
print("***********************")
print("LASSO SCORE ALPHA 5")
print(lasso_model5.score(X_train,y_train))
print(lasso_model5.score(X_test, y_test))
print("***********************")
print("LASSO SCORE ALPHA 10")
print(lasso_model10.score(X_train,y_train))
print(lasso_model10.score(X_test, y_test))

### Coefficient and intercept of Ridge Regression alpha 10
for inx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {:.2f}".format(col_name, ridge_model10.coef_[0][inx]))
print("The intercept is {:.2f}".format(ridge_model10.intercept_[0]))
fig = plt.figure(figsize=(10,8))
sns.residplot(x= X_test['wt'], y= y_test['mpg'], color='green', lowess=True )
fig = plt.figure(figsize=(10,8))
sns.residplot(x= X_test['yr'], y= y_test['mpg'], color='green', lowess=True )
y_pred = regression_model.predict(X_test)
plt.scatter(y_test['mpg'], y_pred)
