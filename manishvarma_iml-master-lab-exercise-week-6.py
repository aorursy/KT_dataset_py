!pip install pygam
!pip install graphviz
!pip install pydotplus
import numpy as np 

import pandas as pd 

import statsmodels.api as sm

import statsmodels.formula.api as sam

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.preprocessing import PolynomialFeatures

from pygam import LogisticGAM,LinearGAM, s, f

from pygam.datasets import default, wage

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn import datasets

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image  

import pydotplus   

from patsy import dmatrix

import statsmodels.formula.api as smf

from sklearn import linear_model as lm

from matplotlib import pyplot



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/Wage.csv")

df.head()
dfInput = df[['age']]

dfOutput = df['wage']



train_x, test_x, train_y, test_y = train_test_split(dfInput, dfOutput, test_size=0.33, random_state = 1)



plt.scatter(train_x, train_y, facecolor='None', edgecolor='k', alpha=0.3)

plt.show()
#Linear Regression

model = LinearRegression()

model.fit(train_x,train_y)

print(model.coef_)

print(model.intercept_)
#Plot for Linear Regression

pred = model.predict(test_x)



plt.subplots(figsize=(12,5))

plt.scatter(test_x, test_y, facecolor='None', edgecolor='k', alpha=0.3)

plt.plot(test_x, pred,  color = 'red')

plt.show()
#RMSE with Linear Regression

print(sqrt(mean_squared_error(test_y, pred)))
#Polynomial Regression

df = pd.read_csv("../input/Wage.csv")

dfInput = df[['age']]

dfOutput = df['wage']



train_x, test_x, train_y, test_y = train_test_split(dfInput, dfOutput, test_size=0.33, random_state = 1)



poly = PolynomialFeatures(degree = 10) 

inputDF_poly = poly.fit_transform(train_x) 



poly.fit(inputDF_poly, train_y) 

lin2 = LinearRegression() 

lin2.fit(inputDF_poly, train_y)
#RMSE for polynomial regression

print(sqrt(mean_squared_error(test_y, lin2.predict(poly.fit_transform(test_x)))))
#Regression Splines

df = pd.read_csv("../input/Wage.csv")

dfInput = df['age']

dfOutput = df['wage']



train_x, test_x, train_y, test_y = train_test_split(dfInput, dfOutput, test_size=0.33, random_state = 1)



df_cut, bins = pd.cut(train_x, 4, retbins=True, right=True)

df_cut.value_counts()
#Bins value

bins
df_steps = pd.concat([train_x, df_cut, train_y], keys=['age','age_cuts','wage'], axis=1)

df_steps.head()
#Transforming categorical variables

df_steps_dummies = pd.get_dummies(df_cut)

df_steps_dummies.columns = ['17.938-33.5','33.5-49','49-64.5','64.5-80'] 

df_steps_dummies.head()
#Checking for normal distribution

pyplot.hist(df_steps.wage)

pyplot.show()
# Fitting GLM

fit3 = sm.GLM(df_steps.wage, df_steps_dummies).fit()



#Binning

bin_mapping = np.digitize(test_x, bins) 

print(bin_mapping)
#RMSE with Regression Splines

X_test = pd.get_dummies(bin_mapping).drop([5], axis=1)

pred2 = fit3.predict(X_test)

print(sqrt(mean_squared_error(test_y, pred2)))
#Plotting with Regression Splines

xp = np.linspace(test_x.min(),test_x.max()-1,70) 

bin_mapping = np.digitize(xp, bins) 

X_valid_2 = pd.get_dummies(bin_mapping) 

pred2 = fit3.predict(X_valid_2)



plt.subplots(figsize=(12,5))

plt.scatter(train_x, train_y,facecolor='None', edgecolor='k', alpha=0.3)   

plt.plot(xp, pred2, color = 'red')   

plt.show()
#Cubic spline with 3 knots 

transformed_x = dmatrix("bs(train, knots=(25,60), degree=3)", {"train": train_x},return_type='dataframe')



fit1 = sm.GLM(train_y, transformed_x).fit()



pred1 = fit1.predict(dmatrix("bs(valid, knots=(25,60),degree=3)", {"valid": test_x}, return_type='dataframe'))
print(sqrt(mean_squared_error(test_y, pred1)))
xp = np.linspace(test_x.min(),test_x.max(),70)

pred1 = fit1.predict(dmatrix("bs(xp, knots=(25,60), include_intercept=False)", {"xp": xp}, return_type='dataframe'))



plt.subplots(1,1, figsize=(12,5))

plt.scatter(dfInput, dfOutput, facecolor='None', edgecolor='k', alpha=0.1)

plt.plot(xp, pred1)

plt.show()
#Logisitic GAM

X, y = default(return_X_y=True)

dfInput = pd.DataFrame({'Student':X[:,0],'Balance':X[:,1],'Income':X[:,2]})

dfInput.head()
dfOutput = pd.DataFrame({'Default': y})

dfOutput.head()
gam = LogisticGAM(f(0) + s(1) + s(2)).gridsearch(X, y)
fig, axs = plt.subplots(1, 3)

titles = ['student', 'balance', 'income']



for i, ax in enumerate(axs):

    XX = gam.generate_X_grid(term=i)

    pdep, confi = gam.partial_dependence(term=i, width=.95)



    ax.plot(XX[:, i], pdep)

    ax.plot(XX[:, i], confi, c='r', ls='--')

    ax.set_title(titles[i]);
gam.accuracy(X, y)
gam.predict(X)
X, y = wage(return_X_y=True)

dfInput_Linear = pd.DataFrame({'Year':X[:,0],'Age':X[:,1],'Education':X[:,2]})

dfInput_Linear.head()
dfOutput_Linear = pd.DataFrame({'Wage': y})

dfOutput_Linear.head()
gam = LinearGAM(s(0) + s(1) + f(2))

gam.gridsearch(X, y)
plt.figure();

fig, axs = plt.subplots(1,3);



titles = ['year', 'age', 'education']

for i, ax in enumerate(axs):

    XX = gam.generate_X_grid(term=i)

    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))

    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')

    if i == 0:

        ax.set_ylim(-30,30)

    ax.set_title(titles[i]);
gam.predict(X)
iris_DC = datasets.load_iris()

irisDF_DC = pd.DataFrame({

    'sepal length':iris_DC.data[:,0],

    'sepal width':iris_DC.data[:,1],

    'petal length':iris_DC.data[:,2],

    'petal width':iris_DC.data[:,3],

    'species':iris_DC.target

})

irisDF_DC.head()
inputDF_DC = irisDF_DC[['sepal length', 'sepal width', 'petal length', 'petal width']]  

outputDF_DC = irisDF_DC['species']



X_train_DC, X_test_DC, y_train_DC, y_test_DC = train_test_split(inputDF_DC, outputDF_DC, test_size=0.30)
clfDC = DecisionTreeClassifier(random_state = 0)



clfDC = clfDC.fit(X_train_DC,y_train_DC)



y_pred_DC = clfDC.predict(X_test_DC)



print("Accuracy:",metrics.accuracy_score(y_test_DC, y_pred_DC))
feature_cols = ['sepal length', 'sepal width', 'petal length', 'petal width']

dot_data = StringIO()

export_graphviz(clfDC, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True,feature_names = feature_cols,class_names=['0','1','2'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('diabetes.png')

Image(graph.create_png())
iris_RF = datasets.load_iris()

irisDF_RF = pd.DataFrame({

    'sepal length':iris_RF.data[:,0],

    'sepal width':iris_RF.data[:,1],

    'petal length':iris_RF.data[:,2],

    'petal width':iris_RF.data[:,3],

    'species':iris_RF.target

})

inputDF_RF = irisDF_RF[['sepal length', 'sepal width', 'petal length', 'petal width']]  

outputDF_RF = irisDF_RF['species']



X_train_RF, X_test_RF, y_train_RF, y_test_RF = train_test_split(inputDF_RF, outputDF_RF, test_size=0.30)
clfRF = RandomForestClassifier(n_estimators=50,random_state = 0)



clfRF.fit(X_train_RF,y_train_RF)



y_pred_RF = clfRF.predict(X_test_RF)



print("Accuracy:",metrics.accuracy_score(y_test_RF, y_pred_RF))

print(sqrt(mean_squared_error(y_test_RF, y_pred_RF)))
feature_imp = pd.Series(clfRF.feature_importances_,index=iris_RF.feature_names).sort_values(ascending=False)

feature_imp
df_DCR = pd.read_csv("../input/Wage.csv")

df_DCR.head()
dfInput = df_DCR[["year","age","sex","education"]]

dfInput = pd.get_dummies(dfInput)

dfOutput = df_DCR["wage"]



X_train, X_test, y_train, y_test = train_test_split(dfInput, dfOutput, test_size=0.30)
dfInput.head()
regressor = DecisionTreeRegressor(random_state = 0, max_depth = 4)  

regressor.fit(X_train, y_train) 

y_pred = regressor.predict(X_test)



print(sqrt(mean_squared_error(y_test, y_pred)))
dot_data = StringIO()

export_graphviz(regressor, out_file =dot_data, 

               feature_names = dfInput.columns.values)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('wage.png')

Image(graph.create_png())
df_DCR = pd.read_csv("../input/Wage.csv")

dfInput = df_DCR[["year","age","sex","education"]]

dfInput = pd.get_dummies(dfInput)

dfOutput = df_DCR["wage"]



X_train, X_test, y_train, y_test = train_test_split(dfInput, dfOutput, test_size=0.30)
regr = RandomForestRegressor(random_state=0,

                            n_estimators=100, max_depth = 4)

regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)



print(sqrt(mean_squared_error(y_test, y_pred)))