import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
from sklearn import datasets

boston= datasets.load_boston()

boston.keys()
data= pd.DataFrame(boston.data, columns=boston.feature_names)
data['MEDV']=boston.target
data.head()
#data Wrangling

data.info()
data.dtypes
data.describe()
data.isna().any
data.isna().sum()
for columns in data:

    plt.figure()

    sns.distplot(data[columns], color='red')
for columns in data:

    plt.figure()

    sns.boxplot(y=data[columns], color='blue')
#Bivariate Analysis

data.corr()
#Correlation of MEDV with other features

data.corr()['MEDV'].sort_values(ascending=False)
#Heatmap

plt.figure(figsize=(10,8))

sns.heatmap(data.corr(), annot=True, linecolor='white', linewidths='0.1', square=True)

plt.xticks(rotation=90)
#PairPlot

sns.pairplot(data)
#Regplot

for columns in data:

    plt.figure()

    sns.regplot(x=columns, y='MEDV' , data=data)
#Splitting the dataset

X= data.iloc[:,:-1]

Y= data.iloc[:,-1]
#Data Preprocessing

#1 - No missing value is there

data.isna().sum()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.25, random_state=1)
#Linear Regression- taking all the variables

from sklearn.linear_model import LinearRegression

lr_reg= LinearRegression(normalize=True)



lr_reg.fit(X_train, Y_train)
#coffecient values

coff= pd.DataFrame(lr_reg.coef_, index=X_train.columns).sort_values(by=[0], ascending=False)

coff.rename(columns={0:'coff'})



print('linear intercept is {}'.format(lr_reg.intercept_))
Y_pred= lr_reg.predict(X_test)

plt.scatter(Y_test, Y_pred)

#less scattered it will be more better it is
#Evaluating metrics for linear regression

from sklearn import metrics

print('R2 score for linear reg is {}'.format(metrics.r2_score(Y_test, Y_pred)))

print('MSE for linear reg is {}'.format(metrics.mean_squared_error(Y_test, Y_pred)))

print('RMSE for linear reg is {}'.format(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))))
#Lets use backward elimination to build model

from scipy.special import factorial

import statsmodels.formula.api as sm



ones= pd.Series(np.ones(shape=(506,1), dtype=int).ravel())

X1= pd.concat([ones, X], axis=1, )



X1_opt=X1.iloc[:,[0,1,2,4,5,6,8,9,10,11,12,13]]
reg_OLS= sm.OLS(endog=Y, exog=X1_opt)

reg_OLS= reg_OLS.fit()

reg_OLS.summary()

#thus 'Age' and 'Indus' has very high p value. So they are statistically insignificant--thus removing them
#Decision tree

from sklearn.tree import DecisionTreeRegressor

dtree= DecisionTreeRegressor(criterion='mse', random_state=1)



dtree.fit(X_train, Y_train)
print('R2 score for decision tree is {}'.format(metrics.r2_score(Y_test, dtree.predict(X_test))))

print('RMSE for decision tree is {}'.format(np.sqrt(metrics.mean_squared_error(Y_test, dtree.predict(X_test)))))
pd.Series(dtree.feature_importances_, index=X_train.columns).sort_values(ascending=False)

#Looks like only two columns has high dependency - LSTAT, RM



sns.barplot(x=X_train.columns, y=dtree.feature_importances_)

plt.xticks(rotation=90)
#lets draw the tree to analyse better

from sklearn.externals.six import StringIO

from IPython.display import Image

from sklearn.tree import export_graphviz

from sklearn import tree

import pydot
#Create DOT data

dot_data= StringIO()

export_graphviz(dtree, out_file=dot_data, feature_names=X_train.columns.tolist(), rounded=True, filled=True)



#Draw Graph

graph= pydot.graph_from_dot_data(dot_data.getvalue())



#Show Graph

Image(graph[0].create_png())
#Using grid search CV for best parameter for decision tree

from sklearn.model_selection import GridSearchCV

dt= DecisionTreeRegressor(random_state=5)

params= [{'max_depth':[4,8,12,16,20], "max_leaf_nodes":range(2,20)}]

grid= GridSearchCV(dt, param_grid=params, refit=True)
grid.fit(X_train, Y_train)

grid_predictions= grid.predict(X_test)



print('Accuracy Score:{}'.format(metrics.r2_score(Y_test, grid_predictions)))
print('Best hyperparameter is {}'.format(grid.best_params_))

print('\n')

print('Best Estimator is :')

grid.best_estimator_
#Using hyperparameter- max-depth=4, max_leaf_nodes=7

dtree1= DecisionTreeRegressor(criterion='mse', max_depth= 4, max_leaf_nodes= 7, random_state=1)

dtree1.fit(X_train, Y_train)

metrics.r2_score(Y_test, dtree1.predict(X_test))

np.sqrt(metrics.mean_squared_error(Y_test, dtree1.predict(X_test)))
#Create DOT data

dot_data= StringIO()

export_graphviz(grid.best_estimator_, out_file=dot_data, feature_names=X_train.columns.tolist(),class_names=['MEDV'], rounded=True, filled=True)



#Draw Graph

graph= pydot.graph_from_dot_data(dot_data.getvalue())



#Show Graph

Image(graph[0].create_png())
#Important features in Decision tree

pd.DataFrame(grid.best_estimator_.feature_importances_, index=X_train.columns, columns=['Importance'])
#using only LSTAT and RM to predict the MEDV value

X_train_dt= X_train[['LSTAT', 'RM']]

X_test_dt= X_test[['LSTAT', 'RM']]



dtr= DecisionTreeRegressor(criterion='mse', max_depth=4, max_leaf_nodes=7, random_state=1)

dtr.fit(X_train_dt, Y_train)





print('The R2 score is {}'.format(metrics.r2_score(Y_test, dtr.predict(X_test_dt))))

print('The RMSE is {}'.format(np.sqrt(metrics.mean_squared_error(Y_test, dtr.predict(X_test_dt)))))
#Random Forest

#Taking max_depth = 4 using Decision tree



from sklearn.ensemble import RandomForestRegressor

rf= RandomForestRegressor(random_state=1,max_depth=4)

params1= [{'n_estimators': range(10,100)}]

grid1= GridSearchCV(rf, param_grid=params1, refit=True)
grid1.fit(X_train, Y_train)
grid_predictions1= grid1.predict(X_test)



print('Accuracy Score: {}'.format(metrics.r2_score(Y_test, grid_predictions1)))
print('The best value for n_estimator is {}'.format(grid1.best_params_))

pd.DataFrame(grid1.best_estimator_.feature_importances_, index=X_train.columns, columns=['Importance'])
print('One of the estimator is {}'.format(grid1.best_estimator_.estimators_[0]))
#Create DOT data

dot_data= StringIO()

export_graphviz(grid1.best_estimator_.estimators_[1], out_file=dot_data, feature_names=X_train.columns.tolist(),class_names=['MEDV'], rounded=True, filled=True)



#Draw Graph

graph= pydot.graph_from_dot_data(dot_data.getvalue())



#Show Graph

Image(graph[0].create_png())
#Regularization

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso
lr= LinearRegression()

lr.fit(X_train, Y_train)
print(lr.coef_)

print('\n')

print('The R2 score for train data is {}'.format(lr.score(X_train, Y_train)))

print('The R2 for test data is {}'.format(lr.score(X_test, Y_test)))
#Ridge Regression

rr1= Ridge(alpha=0.01)

rr1.fit(X_train, Y_train)

print(rr1.coef_)

print('\n')

print('The R2 score for train data is {}'.format(rr1.score(X_train, Y_train)))

print('The R2 score for train data is {}'.format(rr1.score(X_test, Y_test)))



sns.barplot(X_train.columns, rr1.coef_)

plt.xticks(rotation=90)



#looks like there is no change in coff wrt Linear Reg as alpha value is very small
rr100= Ridge(alpha=100)

rr100.fit(X_train, Y_train)

print(rr100.coef_)

print('\n')

print('The R2 score for train data is {}'.format(rr100.score(X_train, Y_train)))

print('The R2 score for train data is {}'.format(rr100.score(X_test, Y_test)))



sns.barplot(X_train.columns, rr100.coef_)

plt.xticks(rotation=90)



#looks like there is no change in coff wrt Linear Reg as alpha value is very small
#Plotting the coff values for LR, Ridge(0.01), Ridge(100)

plt.plot(lr.coef_ ,linestyle='none' , marker='+', color='blue', markersize=10, label='lin reg')

plt.plot(rr1.coef_ ,linestyle='none' , marker='o', color='green', label=' Ridge(0.01)')

plt.plot(rr100.coef_ ,linestyle='none' , marker='*', color='red', label= 'Ridge(100)')

plt.xlabel(boston.feature_names)

plt.legend()

plt.show()
#Zipping the coff values

list(zip(X_train.columns, rr100.coef_))
#Lasso regression

lasso1= Lasso(alpha=0.01)

lasso1.fit(X_train, Y_train)



lasso1.coef_

print('The R2 score for train data is {}'.format(lasso1.score(X_train, Y_train)))

print('The R2 score for test data is {}'.format(lasso1.score(X_test, Y_test)))



sns.barplot(X_train.columns, lasso1.coef_)

plt.xticks(rotation=90)
lasso2= Lasso(alpha=1)

lasso2.fit(X_train, Y_train)



lasso2.coef_

print('The R2 score for train data is {}'.format(lasso2.score(X_train, Y_train)))

print('The R2 score for test data is {}'.format(lasso2.score(X_test, Y_test)))



sns.barplot(X_train.columns, lasso2.coef_)

plt.xticks(rotation=90)
list(zip(X_train.columns, lasso2.coef_))
plt.plot(lr.coef_, linestyle='none', marker='*', color='blue', label='lin reg')

plt.plot(lasso1.coef_, linestyle='none', marker='o',color='red', label='lasso(0.01)')

plt.plot(lasso2.coef_, linestyle='none', marker='+', color='green', label='lasso(1)')

plt.xlabel(X_train.columns)

plt.legend()

plt.show()
print('no of features where coff is not zero: {}'.format(sum((lasso2.coef_!=0))))
train_score=[]

test_score= []



for i in [0.01,0.1,1,2,3,4,5,6,7,8,9,10]:

    lasso= Lasso(alpha=i)

    lasso.fit(X_train, Y_train)

    

    train_sc= lasso.score(X_train, Y_train)

    test_sc= lasso.score(X_test, Y_test)

    

    train_score.append(train_sc)

    test_score.append(test_sc)

    

print(train_score)

print(test_score)
plt.figure()

plt.plot([0.01,0.1,1,2,3,4,5,6,7,8,9,10], train_score, marker='*', color='green')

plt.plot([0.01,0.1,1,2,3,4,5,6,7,8,9,10], test_score, marker='+', color='red')

plt.xlabel('x-values')

plt.ylabel('accuracy score')

plt.plot()



#at x=1 the accuracy value is almost sme for train and test data
#Elastic net Regression

from sklearn.linear_model import ElasticNet

Elastic1= ElasticNet(alpha=1)



Elastic1.fit(X_train, Y_train)

train_score= Elastic1.score(X_train, Y_train)

test_score= Elastic1.score(X_test, Y_test)



print('Train score={}'.format(train_score))

print('Test score={}'.format(test_score))

print('No of features used={}'.format(np.sum(Elastic1.coef_!=0)))
train_score=[]

test_score= []



for i in [0.01,0.1,1,2,3,4,5,6,7,8,9,10]:

    elastic= ElasticNet(alpha=i)

    elastic.fit(X_train, Y_train)

    

    train_sc= elastic.score(X_train, Y_train)

    test_sc= elastic.score(X_test, Y_test)

    

    train_score.append(train_sc)

    test_score.append(test_sc)

    

print(train_score)

print(test_score)
plt.figure()

plt.plot([0.01,0.1,1,2,3,4,5,6,7,8,9,10], train_score, marker='*', color='green')

plt.plot([0.01,0.1,1,2,3,4,5,6,7,8,9,10], test_score, marker='+', color='red')

plt.xlabel('x-values')

plt.ylabel('accuracy score')

plt.plot()



#at x=2 the accuracy value is almost sme for train and test data
#Using Grid search CV for Elatic net regression

from sklearn.model_selection import GridSearchCV

elasticnet= ElasticNet()

param_elastic={'alpha':[0.01,0.1,1,2,3,4,5,6,7,8,9,10,100]}

grid_elastic= GridSearchCV(estimator=elasticnet, param_grid=param_elastic, cv=3, refit= True)
grid_elastic.fit(X_train, Y_train)
print(grid_elastic.best_params_)

metrics.r2_score(Y_test, grid_elastic.best_estimator_.predict(X_test))