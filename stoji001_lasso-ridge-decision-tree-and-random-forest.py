import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn import preprocessing as pro 

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder 

import statsmodels.api as sm

from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error

from sklearn import linear_model

from sklearn.tree import DecisionTreeRegressor 

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from scipy.stats import skew

from numpy import exp 

from scipy.stats import linregress 

from sklearn import linear_model

from sklearn.linear_model import Ridge

import scipy as sp

from scipy.stats import ttest_rel

from sklearn.model_selection import cross_val_score

from scipy.stats import pearsonr 

from sklearn.metrics import make_scorer

from sklearn.ensemble import RandomForestRegressor 



import warnings

warnings.filterwarnings('ignore')



#colour palette for graphics

current_palette = sns.color_palette("GnBu_r")

sns.set_palette(current_palette)



# load data in pandas data frame

data = pd.read_csv('../input/melbourne-housing-market/MELBOURNE_HOUSE_PRICES_LESS.csv')
# removal of outliers 

# set 35 Bevis St to correct sale price ##

data.at[61142, 'Price'] = 904500



# set 5 Cottswold Place bedroom value to 4 ##

data.at[55467, 'Rooms'] = 4



# set 507 Orrong Rd Arrandale bedroom value to 6 ##

data.at[7825, 'Rooms'] = 6 



# set 1 Beddoe Ave bedroom value to 8 ##

data.at[49016, 'Rooms'] = 8 



# deletion of 20 Harrison St Mitcham listing ##

data.drop(59741, inplace=True)



# deletion of 213 station Road Melton ##

data.drop(21337, inplace=True)



# deletion of 84 Flemington Rd Parkville ##

data.drop(29421, inplace=True)



# deletion of 225 McKean St, Fitzroy North ##

data.drop(7452, inplace=True)



# deletion of 445 Warrigal Rd Burwood  ##

data.drop(55673, inplace=True)



# deletion of 10 Berkley St Hawthorn ##

data.drop(39847, inplace=True)



# deletion of 5 Ball Ct, Bundoora

data.drop(27354, inplace=True)   
# add new attribute columns for log of price

data['LogPrice'] = data['Price'].apply(np.log)



# function to convert the date from an object to a date time

data['Date'] = pd.to_datetime(data['Date'])



# I've also conducted some feature engineering by adding month as an attribute 

# add month column to data frame

data['Month'] = data['Date'].dt.month



# add year column to data frame 

data.drop(['Date'], axis=1, inplace=True)



# convert postcode from int to string

data['Postcode'] = data['Postcode'].apply(str)



# log transform distance 

data['DistanceLog'] = np.log1p(data['Distance'])



# log transform property count

data['Propertycountlog'] = np.log1p(data['Propertycount'])



# drop address

data.drop(['Address'], axis=1, inplace=True)



# dropping missing price values 

data.dropna(0, inplace=True)



# data frame for MLR model feature selection

MLR_feature_selection = data 
# dataframe for scatter plot matrix

# columns to be encoded

values = ('Rooms', 'Propertycount', 'Distance')



numdata = pd.DataFrame()



for i in values:

    numdata[i] = data[i]



numdata['Price'] = np.exp(data['LogPrice'])
# one-hot encoding of categorial variables

values = ('Type', 'Suburb', 'Method', 'SellerG', 'Postcode', 'Regionname', 'CouncilArea', 'Month')



for i in values:

  data = pd.concat([data, pd.get_dummies(data[i])], axis=1);



values_drop = ('Type', 'CouncilArea', 'Suburb', 'Method', 'SellerG', 'Postcode', 'Month', 'Price')



for i in values_drop:

    data.drop([i], axis=1, inplace=True)
# To determine the skewneess of the numeric variables I used a histogram and the scipy skewness function

# Given Distance returned a value greater than .8 I performed a log transform



# one hot encoded variables aren't amenable to this type of analysis so I was only able to investigate the

# number of rooms, distance from the CBD, and property count. 



# Rooms

plt.figure(figsize = (24, 12))

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams["axes.labelsize"] = 35

plt.title('Melbourne House Rooms', fontsize=40)

plt.ylabel('Density')

price = sns.distplot(data['Rooms'], bins=10, label='Rooms', axlabel='Number of Rooms', hist_kws=dict(edgecolor="k", linewidth=2))

plt.show()



room_skew = skew(data['Rooms'], axis=0, bias=True)

print("The skewness of 'Rooms' is %s" % room_skew)



# property count

plt.figure(figsize = (24, 12))

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams["axes.labelsize"] = 35

plt.title('Melbourne House Suburb Property Count', fontsize=40)

plt.ylabel('Density')

price = sns.distplot(data['Propertycount'], bins=10, label='Property Count', axlabel='Property Count', hist_kws=dict(edgecolor="k", linewidth=2))

plt.show()



plt.figure(figsize = (24, 12))

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams["axes.labelsize"] = 35

plt.title('Melbourne House Suburb log(Property Count)', fontsize=40)

plt.ylabel('Density')

price = sns.distplot(data['Propertycountlog'], bins=10, label='Log(Property Count)', axlabel='log(Property Count)', hist_kws=dict(edgecolor="k", linewidth=2))

plt.show()



propertycount_skew = skew(data['Propertycount'], axis=0, bias=True)

print("The skewness of 'Property Count' is %s" % propertycount_skew)

propertycountlog_skew = skew(data['Propertycountlog'], axis=0, bias=True)

print("The skewness of 'log(Property Count)' is %s" % propertycountlog_skew)



# Distance

plt.figure(figsize = (24, 12))

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams["axes.labelsize"] = 35

plt.title('Melbourne House Distance', fontsize=40)

plt.ylabel('Density')

price = sns.distplot(data['Distance'], bins=65, label='Distance from CBD', axlabel='Distance from CBD', hist_kws=dict(edgecolor="k", linewidth=2))

plt.show()



# Distancelog

plt.figure(figsize = (24, 12))

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams["axes.labelsize"] = 35

plt.title('Melbourne House Log(Distance)', fontsize=40)

plt.ylabel('Density')

price = sns.distplot(data['DistanceLog'], bins=5, label='Log(Distance) from CBD', axlabel='Log(Distance) from CBD', hist_kws=dict(edgecolor="k", linewidth=2))

plt.show()



distance_skew = skew(data['Distance'], axis=0, bias=True)

distancelog_skew = skew(data['DistanceLog'], axis=0, bias=True)



print("The skewness of 'Distance' is %s" % distance_skew)

print("The skewness of 'DistanceLog' is %s" % distancelog_skew)
# results are slightly better. 

plt.figure(figsize = (24, 20))

sp.stats.probplot(data['Distance'], plot=plt, fit=True)

plt.title('Probability Plot', fontsize=40)

plt.show()



plt.figure(figsize = (24, 20))

sp.stats.probplot(data['DistanceLog'],plot=plt, fit=True)

plt.title('Probability Plot', fontsize=40)

plt.show()



data.drop(['Distance'], axis=1, inplace=True)
# again the improvement is only marginal additionally this resulted in no performance improvement of the model

# hence I chose to leave this varibable untransformed. 



plt.figure(figsize=(24,20))

sp.stats.probplot(data['Propertycount'],plot=plt, fit=True)

plt.title('Probability Plot', fontsize=40)

plt.show()



plt.figure(figsize=(24,20))

sp.stats.probplot(data['Propertycountlog'],plot=plt, fit=True)

plt.title('Probability Plot', fontsize=40)

plt.show()



data.drop(['Propertycountlog'], axis=1, inplace=True)
# A scatter plot matrix of the dependent variable vs the independent variables does

# not reveal any non-linear realtionships. 

# also see the discusison of the residual plots of the regression models. 
# Use seaborn pairplot to produce a scatterplot matrix of numeric variables

plt.rcParams["axes.labelsize"] = 12

sns.pairplot(numdata)

plt.show()
# see residual plots for each regressor given below. 
# breaking data into test and train sets

SLR_X_train, SLR_X_test, SLR_Y_train, SLR_Y_test = train_test_split(data['Rooms'], data['LogPrice'],

                                                              test_size=0.33, random_state=5)
slope, intercept, rvalue_train, pvalue_train, stderr = linregress(SLR_X_train, SLR_Y_train)



predictions_train = []



for i in SLR_X_train:

    predictions_train.append(slope*i + intercept)

    

slope, intercept, rvalue_test, pvalue_test, stderr = linregress(SLR_X_test, SLR_Y_test)



predictions_test = []   



for i in SLR_X_test:

    predictions_test.append(slope*i + intercept)



print('\n' + str(np.sqrt(mean_squared_error(SLR_Y_train, predictions_train))))

print('\n' + str(np.sqrt(mean_squared_error(SLR_Y_test, predictions_test))))
plt.figure(figsize = (24, 20))

plt.rcParams["axes.labelsize"] = 35

plt.rcParams["font.family"] = "Times New Roman"

plt.title('Residual Plot for Simple Linear Regression', fontsize=45)

plt.ylabel('Residual')

plt.xlabel('Log(Price)')

sns.residplot(SLR_Y_train, predictions_train)

sns.residplot(SLR_Y_test, predictions_test, color='r')

plt.show()
SLR_prob_train = predictions_train - SLR_Y_train

plt.rcParams["axes.labelsize"] = 20

plt.figure(figsize=(12,10))

sp.stats.probplot(SLR_prob_train, plot=plt, fit=True)

plt.title('Probability Plot for Simple Linear Regression', fontsize=30)

plt.show()



SLR_prob_test = predictions_test - SLR_Y_test

plt.figure(figsize=(12,10))

sp.stats.probplot(SLR_prob_test, plot=plt, fit=True)

plt.title('Probability Plot for Simple Linear Regression', fontsize=30)

plt.show()
# label encorder variable 

label = pro.LabelEncoder() 



# columns to be encoded

values = ('Type', 'Method', 'SellerG', 'CouncilArea', 'Suburb', 'Regionname')



# label encoding for-loop 

for i in values:

    label.fit(list(MLR_feature_selection[i].values)) 

    MLR_feature_selection[i] = label.transform(list(MLR_feature_selection[i].values))



for i in values:

    print('The correlation and statitstical signifgance between %s and price is %s \n' %

          (i, pearsonr(MLR_feature_selection['LogPrice'], MLR_feature_selection[i])))
# Creating the attirbute and target data frames 

values = ['Rooms', 'DistanceLog', 'h', 't', 'u']



for i in data['Regionname'].unique():

   values.insert(-1,i)



attributes = pd.DataFrame()



for i in values:

    attributes[i] = data[i]
# Creating the train and test splits 

MLR_X_train, MLR_X_test, MLR_Y_train, MLR_Y_test = train_test_split(attributes, data['LogPrice'],

                                                            test_size=0.33, random_state=5)
# drop region column as this was only kept to generate the labels to create the attributes data frame 

data.drop(['Regionname'], axis=1, inplace=True)
lm = LinearRegression()

_ = lm.fit(MLR_X_train, MLR_Y_train)

print('Coefficients:\n {}'.format(str(lm.coef_)))

print('\nIntercept:', lm.intercept_)

print('\nRMSE: {}'.format(np.sqrt(mean_squared_error(MLR_Y_train, lm.predict(MLR_X_train)))))

print('\nR^2 train: {}'.format(lm.score(MLR_X_train, MLR_Y_train)))
print('Coefficients:\n {}'.format(str(lm.coef_)))

print('\nIntercept:', lm.intercept_)

print('\n' + 'RMSE: {}'.format(np.sqrt(mean_squared_error(MLR_Y_test, lm.predict(MLR_X_test)))))

print('\nR^2 train: {}'.format(lm.score(MLR_X_test, MLR_Y_test)))
model2 = sm.OLS(MLR_Y_train, MLR_X_train)

results = model2.fit()

print(results.summary())
model3 = sm.OLS(MLR_Y_test, MLR_X_test)

results2 = model3.fit()

print(results2.summary())
# K-fold cross validation 

CVSlm = cross_val_score(lm, attributes, data['LogPrice'], cv=10)

print('Cross validation scores R^2: {}'.format(CVSlm))

print('Mean cross validated R^2: {:>0.04f} \n'.format(CVSlm.mean()))



def RMSE(x, y):

    return np.sqrt(mean_squared_error(x,y))

            

RMSE_scorer = make_scorer(RMSE)



CVSlmRMSE = cross_val_score(lm, attributes, data['LogPrice'], scoring=RMSE_scorer, cv=10)

print('Cross validation scores RMSE: {}'.format(CVSlmRMSE))

print('Mean cross validated RMSE: {:>0.04f}'.format(CVSlmRMSE.mean()))
plt.figure(figsize = (24, 20))

plt.rcParams["axes.labelsize"] = 35

plt.rcParams["font.family"] = "Times New Roman"

plt.title('Residual Plot for Multiple Linear Regression', fontsize=45)

plt.ylabel('Residual')

plt.xlabel('Log(Price)')

residuals = sns.residplot(MLR_Y_train, lm.predict(MLR_X_train))

residuals_test = sns.residplot(MLR_Y_test, lm.predict(MLR_X_test), color='r')

residuals_test.tick_params(labelsize=25)

plt.show()
MLR_prob_train = lm.predict(MLR_X_train) - MLR_Y_train

plt.rcParams["axes.labelsize"] = 20

plt.figure(figsize=(12,10))

sp.stats.probplot(MLR_prob_train, plot=plt, fit=True)

plt.title('Probability Plot for Multiple Linear Regression', fontsize=30)

plt.show()



MLR_prob_test = lm.predict(MLR_X_test) - MLR_Y_test

plt.figure(figsize=(12,10))

sp.stats.probplot(MLR_prob_test, plot=plt, fit=True)

plt.title('Probability Plot for Multiple Linear Regression', fontsize=30)

plt.show()
attributes = data.drop(['LogPrice'], axis=1)



X_train, X_test, Y_train, Y_test = train_test_split(attributes, data['LogPrice'],

                                                              test_size=0.33, random_state=5)
lasso_para = {'alpha':[1, .1, .02, .01, .001]}



print(GridSearchCV(linear_model.Lasso(), 

                               param_grid=lasso_para).fit(X_train, Y_train).best_estimator_,)
# specficy estimator using tuned hyperparameters 

lasso = Lasso(alpha = .001, max_iter = 1000, random_state=5)



lasso.fit(X_train, Y_train)



predictions = pd.DataFrame(data=lasso.predict(X_train).flatten())



predtransformed = predictions.apply(np.exp)
# train set 

print('Coefficients:\n {}'.format(str(lasso.coef_)))

print('\nIntercept: {:>0.04f}'.format(lasso.intercept_))

print('\n' + 'RMSE: {:>0.04f}'.format(np.sqrt(mean_squared_error(Y_train, lasso.predict(X_train)))))

print('\nR^2: {:>0.04f}'.format(lasso.score(X_train, Y_train)))
# test set



predictions_test = lasso.predict(X_test)



print('Coefficients:\n {}'.format(str(lasso.coef_)))

print('\nIntercept: {:>0.04f}'.format(lasso.intercept_))

print('\n' + 'RMSE: {:>0.04f}'.format(np.sqrt(mean_squared_error(Y_test, lasso.predict(X_test)))))

print('\nR^2: {:>0.04f}'.format(lasso.score(X_test, Y_test)))
# RMSE in dollars 

print('Train RMSE ($): {:>0.2f}'.format(np.sqrt

    (mean_squared_error(np.exp(Y_train), np.exp(lasso.predict(X_train))))))

print('Test RMSE ($): {:>0.2f}'.format(np.sqrt

    (mean_squared_error(np.exp(Y_test), np.exp(lasso.predict(X_test))))))
# K-fold cross validation 

CVSlasso = cross_val_score(lasso, attributes, data['LogPrice'], cv=10)

print('Cross validated R^2: {}\n'.format(CVSlasso))

print('Mean cross validated R^2: {:>0.4f}'.format(CVSlasso.mean()))



CVSlassoRMSE = cross_val_score(lasso, attributes, data['LogPrice'], scoring=RMSE_scorer, cv=10)

print('Cross validated RMSE: {}'.format(CVSlassoRMSE))

print('Mean cross validated RMSE: {:>0.04f}'.format(CVSlassoRMSE.mean()))
plt.figure(figsize = (24, 20))

plt.rcParams["axes.labelsize"] = 35

plt.rcParams["font.family"] = "Times New Roman"

plt.title('Residual Plot for Lasso Regression', fontsize=45)

plt.ylabel('Residual')

plt.xlabel('Log(Price)')

residuals = sns.residplot(Y_train, lasso.predict(X_train))

residuals_test = sns.residplot(Y_test, lasso.predict(X_test), color='r')

residuals_test.tick_params(labelsize=25)

plt.show()

lasso.fit(X_train, Y_train)

plt.rcParams["axes.labelsize"] = 20

lasso_prob_train = lasso.predict(X_train) - Y_train

plt.figure(figsize=(12,10))

sp.stats.probplot(lasso_prob_train, plot=plt, fit=True)

plt.title('Probability Plot for Lasso Regression', fontsize=30)

plt.show()



lasso_prob_test = lasso.predict(X_test) - Y_test

plt.figure(figsize=(12,10))

sp.stats.probplot(lasso_prob_test, plot=plt, fit=True)

plt.title('Probability Plot for Lasso Regression', fontsize=30)

plt.show()
ridge_para = {'alpha': [30, 10, 5, 1, .1, .01, .001]}



print(GridSearchCV(linear_model.Ridge(), param_grid=ridge_para).fit(X_train, Y_train).best_estimator_,)
ridge = Ridge(alpha=5)

ridge.fit(X_train, Y_train)



predictions = ridge.predict(X_train)



print('Coefficients:\n {}'.format(str(ridge.coef_)))

print('\nIntercept: {:>0.4f}'.format(ridge.intercept_))

print('\n' + 'RMSE: {:>0.4f}'.format(np.sqrt(mean_squared_error(Y_train, ridge.predict(X_train)))))

print('\nR^2: {:>0.4f}'.format(ridge.score(X_train, Y_train)))
# test set 

print('Coefficients:\n {}'.format(str(ridge.coef_)))

print('\nIntercept: {:>0.4f}'.format(ridge.intercept_))

print('\n' + 'RMSE: {:>0.4f}'.format(np.sqrt(mean_squared_error(Y_test, ridge.predict(X_test)))))

print('\nR^2: {:>0.4f}'.format(ridge.score(X_test, Y_test)))
#RMSE in dollars

print('Train RMSE ($): {:>0.2f}'.format(np.sqrt

    (mean_squared_error(np.exp(Y_train), np.exp(ridge.predict(X_train))))))

print('Test RMSE ($): {:>0.2f}'.format(np.sqrt

    (mean_squared_error(np.exp(Y_test), np.exp(ridge.predict(X_test))))))
# K-fold cross validation 

CVSridge = cross_val_score(ridge, attributes, data['LogPrice'], cv=10)

print('Cross validated R^2: {} \n'.format(CVSridge))

print('Mean cross validated R^2: {:>0.4f} \n'.format(CVSridge.mean()))



CVSridgeRMSE = cross_val_score(ridge, attributes, data['LogPrice'], scoring=RMSE_scorer, cv=10)

print('Cross validated RMSE: {}'.format(CVSridgeRMSE))

print('Mean cross validated RMSE: {:>0.04f}'.format(CVSridgeRMSE.mean()))
plt.figure(figsize = (24, 20))

plt.rcParams["axes.labelsize"] = 35

plt.rcParams["font.family"] = "Times New Roman"

plt.title('Residual Plot for Ridge Regression', fontsize=45)

plt.ylabel('Residual')

plt.xlabel('Log(Price)')

residuals = sns.residplot(Y_train, ridge.predict(X_train))

residuals_test = sns.residplot(Y_test, ridge.predict(X_test), color='r')

residuals_test.tick_params(labelsize=25)

plt.show()
ridge_prob_train = ridge.predict(X_train) - Y_train

plt.rcParams["axes.labelsize"] = 20

plt.figure(figsize=(12,10))

sp.stats.probplot(ridge_prob_train, plot=plt, fit=True)

plt.title('Probability Plot for Ridge Regression', fontsize=30)

plt.show()



ridge_prob_test = ridge.predict(X_test) - Y_test

plt.figure(figsize=(12,10))

sp.stats.probplot(ridge_prob_test, plot=plt, fit=True)

plt.title('Probability Plot for Ridge Regression', fontsize=30)

plt.show()
DTR_para = {'max_depth':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30],

           'min_samples_leaf': [1, 5, 10, 20, 50]}



print(GridSearchCV(DecisionTreeRegressor(), param_grid=DTR_para).fit(X_train, Y_train).best_estimator_)
max_depth = [1,5,10,20,30,50]



train_error = []

test_error = []



for i in max_depth:

    DTR = DecisionTreeRegressor(max_depth=i, min_samples_leaf=20, random_state=5)

    DTR.fit(X_train, Y_train)

    train_error.append(np.sqrt(mean_squared_error(Y_train, DTR.predict(X_train))))

    test_error.append(np.sqrt(mean_squared_error(Y_test, DTR.predict(X_test))))
plt.figure(figsize = (15, 10))

plt.title('Decision Tree: Model Complexity vs RMSE', fontsize=30)

plt.ylabel('RMSE', fontsize=20)

plt.xlabel('Tree Depth', fontsize=20)

sns.lineplot(max_depth, train_error, label='Train')

sns.lineplot(max_depth, test_error, color='#ce1256', label='Test')

plt.show()
DTR = DecisionTreeRegressor(max_depth=30, min_samples_leaf=20, random_state = 5)

DTR.fit(X_train, Y_train)
# train set

print('\n' + 'RMSE: {:>0.4f}'.format(np.sqrt(mean_squared_error(Y_train, DTR.predict(X_train)))))

print('\nR^2: {:>0.4f}'.format(DTR.score(X_train, Y_train)))
# test set 

print('\n' + 'RMSE: {:>0.4f}'.format(np.sqrt(mean_squared_error(Y_test, DTR.predict(X_test)))))

print('\nR^2: {:>0.4f}'.format(DTR.score(X_test, Y_test)))
#RMSE in dollars

print('Train RMSE ($): {:>0.2f}'.format(np.sqrt

    (mean_squared_error(np.exp(Y_train), np.exp(DTR.predict(X_train))))))

print('Test RMSE ($): {:>0.2f}'.format(np.sqrt

    (mean_squared_error(np.exp(Y_test), np.exp(DTR.predict(X_test))))))
# K-fold cross validation 

CVSDTR = cross_val_score(DTR, attributes, data['LogPrice'], cv=10)

print('Cross validated R^2: {} \n'.format(CVSDTR))

print('Mean cross validated R^2: {:>0.4f}'.format(CVSDTR.mean()))



CVSDTRRMSE = cross_val_score(DTR, attributes, data['LogPrice'], scoring=RMSE_scorer, cv=10)

print('Cross validated RMSE: {}'.format(CVSDTRRMSE))

print('Mean cross validated RMSE: {:>0.04f}'.format(CVSDTRRMSE.mean()))
plt.figure(figsize = (24, 20))

plt.rcParams["axes.labelsize"] = 35

plt.rcParams["font.family"] = "Times New Roman"

plt.title('Residual Plot for Decision Tree', fontsize=45)

plt.ylabel('Residual')

plt.xlabel('Log(Price)')

residual = sns.residplot(Y_train, DTR.predict(X_train))

residual_test = sns.residplot(Y_test, DTR.predict(X_test), color='r')

residual_test.tick_params(labelsize=25)

plt.show()
DTR_prob_train = DTR.predict(X_train) - Y_train

plt.rcParams["axes.labelsize"] = 20

plt.figure(figsize=(12,10))

sp.stats.probplot(DTR_prob_train, plot=plt, fit=True)

plt.title('Probability Plot for Decision Tree', fontsize=30)

plt.show()



DTR_prob_test = DTR.predict(X_test) - Y_test

plt.figure(figsize=(12,10))

sp.stats.probplot(DTR_prob_test, plot=plt, fit=True)

plt.title('Probability Plot for Decision Tree', fontsize=30)

plt.show()
RF_para = { 

    'n_estimators': [2,10, 20, 50, 100]}



print(GridSearchCV(RandomForestRegressor(), param_grid=RF_para).fit(X_train, Y_train).best_estimator_)
from sklearn.ensemble import RandomForestRegressor 



n_estimators = [2,10,20,50,100]



train_error = []

test_error = []



for i in n_estimators:

    RF = RandomForestRegressor(n_estimators=i, random_state=5)

    RF.fit(X_train, Y_train)

    train_error.append(np.sqrt(mean_squared_error(Y_train, RF.predict(X_train))))

    test_error.append(np.sqrt(mean_squared_error(Y_test, RF.predict(X_test))))
plt.figure(figsize = (15, 10))

plt.rcParams["font.family"] = "Times New Roman"

plt.title('Random Forest: Model Complexity vs RMSE', fontsize=35)

plt.ylabel('RMSE', fontsize=20)

plt.xlabel('Number of Trees', fontsize=20)

sns.lineplot(n_estimators, train_error, label='Train')

sns.lineplot(n_estimators, test_error, color='#ce1256', label='Test')

plt.show()
RF = RandomForestRegressor(n_estimators=100, random_state=5)

RF.fit(X_train, Y_train)
# train set

print('\n' + 'RMSE: {:>0.4f}'.format(np.sqrt(mean_squared_error(Y_train, RF.predict(X_train)))))

print('\nR^2: {:>0.4f}'.format(RF.score(X_train, Y_train)))
# test set 

print('\n' + 'RMSE: {:>0.4f}'.format(np.sqrt(mean_squared_error(Y_test, RF.predict(X_test)))))

print('\nR^2: {:>0.4f}'.format(RF.score(X_test, Y_test)))
#RMSE in dollars

print('Train RMSE ($): {:>0.2f}'.format(np.sqrt

    (mean_squared_error(np.exp(Y_train), np.exp(RF.predict(X_train))))))



print('Test RMSE ($): {:>0.2f}'.format(np.sqrt

    (mean_squared_error(np.exp(Y_test), np.exp(RF.predict(X_test))))))
# K-fold cross validation 

CVSRF = cross_val_score(RF, attributes, data['LogPrice'], cv=10)

print('Cross validated R^2: {} \n'.format(CVSRF))

print('Mean cross validated R^2: {:>0.4f} \n'.format(CVSRF.mean()))



CVSRFRMSE = cross_val_score(RF, attributes, data['LogPrice'], scoring=RMSE_scorer, cv=10)

print('Cross validated RMSE: {} \n'.format(CVSRFRMSE))

print('Mean cross validated RMSE: {:>0.04f}'.format(CVSRFRMSE.mean()))
plt.figure(figsize = (24, 20))

plt.rcParams["axes.labelsize"] = 35

plt.rcParams["font.family"] = "Times New Roman"

plt.title('Residual Plot for Random Forest', fontsize=45)

plt.ylabel('Residual')

plt.xlabel('Log(Price)')

residuals = sns.residplot(Y_train, RF.predict(X_train))

residuals_test = sns.residplot(Y_test, RF.predict(X_test), color='r')

residuals_test.tick_params(labelsize=25)

plt.show()
RF_prob_train = RF.predict(X_train) - Y_train

plt.rcParams["axes.labelsize"] = 20

plt.figure(figsize=(12,10))

sp.stats.probplot(RF_prob_train, plot=plt, fit=True)

plt.title('Probability Plot for Random Forest', fontsize=30)

plt.show()



RF_prob_test = RF.predict(X_test) - Y_test

plt.figure(figsize=(12,10))

sp.stats.probplot(RF_prob_test, plot=plt, fit=True)

plt.title('Probability Plot for Random Forest', fontsize=30)

plt.show()
print( '\033[1m' + '\nModel Evaluation:')

print('-'*100)

print('{} {:>30} {:>10} {:>10} {:>17} {:>17}'.format('Statistic',

      'Multiple Linear', 'Lasso', 'Ridge', 'Decision Tree', 'Random Forest'))

print('-'*100 + '\033[0m')



print('{} {:>25.3f} {:>16.3f} {:>10.3f} {:>12.3f} {:>17.3f}'.format('R^2 Test', lm.score(MLR_X_test, MLR_Y_test),

      lasso.score(X_test, Y_test), ridge.score(X_test, Y_test), DTR.score(X_test, Y_test), RF.score(X_test, Y_test)))



print('{} {:>24.3f} {:>16.3f} {:>10.3f} {:>12.3f} {:>17.3f}'.format('RMSE Test', 

      np.sqrt(mean_squared_error(MLR_Y_test, lm.predict(MLR_X_test))),

      np.sqrt(mean_squared_error(Y_test, lasso.predict(X_test))),

      np.sqrt(mean_squared_error(Y_test, ridge.predict(X_test))),

      np.sqrt(mean_squared_error(Y_test, DTR.predict(X_test))),

      np.sqrt(mean_squared_error(Y_test, RF.predict(X_test)))))



print('{} {:>20.0f} {:>16.0f} {:>10.0f} {:>12.0f} {:>17.0f}'.format('RMSE Test ($)',

      np.sqrt(mean_squared_error(np.exp(MLR_Y_test), np.exp(lm.predict(MLR_X_test)))),

      np.sqrt(mean_squared_error(np.exp(Y_test), np.exp(lasso.predict(X_test)))),

      np.sqrt(mean_squared_error(np.exp(Y_test), np.exp(ridge.predict(X_test)))),

      np.sqrt(mean_squared_error(np.exp(Y_test), np.exp(DTR.predict(X_test)))),

      np.sqrt(mean_squared_error(np.exp(Y_test), np.exp(RF.predict(X_test))))))

                                            

print('{} {:>15.0f} {:>16.0f} {:>10.0f} {:>12.0f} {:>17.0f}'.format('95% Confidence ($)',

      2*np.sqrt(mean_squared_error(np.exp(MLR_Y_test), np.exp(lm.predict(MLR_X_test)))),

      2*np.sqrt(mean_squared_error(np.exp(Y_test), np.exp(lasso.predict(X_test)))),

      2*np.sqrt(mean_squared_error(np.exp(Y_test), np.exp(ridge.predict(X_test)))),

      2*np.sqrt(mean_squared_error(np.exp(Y_test), np.exp(DTR.predict(X_test)))),

      2*np.sqrt(mean_squared_error(np.exp(Y_test), np.exp(RF.predict(X_test))))))



print('{} {:>20.3f} {:>16.3f} {:>10.3f} {:>12.3f} {:17.3f}'.format('R^2 Cross val', 

     CVSlm.mean(), CVSlasso.mean(), CVSridge.mean(), CVSDTR.mean(), CVSRF.mean()))



print('{} {:>19.3f} {:>16.3f} {:>10.3f} {:>12.3f} {:17.3f}'.format('RMSE Cross val', 

     CVSlmRMSE.mean(), CVSlassoRMSE.mean(), CVSridgeRMSE.mean(), CVSDTRRMSE.mean(), CVSRFRMSE.mean()))

from scipy.stats import ttest_ind 



l = np.sqrt((Y_test - lasso.predict(X_test))**2)

r = np.sqrt((Y_test - ridge.predict(X_test))**2)

m = np.sqrt((Y_test - lm.predict(MLR_X_test))**2)

d = np.sqrt((Y_test - DTR.predict(X_test) )**2)

rf = np.sqrt((Y_test - RF.predict(X_test) )**2)



# According to the t-test we are able to reject H0. The differences in RMSE are statistically significant.



print('\033[1m' + 'Lasso' '\033[0m' + '\nt: {:>0.1f}, p-value: {:>0.2f} \n'

      .format(ttest_rel(m, l)[0], ttest_rel(m, l)[1]))

print('\033[1m' + 'Ridge' '\033[0m' + ' \nt: {:>0.1f}, p-value: {:>0.2f} \n'

      .format(ttest_rel(m, r)[0], ttest_rel(r, m)[1]))

print('\033[1m' + 'Decision Tree' '\033[0m' + '\nt: {:>0.1f}, p-value: {:>0.2f} \n'

      .format(ttest_rel(m, d)[0], ttest_rel(m, d)[1]))

print('\033[1m' + 'Random Forest' '\033[0m' + '\nt: {:>0.1f}, p-value: {:>0.2f} \n'

      .format(ttest_rel(m, d)[0], ttest_rel(m, rf)[1]))



print('\nCan we reject H0 for the lasso regression?', 'Yes' if ttest_rel(l, m)[1]<0.05 else 'No')

print('\nCan we reject H0 for the ridge regression?', 'Yes' if ttest_rel(r, m)[1]<0.05 else 'No')

print('\nCan we reject H0 for the decision tree?', 'Yes' if ttest_rel(d, m)[1]<0.05 else 'No')

print('\nCan we reject H0 for the random forest?', 'Yes' if ttest_rel(rf, m)[1]<0.05 else 'No')