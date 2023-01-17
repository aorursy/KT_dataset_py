#Importing fundamental data exploration libaries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr





%config InlineBackend.figure_format = 'png' #set 'png' here when working on notebook

%matplotlib inline



df_train = pd.read_csv("../input/kc_house_data.csv")

df_train.rename(columns ={'price': 'SalePrice'}, inplace =True)



df_train.head()
#checking the columns in the dataset

df_train.columns
df_train['SalePrice'].describe()

df_train['SalePrice']=df_train['SalePrice']


#histogram

sns.distplot(df_train['SalePrice'], bins=50, kde=False);







print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
var = 'sqft_living15'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(3,8000000));
var = 'bedrooms'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(14, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=3500000);
var = 'grade'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=3500000);
var = 'bathrooms'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(20, 20))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=3500000);
var = 'yr_built'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'sqft_living', 'grade', 'sqft_above', 'view', 'bathrooms']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();

#missing data

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#standardizing data to mitigate skewdness and kurtosis

from sklearn.preprocessing import StandardScaler

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
from scipy.stats import norm

from scipy import stats

#histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit=norm, bins=50, kde=False);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
#applying log transformation

df_train['SalePrice'] = np.log(df_train['SalePrice'])
#transformed histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit=norm,  bins=50, kde=False);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['sqft_living'], fit=norm, bins=50, kde=False);

fig = plt.figure()

res = stats.probplot(df_train['sqft_living'], plot=plt)
#data transformation

df_train['sqft_living'] = np.log(df_train['sqft_living'])
#transformed histogram and normal probability plot

sns.distplot(df_train['sqft_living'], fit=norm, bins=50, kde=False);

fig = plt.figure()

res = stats.probplot(df_train['sqft_living'], plot=plt)
#scatter plot

plt.scatter(df_train['sqft_living'], df_train['SalePrice']);

df_train.head()
Y = df_train.SalePrice.values

feature_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',

       'view', 'condition', 'grade', 'sqft_above',

       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',

       'sqft_living15', 'sqft_lot15']

X=df_train[feature_cols]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X, Y, random_state=3)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)
accuracy = regressor.score(x_test, y_test)

"Accuracy: {}%".format(int(round(accuracy * 100)))
from sklearn import ensemble, tree, linear_model

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.utils import shuffle

# For accurate scoring

def get_score(prediction, lables):    

    print('R2: {}'.format(r2_score(prediction, lables)))

    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))
def train_test(estimator, x_trn, x_tst, y_trn, y_tst):

    prediction_train = estimator.predict(x_trn)

    # Printing estimator

    print(estimator)

    # Printing train scores

    get_score(prediction_train, y_trn)

    prediction_test = estimator.predict(x_tst)

    # Printing test scores

    print("Test")

    get_score(prediction_test, y_tst)
from sklearn import ensemble, tree, linear_model

ENSTest = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(x_train, y_train)
train_test(ENSTest, x_train, x_test, y_train, y_test)
# Average R2 score and standart deviation of 5-fold cross-validation

scores = cross_val_score(ENSTest, x_test, y_test, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',

                                               min_samples_leaf=15, min_samples_split=10, loss='huber').fit(x_train, y_train)

train_test(GBest, x_train, x_test, y_train, y_test)
# Average R2 score and standart deviation of 5-fold cross-validation

scores = cross_val_score(GBest, x_test, y_test, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, make_scorer

from scipy.stats import skew

from IPython.display import display

import matplotlib.pyplot as plt
# Defining two functions for error measuring: RMSE

scorer = make_scorer(mean_squared_error, greater_is_better = False)



def rmse_cv_train(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))

    return(rmse)



def rmse_cv_test(model):

    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))

    return(rmse)



X_train= x_train

X_test= x_test
# Linear Regression

lr = LinearRegression()

lr.fit(x_train, y_train)



# Look at predictions on training and validation set

print("RMSE on Training set :", rmse_cv_train(lr).mean())

print("RMSE on Test set :", rmse_cv_test(lr).mean())

y_train_pred = lr.predict(x_train)

y_test_pred = lr.predict(x_test)



# Plot residuals

plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 11.5, xmax = 15.5, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([11.5, 15.5], [11.5, 15.5], c = "red")

plt.show()
# 2* Ridge

ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Best alpha :", alpha)



print("Try again for more precision with alphas centered around " + str(alpha))

ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 

                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 

                cv = 10)

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Best alpha :", alpha)



print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())

print("Ridge RMSE on Test set :", rmse_cv_test(ridge).mean())

y_train_rdg = ridge.predict(X_train)

y_test_rdg = ridge.predict(X_test)



# Plot residuals

plt.scatter(y_train_rdg, y_train_rdg - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_rdg, y_test_rdg - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with Ridge regularization")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 11.5, xmax = 15.5, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train_rdg, y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_rdg, y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with Ridge regularization")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([11.5, 15.5], [11.5, 15.5], c = "red")

plt.show()



# Plot important coefficients

coefs = pd.Series(ridge.coef_, index = X_train.columns)

print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \

      str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")

plt.title("Coefficients in the Ridge Model")

plt.show()
# 3* Lasso

lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 

                          0.3, 0.6, 1], 

                max_iter = 50000, cv = 10)

lasso.fit(X_train, y_train)

alpha = lasso.alpha_

print("Best alpha :", alpha)



print("Try again for more precision with alphas centered around " + str(alpha))

lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 

                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 

                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 

                          alpha * 1.4], 

                max_iter = 50000, cv = 10)

lasso.fit(X_train, y_train)

alpha = lasso.alpha_

print("Best alpha :", alpha)



print("Lasso RMSE on Training set :", rmse_cv_train(lasso).mean())

print("Lasso RMSE on Test set :", rmse_cv_test(lasso).mean())

y_train_las = lasso.predict(X_train)

y_test_las = lasso.predict(X_test)



# Plot residuals

plt.scatter(y_train_las, y_train_las - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_las, y_test_las - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with Lasso regularization")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 11.5, xmax = 15.5, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train_las, y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_las, y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with Lasso regularization")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([11.5, 15.5], [11.5, 15.5], c = "red")

plt.show()



# Plot important coefficients

coefs = pd.Series(lasso.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \

      str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")

plt.show()
# 4* ElasticNet

elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],

                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 

                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 

                          max_iter = 50000, cv = 10)

elasticNet.fit(X_train, y_train)

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )



print("Try again for more precision with l1_ratio centered around " + str(ratio))

elasticNet = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],

                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 

                          max_iter = 50000, cv = 10)

elasticNet.fit(X_train, y_train)

if (elasticNet.l1_ratio_ > 1):

    elasticNet.l1_ratio_ = 1    

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )



print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) + 

      " and alpha centered around " + str(alpha))

elasticNet = ElasticNetCV(l1_ratio = ratio,

                          alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9, 

                                    alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, 

                                    alpha * 1.35, alpha * 1.4], 

                          max_iter = 50000, cv = 10)

elasticNet.fit(X_train, y_train)

if (elasticNet.l1_ratio_ > 1):

    elasticNet.l1_ratio_ = 1    

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )



print("ElasticNet RMSE on Training set :", rmse_cv_train(elasticNet).mean())

print("ElasticNet RMSE on Test set :", rmse_cv_test(elasticNet).mean())

y_train_ela = elasticNet.predict(X_train)

y_test_ela = elasticNet.predict(X_test)



# Plot residuals

plt.scatter(y_train_ela, y_train_ela - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_ela, y_test_ela - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with ElasticNet regularization")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 11.5, xmax = 15.5, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train, y_train_ela, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test, y_test_ela, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with ElasticNet regularization")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([11.5, 15.5], [11.5, 15.5], c = "red")

plt.show()



# Plot important coefficients

coefs = pd.Series(elasticNet.coef_, index = X_train.columns)

print("ElasticNet picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")

plt.title("Coefficients in the ElasticNet Model")

plt.show()