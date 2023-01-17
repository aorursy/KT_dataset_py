# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split



warnings.filterwarnings('ignore')

%matplotlib inline

# Any results you write to the current directory are saved as output.

housing_train = pd.read_csv('../input/train.csv')

housing_test = pd.read_csv('../input/test.csv')
housing_train.isnull().sum()
housing_test.isnull().sum()
housing_train.info()
# in EDA we can visualise the categorical objects
housing_train.shape
# VIsually exploring relations between features:
_ = sns.swarmplot(x='Street', y='SalePrice', data=housing_train)



# Label axes

_ = plt.xlabel('Street type')

_ = plt.ylabel('Sale Price $')

_= plt.title('The impact of Street type on the proprty Sale price in $')

# Make bee swarm plot



# Show the plot

plt.show()

# the figure represents the distripbution of the impact of street type on the proprty sale price
_ = sns.swarmplot(x='OverallQual', y='SalePrice', data=housing_train)



# Label axes

_ = plt.xlabel('overall quality')

_ = plt.ylabel('Sale Price $')

_= plt.title('The impact of Overall Quality rangeing from 1 to 10')

# Make bee swarm plot



# Show the plot

plt.show()
_ = sns.swarmplot(x='YearBuilt', y='SalePrice', data=housing_train)



# Label axes

_ = plt.xlabel('year built')

_ = plt.ylabel('Sale Price $')

_= plt.title('the impact of as the year of bulding increase price do to')

# Make bee swarm plot



# Show the plot

plt.show()
_ = sns.swarmplot(x='TotalBsmtSF', y='SalePrice', data=housing_train)



# Label axes

_ = plt.xlabel('Total Basment SF')

_ = plt.ylabel('Sale Price $')

_= plt.title('Impact of Total Basment SF on SalePrice')

# Make bee swarm plot



# Show the plot

plt.show()
fig, ax =plt.subplots(figsize=(20,20))

sns.heatmap(housing_train.corr(),  center=0, square=True, linewidths=1, ax=ax, annot=True);



#Tsale price's

sns.distplot(h_train.SalePrice);



# Show the plot

plt.show()

d =h_train.SalePrice

sx=h_train.GarageCars

sns.barplot(y= d, x=sx);

plt.ylabel('SalePrice')

plt.xlabel('GurageCars')

plt.title('As Gurage Cars room increase the Price do to')
sns.distplot(housing_train['SalePrice']);
#skewness and kurtosis

print("Skewness: %f" % housing_train['SalePrice'].skew())

print("Kurtosis: %f" % housing_train['SalePrice'].kurt())
#highly related numerical variables

var = 'GrLivArea'

data = pd.concat([housing_train['SalePrice'], housing_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([housing_train['SalePrice'], housing_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#to show the relation between the price and total basment space, they are highly related

# or we should put two features as x and compir their affect on price, the one with highes, is higher weight

# they should be two values in one column to compare between them , how each one affects sale, thats whay not working, here i put 2d array



_ = sns.swarmplot(x='TotalBsmtSF', y='SalePrice', data=housing_train)



# Label axes

_ = plt.xlabel('TotalBsmtSF')

_ = plt.ylabel('Sale price$')



# Show the plot

plt.show()
## Compute ECDFs

import numpy as np

from statsmodels.distributions.empirical_distribution import ECDF

x_TotalBsmtSF, y_TotalBsmtSF = ecdf('TotalBsmtSF')

x_GrLivArea, y_GrLivArea = ecdf('GrLivArea')



# Plot the ECDFs

_ = plt.plot(x_TotalBsmtSF, y_TotalBsmtSF, marker='.', linestyle='none')

_ = plt.plot(x_GrLivArea, y_GrLivArea, marker='.', linestyle='none')



# Set margins

plt.margins(0.02)



# Add axis labels and legend

_ = plt.xlabel('value')

_ = plt.ylabel('ECDF')

_ = plt.legend(('TotalBsmtSF', 'GrLivArea'), loc='lower right')



# Show the plot

plt.show()
#do more analyzing (feature engineering) the relationship between features and SalePrice feature by plotting

# we mast perform the heat map to se the relation ship 

# plasma 
## standardizing: 

#from sklearn.preprocessing import StandardScaler

#scaler =StandardScaler()



# Take a subset of the DataFrame you want to scale

#scaled_df = pd.DataFrame(scaler.fit_transform(housing_train), columns= housing_train.columns)

all_data= pd.concat([housing_train,housing_test])
all_data.isnull().sum()
### step 1: generating preprocessing   :

## step 2: concatenate the two dataframes

# step 3: transfering objects to numerical dummies:

#step 4: cleaned the data 

#step 5: model on the h_test:

# step 6: evalute models score results 

#step 7: apply the sutable model on the h_test dataframe

all_data.shape
#all_data
all_data.info()
housing_test.shape, housing_train.shape
all_data["GarageYrBlt"].fillna(0, inplace = True)
all_data["GarageArea"].fillna(0, inplace = True)

# filling the missing values: 

all_data['LotFrontage'].fillna(method = "ffill", inplace=True)

#housing_train[''].fillna(method = "", inplace=True)

all_data["GarageType"].fillna("No Gurage", inplace = True)

all_data["GarageFinish"].fillna("No Gurage", inplace = True)

all_data["GarageQual"].fillna("No Gurage", inplace = True)

all_data["GarageCond"].fillna("No Gurage", inplace = True)



all_data["BsmtCond"].fillna("No Basment", inplace = True)

all_data["BsmtExposure"].fillna("No Basment", inplace = True)

all_data["BsmtFinType1"].fillna("No Basment", inplace = True)

all_data["BsmtFinSF2"].fillna(0, inplace = True)

all_data["BsmtFinSF1"].fillna(0, inplace = True)

all_data["BsmtFinType2"].fillna("No Basment", inplace = True)

all_data["BsmtFullBath"].fillna(0, inplace = True)

all_data["BsmtHalfBath"].fillna(0, inplace = True)

all_data["BsmtQual"].fillna('No Basment', inplace = True)

all_data["BsmtUnfSF"].fillna(0, inplace = True)

all_data["Electrical"].fillna("Unknown", inplace = True)

all_data["Exterior1st"].fillna("Unknown", inplace = True)

all_data["Exterior2nd"].fillna("Unknown", inplace = True)

all_data["FireplaceQu"].fillna(0, inplace = True)

all_data["MasVnrType"].fillna("Unknown", inplace = True)

all_data["MasVnrArea"].fillna(0, inplace = True)

all_data["MiscFeature"].fillna("No ExtraFeatures", inplace = True)



all_data["SaleType"].fillna("Unknown", inplace = True)

all_data["TotalBsmtSF"].fillna(0, inplace = True)

all_data["Utilities"].fillna("Unknown", inplace = True)
all_data["SalePrice"].fillna(0, inplace = True)

all_data['Functional'].fillna('Typ', inplace =True)

all_data['MSZoning'].fillna(0, inplace= True)
all_data.drop(['Alley'], axis=1,inplace=True)

all_data.drop(['PoolQC'], axis=1,inplace=True)

all_data.drop(['Fence'], axis=1,inplace=True)

all_data.isnull().sum()

#all_data.SalePrice
#change the objects to dummies

all_dummy= pd.get_dummies(all_data, drop_first=True)
all_dummy.shape
h_train= all_dummy.iloc[:1460,:]
#h_train
h_train.shape
h_test=all_dummy.iloc[1460:,:]
h_test.shape
h_train['SalePrice'].describe()
#step 5: 

#5.1 : split the train sample into: test_ and train

#from sklearn.model_selection import cross_val_score

#from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split

X_t = h_train.drop(['SalePrice'], axis=1)

y= h_train.SalePrice

X_train, x_test, y_train, y_test = train_test_split(X_t, y, test_size=0.3, random_state=42)

lasso= Lasso(alpha=0.1, normalize=True)

lasso.fit(X_train, y_train)

Lasso.predict(x_test)

Lasso.score(x_test, y_test)



np.mean(cv_results)
h_train.isnull().sum().sort_values()
X_t.isnull().sum()
h_train['GarageYrBlt'].fillna(method = "ffill", inplace=True)

h_train['LotFrontage'].fillna(method = "ffill", inplace=True)
#step 5: 

#5.1 : split the train sample into: test_ and train

#from sklearn.model_selection import cross_val_score

#from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split

#X= h_train.drop(['SalePrice'], axis=1).astype('float64') --- X_t = h_train.drop(['SalePrice'], axis=1)

#X = h_train['2ndFlrSF','KitchenQual_Gd','WoodDeckSF','Neighborhood_NoRidge','MasVnrType_Stone','GarageType_Attchd']---y= h_train['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(X_t, y, test_size=0.7, random_state=42)

lasso= Lasso(alpha=0.1, normalize=True)

lasso.fit(x_train, y_train)

Lasso_pred= Lasso.predict(x_test)

Lasso.score(x_test, y_test,y)



np.mean(cv_results)

# Import Lasso

from sklearn.linear_model import Lasso



# Instantiate a lasso regressor: lasso

lasso = Lasso(alpha=0.8, normalize=True)



# Fit the regressor to the data

lasso.fit(X_t, y)

X_t = h_train.drop(['SalePrice'], axis=1)

y= h_train['SalePrice']



# Compute and print the coefficients

lasso_coef = lasso.coef_

print(lasso_coef)

np.mean(lasso_coef)

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor



# Read the data train = pd.read_csv('../input/train.csv')



# pull data into target (y) and predictors (X)

train_y = h_train.SalePrice

predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd','Id']



# Create training predictors data

train_X = h_train[predictor_cols]



my_model = RandomForestRegressor()

my_model.fit(train_X, train_y)
# Import necessary modules

from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score

X_t = h_train.drop(['SalePrice'], axis=1)

y= h_train['SalePrice']

# Setup the array of alphas and lists to store scores

alpha_space = np.logspace(-4, 0, 50)

ridge_scores = []

ridge_scores_std = []



# Create a ridge regressor: ridge

ridge = Ridge(normalize=True)



# Compute scores over range of alphas

for alpha in alpha_space:



    # Specify the alpha value to use: ridge.alpha

    ridge.alpha = alpha

    

    # Perform 10-fold CV: ridge_cv_scores

    ridge_cv_scores = cross_val_score(ridge, X_t, y, cv=5)

    

    # Append the mean of ridge_cv_scores to ridge_scores

    ridge_scores.append(np.mean(ridge_cv_scores))

    

    # Append the std of ridge_cv_scores to ridge_scores_std

    ridge_scores_std.append(np.std(ridge_cv_scores))



# Display the p

ridge_scores, ridge_scores_std
# Import necessary modules to show the MSE

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

X_t = h_train.drop(['SalePrice'], axis=1)

y= h_train['SalePrice']

# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size = 0.3, random_state=42)



# Create the regressor: reg_all

reg_all = LinearRegression()



# Fit the regressor to the training data

reg_all.fit(X_train, y_train)



# Predict on the test data: y_pred

y_pred = reg_all.predict(X_test)



# Compute and print R^2 and RMSE

print("R^2: {}".format(reg_all.score(X_test, y_test)))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Root Mean Squared Error: {}".format(rmse))


from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score



# Create a linear regression object: reg

reg = LinearRegression()



# Perform 3-fold CV

cvscores_3 = cross_val_score(reg, X_t, y, cv = 3)

print(cvscores_3)



# Perform 2-fold CV

cvscores_2 = cross_val_score(reg, X_t, y, cv = 7)

print(cvscores_2)
from sklearn.linear_model import RidgeCV

from sklearn.model_selection import train_test_split



X_train, x_test, y_train, y_test = train_test_split(X_t, y, test_size=0.7, random_state=42)

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



#print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())

#print("Ridge RMSE on Test set :", rmse_cv_test(ridge).mean())

y_train_rdg = ridge.predict(X_train)

y_test_rdg = ridge.predict(X_test)

# Find most important features relative to target

print("Find most important features relative to target")

corr = h_train.corr()

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

#print(corr.SalePrice



corr.SalePrice

#np.where(np.isnan(X)) 
#np.nan_to_num(X)
#np.isnan(x)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score



# Create a linear regression object: reg

X_t = h_train.drop(['SalePrice'], axis=1)

y= h_train['SalePrice']

reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores

cv_scores = cross_val_score(reg, X_t, y, cv=5)



# Print the 5-fold cross-validation scores

print(cv_scores)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score



# Create a linear regression object: reg

X_t = h_train.drop(['SalePrice'], axis=1)

y= h_train.SalePrice

reg = LinearRegression()
from sklearn.model_selection import train_test_split

#1- prpare the data: (Choose somthing to predict as the target, example: tip ammount so we drop it)

X=h_train.drop('SalePrice', axis=1)

#so we will put all the SalePrice's in y

y=h_train.SalePrice



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
# init , train, test model:

from sklearn.linear_model import LinearRegression

model =LinearRegression()

model.fit(X_train, y_train)

model.score(X_test, y_test)
#model evaluation

import numpy as np

from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
# training score is the out score, here comparing the training score with testing score

model.score(X_train, y_train), model.score(X_test, y_test)
#cross validation (here we are doing the test on the whole samples,without the model testing)

# we can shuffle the X, and y by rearranging them by randomizing them, or use the KFold()

#so for cv you can KFold object with shuffle

from sklearn.model_selection import cross_val_score

cross_val_score(LinearRegression(), X, y, cv=5)
from sklearn.model_selection import cross_val_score

reg =LinearRegression()

cros_results = cross_val_score(reg, X_t, y, cv=6)

print(cros_results)

np.mean(cros_results)
# Import Lasso

from sklearn.linear_model import Lasso



# Instantiate a lasso regressor: lasso

lasso = Lasso(alpha=0.8, normalize=True)



# Fit the regressor to the data

lasso.fit(X_t, y)

df_columns= X_t.columns

# Compute and print the coefficients

lasso_coef = lasso.coef_

print(lasso_coef)

# Plot the coefficients

plt.plot(range(len(df_columns)), lasso_coef)

plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)

plt.margins(0.02)

plt.show()
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
X_t = h_train.drop(['SalePrice'], axis=1)

y= h_train.SalePrice

model_lasso = LassoCV(alphas = [8, 6]).fit(X_t, y)
LassoCV.predict(x_test)
coef = pd.Series(model_lasso.coef_, index = X_t.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
# Read the test data

test = h_test

# Treat the test data in the same way as training data. In this case, pull same columns.

test_X = test[predictor_cols]

# Use the model to make predictions

predicted_prices = my_model.predict(test_X)

# We will look at the predicted prices to ensure we have something sensible.

print(predicted_prices)
h_test["GarageCars"].fillna(0, inplace = True)

my_submission = pd.DataFrame({'Id': h_test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
print(my_submission.Id, predicted_prices)