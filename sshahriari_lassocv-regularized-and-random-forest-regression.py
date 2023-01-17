import numpy as np 

import pandas as pd 

from collections import OrderedDict



import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns



from scipy.stats import skew



from sklearn import linear_model

from sklearn.linear_model import LassoCV

from sklearn.cross_validation import cross_val_score



# “ggplot”: matches the beauty of ggplot plotting package for R

plt.style.use('ggplot')



# To display matplotlib plots in a notebook cell rather than in another window.

%matplotlib inline  
# Read train and test data



train_orig = pd.read_csv("../input/train.csv")

test_orig = pd.read_csv("../input/test.csv")



train = train_orig [:]

test = test_orig [:]



print(train.shape)



print(train.head())



#To check if the variables have correct type

print(train.dtypes)
print("Housing Sale Price Statistic Description:")

print(train["SalePrice"].describe())
train[pd.isnull(train).any(axis=1)].shape
for i in train.columns:

    print(i, sum(train[i].isnull()))    
train_Num = train.select_dtypes(include = ["float64", "int64"])

train_NaNCol_Num = train_Num.columns[train_Num.isnull().any()].tolist()



train_Obj = train.select_dtypes(include = ["object"])

train_NaNCol_Obj = train_Obj.columns[train_Obj.isnull().any()].tolist()



print("\nList of numerical (Float, Int) attributes with NaN values:\n",train_NaNCol_Num)



print("\nList of categorical attributes with NaN values:\n",train_NaNCol_Obj)
test_Num = test.select_dtypes(include = ["float64", "int64"])

test_NaNCol_Num = test_Num.columns[test_Num.isnull().any()].tolist()



test_Obj = test.select_dtypes(include = ["object"])

test_NaNCol_Obj = test_Obj.columns[test_Obj.isnull().any()].tolist()



print("\nList of numerical (Float, Int) attributes with NaN values:\n",test_NaNCol_Num)



print("\nList of categorical attributes with NaN values:\n",test_NaNCol_Obj)
fig_dims = (len(train_NaNCol_Num), 2)



for i in range(len(train_NaNCol_Num)):

    col=train_NaNCol_Num[i]

    plt.subplot2grid(fig_dims, (i, 0))

    plt.xlabel(col)

    plt.ylabel("Frequency")   

    plt.hist(train[col][~train[col].isnull()])

    

    plt.subplot2grid(fig_dims, (i, 1))

    plt.xlabel(col)

    plt.ylabel("SalePrice")    

    plt.scatter(train[col],train["SalePrice"],s=10)
train.LotFrontage = train.LotFrontage.fillna(train.LotFrontage.median())

train.MasVnrArea = train.MasVnrArea.fillna(train.MasVnrArea.median())
test.LotFrontage = test.LotFrontage.fillna(test.LotFrontage.median())

test.MasVnrArea = test.MasVnrArea.fillna(test.MasVnrArea.median())
plt.xlabel("YearBuilt")

plt.ylabel("GarageYrBlt")

plt.scatter(train["YearBuilt"],train["GarageYrBlt"],s=10)
train.GarageYrBlt = train.GarageYrBlt.fillna(train.YearBuilt)
test.GarageYrBlt = test.GarageYrBlt.fillna(test.YearBuilt)
train.GarageYrBlt = train.GarageYrBlt.astype(int)
test.GarageYrBlt = test.GarageYrBlt.astype(int)
for i in test_NaNCol_Num:

    test[i] = test[i].fillna(test[i].mean())
sum(~train.PoolQC.isnull())
plt.figure(figsize=(4, 5))

plt.xlabel("Fence")

plt.ylabel("SalePrice")         

sns.boxplot(x="Fence", y="SalePrice", data=train)
train = train.drop(labels=["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1)  

test = test.drop(labels=["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1)  
# Updating the list of categorical variables with NaN:

train_NaNCol_Obj = [e for e in train_NaNCol_Obj if e not in ("Alley", "PoolQC", "Fence", "MiscFeature")]

for i in train_NaNCol_Obj:

    train[i] = train[i].fillna(train[i].value_counts().index[0])

    

test_NaNCol_Obj = [e for e in test_NaNCol_Obj if e not in ("Alley", "PoolQC", "Fence", "MiscFeature")]

for i in test_NaNCol_Obj:

    test[i] = test[i].fillna(test[i].value_counts().index[0])   
print(train.isnull().any().any()) 

print(test.isnull().any().any())
# Look at the histogram for "SalePrice":

plt.hist(train["SalePrice"])
corr = train.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

corr_coef_series = abs(corr['SalePrice']).sort_values(ascending=False)

print(corr_coef_series)
train_HighCorr_Sale = corr_coef_series[abs(corr['SalePrice']) > 0.6].index.values

train_HighCorr_Sale = np.delete(train_HighCorr_Sale, 0)
plt.figure(figsize=(12, 12))



for i in range(len(train_HighCorr_Sale)):

    h_cor_var=train_HighCorr_Sale[i]

    plt.subplot(3,2,i+1)    

    plt.xlabel(h_cor_var)

    plt.ylabel("SalePrice")  

    sns.regplot(train[h_cor_var], train["SalePrice"], line_kws={'color': 'black'})
train_LowCorr_Sale = corr_coef_series[abs(corr['SalePrice']) < 0.2].index.values

train_LowCorr_Sale = np.delete(train_LowCorr_Sale, 0)
train = train.drop(labels = train_LowCorr_Sale, axis=1)  

test = test.drop(labels = train_LowCorr_Sale, axis=1)  
train_Num = train.select_dtypes(include = ["object"])

train_Num.columns.tolist()
fig, ax = plt.subplots(2, 1, figsize = (5, 10))

sns.boxplot(x = "MSZoning", y = "SalePrice", data = train, ax = ax[0])

sns.boxplot(x = "CentralAir", y = "SalePrice", data = train, ax = ax[1])
train = train.drop(labels=["Functional", "LotConfig", "LandSlope", "BsmtFinType1", "BsmtFinType2", "GarageCond", "Utilities"], axis=1)  

test = test.drop(labels=["Functional", "LotConfig", "LandSlope", "BsmtFinType1", "BsmtFinType2", "GarageCond", "Utilities"], axis=1)  
print(train_orig.shape,train.shape)

print(test_orig.shape,test.shape)
all_data = pd.concat((train.loc[:,"MSZoning":"SaleCondition"],test.loc[:,"MSZoning":"SaleCondition"]))
matplotlib.rcParams["figure.figsize"] = (8.0, 4.0) 

prices = pd.DataFrame({"SalePrice" : train["SalePrice"], "log(SalePrice + 1)" : np.log1p(train["SalePrice"])})

prices.hist()
# log transform the target ("SalePrice"):

train["SalePrice"] = np.log1p(train["SalePrice"])



# log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index   



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) 

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
model_lasso = LassoCV(alphas = [0.5, 0.1, 0.05, 0.001, 0.0005]).fit(X_train, y)
rmse_lassocv =np.sqrt(-cross_val_score(model_lasso, X_train, y, scoring="neg_mean_squared_error", cv = 5)).mean()

print(rmse_lassocv)
coef = pd.Series(model_lasso.coef_, index = X_train.columns)

imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)  

imp_coef.plot(kind = "bar")

plt.title("Weight Coefficients in the Lasso Model")
# Fitting the model to the training set

from sklearn.ensemble import RandomForestRegressor

model_RF = RandomForestRegressor(n_estimators = 100) 



# Applying Grid Search to find the best model and the best parameters 

from sklearn.grid_search import GridSearchCV



parameters = [{'n_estimators':[200,300,400], 'min_samples_split':[3,9,15],'min_samples_leaf':[3,5,7]},

             {'n_estimators':[50, 60, 70], 'max_depth': [5, 7, 9], 'min_samples_leaf': [30, 40, 50]}]



grid_search = GridSearchCV(estimator = model_RF,

                           param_grid = parameters,

                           scoring = 'neg_mean_squared_error',

                           cv = 5,           

                           n_jobs = -1)      



grid_search = grid_search.fit(X_train, y)



best_scoring = grid_search.best_score_

rmse_regress = np.sqrt(np.abs(best_scoring))  # In above, I selected the 'mean_squared_error: mse' scoring. 

print(rmse_regress)



best_parameters = grid_search.best_params_

print("best_parameters:",best_parameters)
pred_test = np.expm1(model_lasso.predict(X_test))



solution = pd.DataFrame(OrderedDict([("id", test.Id), ("SalePrice", pred_test)]))

solution.to_csv("solutionLassoCV.csv", index = False)