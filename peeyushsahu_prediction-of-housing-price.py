# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import skew
from scipy import stats
from sklearn.metrics import mean_squared_error, make_scorer
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv', header=0, sep=',')
train_df = train_df.loc[:,'MSSubClass':]
#print('Training data\n', train_df.head())
print('Training data columns:', train_df.columns)
print('Training data shape', train_df.shape)

test_df = pd.read_csv('../input/test.csv', header=0, sep=',')
test_df = test_df.loc[:,'MSSubClass':]
#print('Test data\n', test_df.head())
print('Training data columns:', test_df.columns)
print('Test data shape\n', test_df.shape)
# Missing data
print('Missing data points in every features')
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data[:20])
# Findout how many of the columns are caregorical or numerical
nan_columns = missing_data[missing_data['Percent'] > 0.0].index
filr_train_df = train_df[missing_data[missing_data['Percent'] <= 0.0].index]
numr_cols = filr_train_df.dtypes[filr_train_df.dtypes != "object"].index # Numerical columns
catg_cols = filr_train_df.dtypes[filr_train_df.dtypes == "object"].index # Categorical columns
print(catg_cols)
#dealing with missing data
for ind, col in filr_train_df[catg_cols].iteritems(): # What kind of data in categorical columns
    print(ind, set(list(col)))
# Corrilation of numerical features with sales prize
print('Number of features:', len(filr_train_df.columns))
filr_train_df_corr = filr_train_df.corr()
highly_corr = filr_train_df_corr[(filr_train_df_corr['SalePrice'] > 0.1)].index # No negative correlation in data
corr_sale = filr_train_df.loc[:,highly_corr].corr()
print('Number of features with corr:', len(corr_sale))
plt.figure(figsize=(10,10))
sns.heatmap(corr_sale, annot=True, square=True)
# Now check for skewnwss of numerical features
skewed_feats = filr_train_df[highly_corr].apply(lambda x: skew(x.dropna())) #compute skewness
print('Skewness in feature data\n',skewed_feats)
# Correct skewness for features with positive skewness > 0.5 with log1p transformation
LT_columns = []
numLT_train_df = pd.DataFrame()
for ind, skew in skewed_feats.iteritems():
    if (skew > 0.5):
        numLT_train_df = pd.concat([numLT_train_df, np.log1p(filr_train_df[ind])], axis=1)
        LT_columns.append(ind)
    else:
        numLT_train_df = pd.concat([numLT_train_df, filr_train_df[ind]], axis=1)
        
# Example of skewness
print('Example feature:', LT_columns[0])
skew_ex = pd.DataFrame({'Not_trsf': filr_train_df[LT_columns[0]], 'log_trsf':numLT_train_df[LT_columns[0]]})
skew_ex.hist()
# Can also look at normal prob plot
#histogram and normal probability plot
sns.set()
sns.distplot(numLT_train_df[LT_columns[0]], fit=norm);
fig = plt.figure()
res = stats.probplot(numLT_train_df[LT_columns[0]], plot=plt)
print(numLT_train_df.shape)
sns.pairplot(numLT_train_df.iloc[:,:5])
# What features we want to take
catg_cols # these are our categorical features
highly_corr # numeric features which we have selected based on correlation
LT_columns # numeric features which needs to be transformed
# Concat test and training for preprocessing
tot_data = pd.concat([train_df.loc[:,'MSSubClass':], test_df.loc[:,'MSSubClass':]], ignore_index=True)
print(tot_data.shape)
print(tot_data.head())
# Create dummy variable for categorical features
tot_data_cat = pd.get_dummies(tot_data[catg_cols])
tot_data_cat.head()
# Log transform numerical data
tot_data_num = tot_data[highly_corr]
print(tot_data_num.shape)
for cols in LT_columns:
    tot_data_num.loc[:,cols] = np.log1p(tot_data_num.loc[:,cols])
print(tot_data_num.shape)
tot_data_num.head()
# Combining preprocessed data
tot_data_pro = pd.concat([tot_data_cat, tot_data_num],axis=1)
tot_data_pro.head()
# Creating matrix for sklern 
pr_trainData = tot_data_pro[:1000]
pr_testData = tot_data_pro[1000:1400]
pr_testData = pr_testData.fillna(pr_testData.mean())
print(pr_trainData.shape, pr_testData.shape)
Y = pr_trainData['SalePrice']
X_trainData = pr_trainData.drop('SalePrice', axis=1)
X_testData = pr_testData.drop('SalePrice', axis=1)
print(X_trainData.shape, X_testData.shape)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score, train_test_split

# Define error measure for official scoring : RMSE
scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, X_trainData, Y, scoring = scorer, cv = 5))
    return(rmse)
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge.fit(X_trainData, Y)
alpha = ridge.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 5)
ridge.fit(X_trainData, Y)
alpha = ridge.alpha_
print("Best alpha :", alpha)

print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())
y_train_rdg = ridge.predict(X_trainData)
y_test_rdg = ridge.predict(X_testData)
# Plot residuals
sns.set()
plt.scatter(y_train_rdg, y_train_rdg - Y, c = "blue", marker = "o", label = "Training data", alpha=0.7)
plt.scatter(y_test_rdg, y_test_rdg - pr_testData['SalePrice'], c = "green", marker = "o", label = "Validation data", alpha=0.7)
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.scatter(y_train_rdg, Y, c = "blue", marker = "o", label = "Training data", alpha=0.7)
plt.scatter(y_test_rdg, pr_testData['SalePrice'], c = "green", marker = "o", label = "Validation data", alpha=0.7)
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

# Plot important coefficients
coefs = pd.Series(ridge.coef_, index = X_trainData.columns)
print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Ridge Model")
plt.show()
# Here we can see the predicted price
#saleprice = pd.Series(y_test_rdg, index=X_testData.index, name='SalePrice')
#pd.concat([X_testData, saleprice], axis=1).head()
lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1], 
                max_iter = 50000, cv = 10)
lasso.fit(X_trainData, Y)
alpha = lasso.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 10)
lasso.fit(X_trainData, Y)
alpha = lasso.alpha_
print("Best alpha :", alpha)

print("Lasso RMSE on Training set :", rmse_cv_train(lasso).mean())
y_train_las = lasso.predict(X_trainData)
y_test_las = lasso.predict(X_testData)

# Plot residuals
plt.scatter(y_train_las, y_train_las - Y, c = "blue", marker = "s", label = "Training data", alpha=0.7)
plt.scatter(y_test_las, y_test_las - pr_testData['SalePrice'], c = "green", marker = "s", label = "Validation data", alpha=0.7)
plt.title("Linear regression with Lasso regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.scatter(y_train_las, Y, c = "blue", marker = "s", label = "Training data", alpha=0.7)
plt.scatter(y_test_las, pr_testData['SalePrice'], c = "green", marker = "s", label = "Validation data", alpha=0.7)
plt.title("Linear regression with Lasso regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

# Plot important coefficients
coefs = pd.Series(lasso.coef_, index = X_trainData.columns)
print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()