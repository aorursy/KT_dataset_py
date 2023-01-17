# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Helpful packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from scipy.stats import norm, skew

from sklearn.linear_model import LassoCV, LassoLarsCV

from sklearn import metrics



train_data = pd.read_csv("../input/train.csv")

train_data.head()
train_data.info()

train_data.columns



numeric_columns = [col for col in train_data.columns if train_data.dtypes[col] != 'object']

categorical_columns = [col for col in train_data.columns if train_data.dtypes[col] == 'object']

print(f' Numeric columns: \n {numeric_columns} \n\n')

print(f' Categorical columns: \n {categorical_columns}')

train_data['SalePrice'].describe()
# Right (positive) skewness on the histogram, i.e. many houses are sold for less than the average value

plt.figure(figsize=(12,4))

sns.distplot(train_data['SalePrice'], bins=60, fit=norm)
# Kurtosis > 3 indicating that the prices are heavy-tailed or profusion of outliers



skewness = train_data['SalePrice'].skew()

kurt = train_data['SalePrice'].kurt()

print(f' Skewness: {skewness} \n Kurtosis: {kurt}')

# Matrix form for correlation data

# Obviously there are some strongly correlated variables

# (e.g., GarageCars/GarageArea, TotalBsmtSF/1stFlrSF, YearBlt/GarageYearBlt)



train_data.corr()



plt.figure(figsize=(14,8))

sns.heatmap(train_data.corr(), cmap="coolwarm", linecolor="white", linewidth=1)
price_corr_values = train_data.corr()[['SalePrice']]

price_corr_values.sort_values(by="SalePrice", axis=0, ascending=False, inplace=True)

price_corr_values.head(10)
# Let's look at a few features with positive correlation: OverallQual, GrLivArea, GarageCars/Area, TotalBsmtSF

# Examining scatter plots we can find some linear patterns  

# OverallQual: the overall material and finish of the house

sns.scatterplot(x=train_data['OverallQual'], y=train_data['SalePrice'])
# check GrLivArea: Above grade (ground) living area square feet



sns.scatterplot(x=train_data['GrLivArea'], y=train_data['SalePrice'])
# scatter plot for GarageArea: Size of garage in square feet



sns.scatterplot(x=train_data['GarageArea'], y=train_data['SalePrice'])
# scatter plot for TotalBsmtSF: Total square feet of basement area



sns.scatterplot(x=train_data['TotalBsmtSF'], y=train_data['SalePrice'])

sns.set()

columns = ['SalePrice','OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF']

sns.pairplot(train_data[columns])

plt.show()
# Concatenating the test and training data sets allows us to get around issues with the test data:

# 1. there are features that contain nan/null values in the test set that the train set doesn't

# 2. there are categorical levels in the test data that are not in the train data

X_test = pd.read_csv("../input/test.csv")

y_test = pd.read_csv("../input/sample_submission.csv")['SalePrice']  # assuming we have real prices for the test data

test_data = pd.concat([X_test, y_test], axis=1)

combined_data = pd.concat([train_data, test_data], axis=0)

# Check duplicates 

uniqueRows = len(set(combined_data.Id))

totalRows = len(combined_data.Id)

duplicateRows = totalRows - uniqueRows



assert duplicateRows == 0, "Warning: there are duplicate entries in the dataset"
# Check missing data - there are more than 30 features with missing observations

missing_elements = combined_data.isnull().sum().sort_values(ascending=False)

missing_elements[missing_elements > 0]
# PoolQC, MiscFeature, Alley, Fence, FireplaceQu have too many missing values (more than 40%) 

# so filling the missing data doesn't make any sense for these columns



# We will be using 'GarageArea' - it keeps the most important information regarding garages. So 'GarageType', 'GarageYrBlt', 

# 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond' can be dropped too.





# For features 'BsmtX', 'MasVnrX', etc. fill missing values with the most common occurances or the average values

# TO DO: avoid using test data when getting mean/mode for the train set



to_fill = ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtCond','BsmtQual', 'MasVnrArea', 'MasVnrType', 'Electrical', 'MSZoning', 

           'Utilities', 'BsmtHalfBath', 'BsmtFullBath', 'Functional', 'Exterior1st', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 

           'GarageArea', 'KitchenQual', 'SaleType', 'BsmtFinSF1', 'Exterior2nd', 'LotFrontage']

for col in to_fill:

   combined_data[col].fillna(combined_data[col].dropna().mode()[0] if col in categorical_columns

                             else combined_data[col].dropna().mean(), inplace=True) 



# removing missing data

missing_elements = combined_data.isnull().sum().sort_values(ascending=False)

combined_data = combined_data.drop((missing_elements[missing_elements > 0]).index,1)

combined_data.isnull().sum().max()



# Converting categorical variable into dummy

combined_data = pd.get_dummies(combined_data)



# Split back to the train and test data sets

train_data = combined_data.head(1460)

test_data = combined_data.tail(1459)

# Detecting outliers

# Any data-point that has a z-score higher than 3 is likely to be an anomaly



def detect_outlier(data):

    

    threshold=3

    outliers=[]

    

    mean = np.mean(data)

    std = np.std(data)

    

    

    for y in data:

        z_score= (y - mean)/std 

        if np.abs(z_score) > threshold:

            outliers.append(y)

    return outliers



price_outliers = detect_outlier(train_data['SalePrice'])

price_outliers
# Extreme value analyses

# Checking values more than 1.5 times from the first or third quartile



q1, q3 = np.percentile(train_data['SalePrice'],[25,75])

iqr = q3 - q1

lower_bound = q1 - (1.5 * iqr)

upper_bound = q3 + (1.5 * iqr)

print(f' Lower bound: {lower_bound} \n Upper bound: {upper_bound}')



# Removing abnormal prices

train_data_processed = train_data.copy()

for i in price_outliers:

    train_data_processed.drop(train_data_processed[train_data_processed['SalePrice'] == i].index, inplace=True)

# ID column is not needed for further analyses

train_data = train_data_processed.drop('Id', axis=1)





# updating the lists of columns used for analyses



#numeric_columns = train_data.dtypes[train_data.dtypes != "object"].index 

numeric_columns =  [col for col in train_data.columns if train_data.dtypes[col] != 'object'] 

categorical_columns = [col for col in train_data.columns if train_data.dtypes[col] == 'object']



# As shown above, the distribution is not normal. We can observe 'peakedness' and the house proces are positively skewed

# Applying log transformation ensures that errors in predicting expensive houses and cheap houses affect the result equally.



train_data['SalePrice'] = np.log(train_data['SalePrice'])
#transformed histogram and normal probability plot

sns.distplot(train_data['SalePrice'], fit=norm)
# Source: https://www.kaggle.com/apapiu/regularized-linear-models

# Log transformation of numeric features which are heavily skewed

# This will make the feature more normally distributed and linear regression perform better - since linear regression is sensitive to outliers. 



skewed_feats = train_data[numeric_columns].apply(lambda x: skew(x.dropna()))  #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75] 

skewed_feats = skewed_feats.index

train_data[skewed_feats] = np.log1p(train_data[skewed_feats])

test_data[skewed_feats] = np.log1p(test_data[skewed_feats])
# Creating and Training the Model



X_train = train_data.drop(['SalePrice'], axis=1)

y_train = train_data['SalePrice']

model_lasso = LassoCV(alphas = [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003]).fit(X_train, y_train)

coef = pd.Series(model_lasso.coef_, index = X_train.columns)

alpha = model_lasso.alpha_



print(f'The amount of penalization is {alpha} \n{sum(coef != 0)} features have been selected by Lasso. \n{sum(coef == 0)} variables have been eliminated.')

pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
predictions_train = model_lasso.predict(X_train)



print('MAE:', metrics.mean_absolute_error(y_train, predictions_train))

print('MSE:', metrics.mean_squared_error(y_train, predictions_train))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, predictions_train)))
df_pred = pd.DataFrame({"predicted_vals":predictions_train, "true_vals":y_train})

df_pred["residuals"] = df_pred["true_vals"] - df_pred["predicted_vals"]

df_pred.plot(x = "predicted_vals", y = "residuals", kind = "scatter")
X_test = test_data.drop(['Id', 'SalePrice'], axis=1)

y_test = test_data['SalePrice']



predictions = np.exp(model_lasso.predict(X_test))

plt.scatter(y_test,predictions)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#Create a  DataFrame with Ids and our price predictions

submission = pd.DataFrame({'Id':test_data['Id'],'SalePrice':predictions})



submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'House Prices Predictions - Lasso Regression.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)