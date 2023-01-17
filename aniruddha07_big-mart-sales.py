import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pylab 
import scipy.stats as stats
from sklearn.model_selection import cross_validate, cross_val_score
#read Train file, examine the Train data set
train = pd.read_csv("../input/train/Train.csv", header=0)
print(train.shape)
train.head()

#checking the most recent Outlet Establishment

print(train['Outlet_Establishment_Year'].max())
print("\n")

# 2013 data and thus taking only the difference of the outlet establishment year into account

train.Outlet_Establishment_Year = 2013-train.Outlet_Establishment_Year

train['Outlet_Establishment_Year'].value_counts()
#read Test file, examine the Test data set
test = pd.read_csv("../input/testpr-a102/Test.csv", header=0)
test.shape
test.head()
#checking the most recent Outlet Establishment

print(test['Outlet_Establishment_Year'].max())

# 2013 data and thus taking only the difference of the outlet establishment year into account

test.Outlet_Establishment_Year = 2013-test.Outlet_Establishment_Year

test.Outlet_Establishment_Year.value_counts()

#checking for nulls NaN in Training Set
train.isnull().head()
train.isnull().sum()
train.describe(include = 'all')
#checking for nulls NaN in Test Set
test.isnull().head()
test.isnull().sum()
test.describe(include = 'all')
#print null values of the train set

train[train.Item_Weight.isnull()]
train[train.Outlet_Size.isnull()]
item_weights_train = train['Item_Weight'].value_counts()
outlet_size_train = train['Outlet_Size'].value_counts()
train['Item_Weight'].value_counts(dropna = False)
train['Outlet_Size'].value_counts(dropna = False)
train.groupby('Outlet_Type').Outlet_Size.value_counts(dropna = False)
#print null values of the test set

test[test.Item_Weight.isnull()]
test[test.Outlet_Size.isnull()]
item_weights_test = test['Item_Weight'].value_counts()
outlet_size_test = test['Outlet_Size'].value_counts()
test['Item_Weight'].value_counts(dropna = False)
test['Outlet_Size'].value_counts(dropna = False)
#imputing using various methods i.e. mean, median, mode, kNN, etc. and creating new data frames for them same method for both the data sets

#mean imputing for Item_Weight and using Small for Outlet_Size all throughout

#changing both Test and Train datasets for Outlet_size

train['Outlet_Size'].fillna(value = 'Small',inplace = True)
test['Outlet_Size'].fillna(value = 'Small',inplace = True)

train.head()
test.head()
#check for same type occuring more than once

train.groupby('Item_Type').Item_Fat_Content.value_counts(dropna = False)

#lf, LF, Low Fat, low fat all are the same, make Low Fat, reg and Regular are the same make Regular
#imputing mean for Item_Weight 

train_1 = train.dropna(subset = ['Item_Weight'])
train_mean = train.fillna(value = train_1["Item_Weight"].mean())

test_1 = test.dropna(subset = ['Item_Weight'])
test_mean = test.fillna(value = test_1["Item_Weight"].mean())
train_mean.head()
test_mean.head()
#imputing median for Item_Weight

train_2 = train.dropna(subset = ['Item_Weight'])
train_median = train.fillna(value = train_2["Item_Weight"].median())

test_2 = test.dropna(subset = ['Item_Weight'])
test_median = train.fillna(value = train_2["Item_Weight"].median())
train_median.head()
test_median.head()
#imputing mode for Item_Weight

train_3 = train.dropna(subset = ['Item_Weight'])
train_mode = train.fillna(value = train_3["Item_Weight"].mode())

test_3 = test.dropna(subset = ['Item_Weight'])
test_mode = test.fillna(value = test_3["Item_Weight"].mode())
train_mode.head()
test_mode.head()
#histogrtam before and after imputing with mean for train and test sets

#mean

#before imputing
%matplotlib inline                            
train.Item_Weight.plot(kind = 'hist', color = 'white', edgecolor = 'black', facecolor = 'blue', figsize = (12,6), title = 'Item Weight Histogram')

#after imputing

%matplotlib inline                           
train_mean.Item_Weight.plot(kind = 'hist', color = 'white', edgecolor = 'black', facecolor = 'blue', figsize = (12,6), title = 'Item Weight Histogram')

#before imputing test set

%matplotlib inline                           
test.Item_Weight.plot(kind = 'hist', color = 'white', edgecolor = 'black', facecolor = 'blue', figsize = (12,6), title = 'Item Weight Histogram')

#after imputing test set

%matplotlib inline                           
test_mean.Item_Weight.plot(kind = 'hist', color = 'white', edgecolor = 'black', facecolor = 'blue', figsize = (12,6), title = 'Item Weight Histogram')

#with mode imputation train set

#after imputing

%matplotlib inline                           
train_mode.Item_Weight.plot(kind = 'hist', color = 'white', edgecolor = 'black', facecolor = 'blue', figsize = (12,6), title = 'Item Weight Histogram')

# with median train set

#after imputing

%matplotlib inline                           
train_median.Item_Weight.plot(kind = 'hist', color = 'white', edgecolor = 'black', facecolor = 'blue', figsize = (12,6), title = 'Item Weight Histogram')

#with mode imputation test set

#after imputing

%matplotlib inline                           
test_mode.Item_Weight.plot(kind = 'hist', color = 'white', edgecolor = 'black', facecolor = 'blue', figsize = (12,6), title = 'Item Weight Histogram')

#with median test set

#after imputing

%matplotlib inline                           
test_median.Item_Weight.plot(kind = 'hist', color = 'white', edgecolor = 'black', facecolor = 'blue', figsize = (12,6), title = 'Item Weight Histogram')

# combining train and test dataframes

test['Item_Outlet_Sales'] = np.nan
test.shape
combined = train.append(test)
train.shape
test.shape
combined.shape
combined.loc[combined.Item_Fat_Content.isin(['LF', 'low fat']),'Item_Fat_Content'] = 'Low Fat'
combined.loc[combined.Item_Fat_Content.isin(['reg']),'Item_Fat_Content'] = 'Regular'

combined.Item_Fat_Content.value_counts(dropna = False)
#categories to which fat content doesnot apply are relabelled as none

combined.loc[combined.Item_Type.isin(['Health and Hygiene', 'Household', 'Others']),'Item_Fat_Content'] = 'None'
combined.Item_Fat_Content.value_counts()
combined.Outlet_Size.value_counts(dropna = False)
# density plot of MRP

import seaborn as sns

sns.kdeplot(data=combined.Item_MRP, bw=.2)
# as shown above, the density plot indicates a wide variation and hence it is better to
# classify the MRP into categories

combined['MRP_Factor'] = pd.cut(combined.Item_MRP, [0,70,130,201,400], labels=['Low', 'Medium', 'High', 'Very High'])
combined.drop('Item_MRP', axis=1, inplace=True)
combined.MRP_Factor.value_counts()
combined.Item_Visibility.value_counts(dropna = False)
combined.Item_Visibility.isnull().sum()
# checking minimum values in each feild

combined.min()
# check if the min value is still zero

combined.Item_Visibility.describe()

# min value observed is non-zero
#looking for the data types

combined.dtypes
# checking for outliers

combined.boxplot("Item_Weight")
combined.boxplot("Outlet_Establishment_Year")
combined.boxplot("Item_Visibility")

# there are many outliers and hence look at the histogram
combined.Item_Visibility.plot(kind='hist')


# the histogram is skewed and the no. of outliers is also large and hence cannot 
# remove outliers
#looking at the categorial variables

combined.head()
combined.isnull().sum()
# seen above is that item visibility is 0 for some of the Items which doesnot mean anything 
# for our case and hence we will group them by Item Identifier, take the mean of each group
# and use that for each of the groups

visibility_mean = combined.loc[combined.Item_Visibility != 0].groupby('Item_Identifier').Item_Visibility.mean()
for index, row in combined.iterrows():
    if(row.Item_Visibility == 0):
        combined.loc[index, 'Item_Visibility'] = visibility_mean[row.Item_Identifier]
combined.head()
combined_1 = combined.iloc[:,:12].values
combined_1
# encoding the categorical variables

for i in [0,2,4,5,7,8,9,11]:
    combined_1[:,i] = LabelEncoder().fit_transform(combined_1[:,i])+1
combined_1
test.shape
train.shape
train_1 = combined_1[0:8523,:12]
train_1.shape
test_1 = combined_1[8523:14204,:12]
test_1.shape
type(train_1)
type(test_1)
train_df = pd.DataFrame(data=train_1, columns = ['Item_Identifier', 'Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Outlet_Sales','MRP_Factor'])
train_df.head()
test_df = pd.DataFrame(data=test_1, columns = ['Item_Identifier', 'Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Outlet_Sales','MRP_Factor'])
test_df.head()
train_train = train_df[pd.notnull(train_df['Item_Weight'])]
train_train.isnull().sum()
train_train = train_train.convert_objects(convert_numeric=True)
train_test = train_df[pd.isnull(train_df['Item_Weight'])]
train_test.isnull().sum()
train_train.head()
train_test.head()
test_train = test_df[pd.notnull(test_df['Item_Weight'])]
test_train.isnull().sum()
test_train = test_train.convert_objects(convert_numeric=True)
test_test = test_df[pd.isnull(test_df['Item_Weight'])]
test_test.isnull().sum()
# checking the assumptions of linear regression 
# Q-Q plots

quantile = train_train.Item_Weight

stats.probplot(quantile, dist="uniform", plot=pylab)
pylab.show()
from seaborn import residplot
residplot('Item_Identifier', 'Item_Weight', train_train)
residplot('Item_Fat_Content', 'Item_Weight', train_train)
residplot('Item_Visibility', 'Item_Weight', train_train)
residplot('Item_Type', 'Item_Weight', train_train)
residplot('Outlet_Identifier', 'Item_Weight', train_train)
residplot('Outlet_Type', 'Item_Weight', train_train)
residplot('Outlet_Location_Type', 'Item_Weight', train_train)
residplot('Outlet_Size', 'Item_Weight', train_train)
residplot('Outlet_Establishment_Year', 'Item_Weight', train_train)
residplot('MRP_Factor', 'Item_Weight', train_train)
residplot('Item_Identifier', 'Item_Weight', test_train)
residplot('Item_Fat_Content', 'Item_Weight', test_train)
residplot('Item_Visibility', 'Item_Weight', test_train)
residplot('Item_Type', 'Item_Weight', test_train)
residplot('Outlet_Identifier', 'Item_Weight', test_train)
residplot('Outlet_Type', 'Item_Weight', test_train)
residplot('Outlet_Location_Type', 'Item_Weight', test_train)
residplot('Outlet_Size', 'Item_Weight', test_train)
residplot('Outlet_Establishment_Year', 'Item_Weight', test_train)
residplot('MRP_Factor', 'Item_Weight', test_train)
cols = [i for i in train_train.columns if i not in['Item_Weight','Item_Outlet_Sales']]
cols
from sklearn.metrics import mean_squared_error as mse
# imputing weight using linear regression, then using ridge and lasso regression, not using 
# sales as a feature for both the training and test sets

from sklearn.linear_model import LinearRegression
fit_train = LinearRegression(normalize = True).fit(train_train.loc[:,cols],train_train.loc[:,train_train.columns=='Item_Weight'])
train_test1 = train_test
train_test1['Item_Weight'] = fit_train.predict(train_test1.loc[:,cols])
train_test1.head()
train_linreg = train_train.append(train_test1)
train_linreg.tail()
# RMSE 
predict1 = fit_train.predict(train_train.loc[:,cols])
np.sqrt(mse(train_train.Item_Weight, predict1))
# cross validation 

cv_score = cross_val_score(LinearRegression(normalize = True), train_train.loc[:,cols], train_train['Item_Weight'], cv=10, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print ("CV Score : Mean = %.6g | Std = %.6g | Min = %.6g | Max = %.6g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
train_test2 = train_test
from sklearn.linear_model import Ridge
fit_train_ridge = Ridge(alpha=0.05,normalize=True).fit(train_train.loc[:,cols],train_train.loc[:,train_train.columns=='Item_Weight'])
train_test2['Item_Weight'] = fit_train_ridge.predict(train_test2.loc[:,cols])
train_test2.head()
train_ridgereg = train_train.append(train_test2)
train_ridgereg.tail()
# RMSE 
predict2 = fit_train_ridge.predict(train_train.loc[:,cols])
np.sqrt(mse(train_train.Item_Weight, predict2))
# cross validation 

cv_score = cross_val_score(Ridge(alpha=0.05,normalize=True), train_train.loc[:,cols], train_train['Item_Weight'], cv=10, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print ("CV Score : Mean = %.6g | Std = %.6g | Min = %.6g | Max = %.6g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
train_test3 = train_test
from sklearn.linear_model import Lasso
fit_train_lasso = Lasso(alpha = 0.5, normalize = True).fit(train_train.loc[:,cols],train_train.loc[:,train_train.columns=='Item_Weight'])
train_test3['Item_Weight'] = fit_train_lasso.predict(train_test3.loc[:,cols])
train_test3.head()
train_lassoreg = train_train.append(train_test3)
train_lassoreg.tail()
# RMSE 
predict3 = fit_train_lasso.predict(train_train.loc[:,cols])
np.sqrt(mse(train_train.Item_Weight, predict3))
cv_score = cross_val_score(Lasso(alpha = 0.5, normalize = True), train_train.loc[:,cols], train_train['Item_Weight'], cv=10, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print ("CV Score : Mean = %.6g | Std = %.6g | Min = %.6g | Max = %.6g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
test_test1 = test_test
test_test2 = test_test
test_test3 = test_test
fit_test = LinearRegression().fit(test_train.loc[:,cols],test_train.loc[:,test_train.columns=='Item_Weight'])
test_test1['Item_Weight'] = fit_train.predict(test_test1.loc[:,cols])
test_test1.head()
test_linreg = test_train.append(train_test1)
test_linreg.tail()
# RMSE 
predict_t1 = fit_train.predict(train_train.loc[:,cols])
np.sqrt(mse(train_train.Item_Weight, predict_t1))
# cross validation 

cv_score = cross_val_score(LinearRegression(normalize = True), test_train.loc[:,cols], test_train['Item_Weight'], cv=10, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print ("CV Score : Mean = %.6g | Std = %.6g | Min = %.6g | Max = %.6g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
fit_test_ridge = Ridge(alpha=0.05,normalize=True).fit(test_train.loc[:,cols],test_train.loc[:,test_train.columns=='Item_Weight'])
test_test2['Item_Weight'] = fit_test_ridge.predict(test_test2.loc[:,cols])
train_test2.head()
test_ridgereg = test_train.append(train_test2)
test_ridgereg.tail()

predict_t2 = fit_test_ridge.predict(test_train.loc[:,cols])
np.sqrt(mse(test_train.Item_Weight, predict_t2))
cv_score = cross_val_score(Ridge(alpha=0.05,normalize=True), test_train.loc[:,cols], test_train['Item_Weight'], cv=10, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print ("CV Score : Mean = %.6g | Std = %.6g | Min = %.6g | Max = %.6g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
fit_test_lasso = Lasso(alpha = 0.5, normalize = True).fit(test_train.loc[:,cols],test_train.loc[:,test_train.columns=='Item_Weight'])
test_test3['Item_Weight'] = fit_test_lasso.predict(test_test3.loc[:,cols])
test_test3.head()
test_lassoreg = test_train.append(train_test3)
test_lassoreg.tail()
# RMSE 
predict_t3 = fit_test_lasso.predict(test_train.loc[:,cols])
np.sqrt(mse(test_train.Item_Weight, predict_t3))
cv_score = cross_val_score(Lasso(alpha = 0.5, normalize = True), test_train.loc[:,cols], test_train['Item_Weight'], cv=10, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print ("CV Score : Mean = %.6g | Std = %.6g | Min = %.6g | Max = %.6g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
# RMSE is observed to be the least in the case of simple linear regression and we go by that

trainf = train_linreg.convert_objects(convert_numeric=True)
testf = test_linreg.convert_objects(convert_numeric=True)
# checking the assumptions of linear regression 
# Q-Q plots

quantile = trainf.Item_Outlet_Sales

stats.probplot(quantile, dist="uniform", plot=pylab)
pylab.show()
residplot('Item_Identifier', 'Item_Outlet_Sales', trainf)
residplot('Item_Weight', 'Item_Outlet_Sales', trainf)
residplot('Item_Fat_Content', 'Item_Outlet_Sales', trainf)
residplot('Item_Visibility', 'Item_Outlet_Sales', trainf)
residplot('Item_Type', 'Item_Outlet_Sales', trainf)
residplot('Outlet_Identifier', 'Item_Outlet_Sales', trainf)
residplot('Outlet_Establishment_Year', 'Item_Outlet_Sales', trainf)
residplot('Outlet_Size', 'Item_Outlet_Sales', trainf)
residplot('Outlet_Type', 'Item_Outlet_Sales', trainf)
residplot('Outlet_Location_Type', 'Item_Outlet_Sales', trainf)
residplot('MRP_Factor', 'Item_Outlet_Sales', trainf)
# using mean as the predictor

Out_Sales = trainf.Item_Outlet_Sales.mean()
predict_s = [Out_Sales for i in range(0,8523)]

#RMSE
np.sqrt(mse(trainf.Item_Outlet_Sales, predict_s))

# The error is quite huge and hence we need to use predictive modelling

o_var = 'Item_Outlet_Sales'

def regmodel(reg_model, train, test, features, o_var):
    
    # Fit the regression model on the data
    
    reg_model.fit(train[features], train[o_var])
        
    # Predicting on training set
    
    prediction_train = reg_model.predict(train[features])

    # Performing cross validation, 10fold cross validation is used
    
    cv_score = cross_val_score(reg_model, train[features], train[o_var], cv=10, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    # Print model report
    
    print ("\nRegression Model Report")
    print ("RMSE : %.4g" % np.sqrt(mse(train[o_var].values, prediction_train)))
    print ("CV Score : Mean = %.6g | Std = %.6g | Min = %.6g | Max = %.6g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    # Predict on testing data:
    
    test[o_var] = reg_model.predict(test[features])
    
    
# Using Linear Regression 

features = [x for x in trainf.columns if x not in [o_var]]

linreg = LinearRegression(normalize=True)
regmodel(linreg, trainf, testf, features, o_var)

coef_linreg = pd.Series(linreg.coef_, features).sort_values()
coef_linreg.plot(kind='bar', title='Linear Regression Coefficients', figsize=(10,6))
# QQ plot

stats.probplot(coef_linreg, dist="uniform", plot=pylab)
pylab.show()
# Using Ridge Regression Model:

ridgereg = Ridge(alpha=0.05,normalize=True)
regmodel(ridgereg, trainf, testf, features, o_var)
coef_ridgereg = pd.Series(ridgereg.coef_, features).sort_values()
coef_ridgereg.plot(kind='bar', title='Ridge Regression Coefficients', figsize=(10,6))
stats.probplot(coef_ridgereg, dist="norm", plot=pylab)
pylab.show()
# Using Lasso Regression Model:
lassoreg = Lasso(alpha=0.05,normalize=True)
regmodel(lassoreg, trainf, testf, features, o_var)
coef_lassoreg = pd.Series(lassoreg.coef_, features).sort_values()
coef_lassoreg.plot(kind='bar', title='Lasso Regression Coefficients', figsize=(10,6))

stats.probplot(coef_lassoreg, dist="norm", plot=pylab)
pylab.show()