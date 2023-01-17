import numpy as np 

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn import metrics

from sklearn.preprocessing import StandardScaler





from scipy import stats

from scipy.stats import norm, skew



import pandas_profiling
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
test_df.info()
#Looking at the correlations between the different features



corrmat = train_df.corr()

f, ax = plt.subplots(figsize=(15,15))

sns.heatmap(corrmat, square=True, vmax=1, vmin=-1, linewidths=1);

plt.show()
train_df.corr()['SalePrice'].sort_values(ascending=False)
#Checking the shape of the Train and Test dataframes



print("The train data size is : {} ".format(train_df.shape))

print("The test data size is : {} ".format(test_df.shape))
#Id field can be removed as it does not influence the prediction process, as stated by author of the Ames Dataset, D de Kock.

#Save the 'Id' column before dropping it as you will need it later for your submission



train_ID = train_df["Id"]

test_ID = test_df["Id"]



train_df.drop("Id", axis = 1, inplace = True)

test_df.drop("Id", axis = 1, inplace = True)
#Rechecking the shape of the Train and Test dataframes after dropping 'Id' columns



print("Train data size is : {} ".format(train_df.shape))

print("Test data size is : {} ".format(test_df.shape))
#As per the author's advice, we check for outliers, and delete them if any found.



fig, ax = plt.subplots()

ax.scatter(x = train_df['GrLivArea'], y = train_df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)
#From the below graph you can see that there is still 2 points that seem like outliers, but the author explains it as

#"two of them are simply unusual sales (very large houses priced relatively appropriately)."



fig, ax = plt.subplots()

ax.scatter(train_df['GrLivArea'], train_df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
#Remove the Target variable from the train dataset



y = train_df['SalePrice'].values 
#Drawing a distribution plot to show the spread of data



sns.distplot(train_df['SalePrice'] , fit=norm); 



# Get the fitted parameters used by the function



(mean, sd) = norm.fit(train_df['SalePrice'])

print( '\n Mean SalePrice = {:.2f} and sd Saleprice = {:.2f}\n'.format(mean, sd))



#From the plot we see that SalePrice is skewed to the right, and linear regression is all about normal distribution - 

#from Assumptions of Linear Regression



plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#QQ -Plot shows the Linearity of points distributed - from the plot we can see that the points are not in a 

#linear relation to the line.



fig = plt.figure()

res = stats.probplot(train_df['SalePrice'], plot=plt)

plt.show()
#log transform the target variable



train_df["SalePrice"] = np.log1p(train_df["SalePrice"])



#Kernel Density Plot



sns.distplot(train_df["SalePrice"],fit=norm);

plt.ylabel=('Frequency')

plt.title=('SalePrice distribution');



#Get the fitted parameters used by the function



(mean,sd)= norm.fit(train_df['SalePrice'].values);



#QQ plot - After log transformation the line and points are more linear



fig = plt.figure()

res = stats.probplot(train_df['SalePrice'], plot=plt)

plt.show()
new_train = train_df.shape[0] #create a new train variable to hold the 0 index shape value

new_test = test_df.shape[0]   #create a new test variable to to hold the 0 index shape value





#join the two datasets



combine_df = pd.concat((train_df, test_df)).reset_index(drop=True) 



#drop the SalePrice because we don't want to make any changes to it since its our target variable



combine_df.drop(['SalePrice'], axis=1, inplace=True)  

print("Combined data size is : {}".format(combine_df.shape))
combine_df_na = (combine_df.isnull().sum() / len(combine_df)) * 100

combine_df_na = combine_df_na.drop(combine_df_na[combine_df_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Percentage Missing' :combine_df_na})

missing_data.head(20)
#PoolQC : data description says NA means "No Pool". 



#That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general. 



combine_df["PoolQC"] = combine_df["PoolQC"].fillna("None")



#MiscFeature : data description says NA means "no misc feature"



combine_df["MiscFeature"] = combine_df["MiscFeature"].fillna("None")



#Alley : data description says NA means "no alley access"



combine_df["Alley"] = combine_df["Alley"].fillna("None")



#Fence : data description says NA means "no fence"



combine_df["Fence"] = combine_df["Fence"].fillna("None")



#FireplaceQu : data description says NA means "no fireplace"



combine_df["FireplaceQu"] = combine_df["FireplaceQu"].fillna("None")



#LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other 

#houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.



combine_df["LotFrontage"] = combine_df.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



#GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    combine_df[col] = combine_df[col].fillna('None')

    

#GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    combine_df[col] = combine_df[col].fillna(0)



#BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : 

#missing values are likely zero for having no basement



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    combine_df[col] = combine_df[col].fillna(0)

    

#BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : 

#For all these categorical basement-related features, NaN means that there is no basement.



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    combine_df[col] = combine_df[col].fillna('None')

    

#MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. 

#We can fill 0 for the area and None for the type. 



combine_df["MasVnrType"] = combine_df["MasVnrType"].fillna("None")

combine_df["MasVnrArea"] = combine_df["MasVnrArea"].fillna(0)



#MSZoning (The general zoning classification) : #'RL' the most common value. So we can fill in missing values with 'RL'



combine_df['MSZoning'] = combine_df['MSZoning'].fillna(combine_df['MSZoning'].mode()[0])



#Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house 

#with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can remove it.



combine_df = combine_df.drop(['Utilities'], axis=1)



#Functional : data description says NA means typical



combine_df["Functional"] = combine_df["Functional"].fillna("Typ")



#Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.



combine_df['Electrical'] = combine_df['Electrical'].fillna(combine_df['Electrical'].mode()[0])



#KitchenQual: Only one NA value, and same as Electrical, we set 'TA' 

#(which is the most frequent) for the missing value in KitchenQual.



combine_df['KitchenQual'] = combine_df['KitchenQual'].fillna(combine_df['KitchenQual'].mode()[0])



#Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. 

#We will just substitute in the most common string



combine_df['Exterior1st'] = combine_df['Exterior1st'].fillna(combine_df['Exterior1st'].mode()[0])

combine_df['Exterior2nd'] = combine_df['Exterior2nd'].fillna(combine_df['Exterior2nd'].mode()[0])



#SaleType : Fill in with most frequent which is "WD"



combine_df['SaleType'] = combine_df['SaleType'].fillna(combine_df['SaleType'].mode()[0])



#MSSubClass : Na most likely means No building class. We can replace missing values with None



combine_df['MSSubClass'] = combine_df['MSSubClass'].fillna("None")
#Checking missing percentages after filling missing values/columns



combine_df_na = (combine_df.isnull().sum() / len(combine_df)) * 100

combine_df_na = combine_df_na.drop(combine_df_na[combine_df_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Percentage Missing' :combine_df_na})

missing_data.head(100)
#Transforming the Numerical values that are actually Categorical values



#MSSubClass = The building class



combine_df['MSSubClass'] = combine_df['MSSubClass'].astype(str)





#Changing OverallCond/OverallQual into a categorical variable



combine_df['OverallCond'] = combine_df['OverallCond'].astype(str)

combine_df['OverallQual'] = combine_df['OverallQual'].astype(str)



#Year and month sold are transformed into categorical features.



combine_df['YrSold'] = combine_df['YrSold'].astype(str)

combine_df['MoSold'] = combine_df['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



# process columns, apply LabelEncoder to categorical features



for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(combine_df[c].values)) 

    combine_df[c] = lbl.transform(list(combine_df[c].values))



# shape        



print('Shape combine_df: {}'.format(combine_df.shape))
#Since the area of the house is looked to as a singular unit rather that different parts, 

#I add one more feature which is the total area of basement, first and second floor areas of each house



combine_df['TotalSF'] = combine_df['TotalBsmtSF'] + combine_df['1stFlrSF'] + combine_df['2ndFlrSF']
combine_dummies = pd.get_dummies(combine_df) #Convert categorical variable into dummy/indicator variables



print(combine_df.shape)
result = combine_dummies.values
#Standardize features by removing the mean and scaling to unit variance



scaler = StandardScaler() 

result = scaler.fit_transform(result)
X = result[:new_train] #items from the beginning through stop-1



test_values = result[new_train:] #items start through the rest of the array
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)



#Increased Alpha gives better r^2 score, but it zeros coeficients

#https://chrisalbon.com/machine_learning/linear_regression/effect_of_alpha_on_lasso_regression/



lasso = Lasso(alpha=0.05) 



lasso.fit(X_train, y_train)



y_pred = lasso.predict(X_test)



y_train_pred = lasso.predict(X_train)
from sklearn.metrics import r2_score



print("Train accuracy: " , r2_score(y_train, y_train_pred))

print("Test accuracy: ", r2_score(y_test, y_pred))
final_prices = lasso.predict(test_values)

final_prices
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

print('MSE:', metrics.mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
plt.scatter(y_test,y_pred)
print(y_pred.mean())



print(y_pred.std())
linear_train = LinearRegression()

linear_train.fit(X_train, y_train)

linear_pred = linear_train.predict(X_test)
linear_train = LinearRegression()

linear_train.fit(X_train, y_train)

linear_pred = linear_train.predict(X_test)
plt.scatter(y_test,linear_pred)
print(linear_pred.mean())



print(linear_pred.std())
ridge_train = Ridge(alpha = 1600)

ridge_train.fit(X_train, y_train)

ridge_pred = ridge_train.predict(X_test)
print("R2:Train accuracy: " , r2_score(y_test, ridge_pred))

print('MSE:', metrics.mean_squared_error(y_test, ridge_pred))
plt.scatter(y_test,ridge_pred)
print(ridge_pred.mean())



print(ridge_pred.std())
print('Testing MSE')

print('Linear:', metrics.mean_squared_error(y_test, linear_pred))

print('Ridge :', metrics.mean_squared_error(y_test, ridge_pred))

print('Lasso:', metrics.mean_squared_error(y_test, y_pred))
print('Testing R2')

print("R2:Linear: " , r2_score(y_test, linear_pred))

print("R2:Ridge: " , r2_score(y_test, ridge_pred))

print("R2:Lasso: " , r2_score(y_test, y_pred))
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = final_prices

sub.to_csv('submission.csv',index=False)