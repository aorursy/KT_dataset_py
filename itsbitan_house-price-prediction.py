# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing the libraries for ploting

import matplotlib.pyplot as plt

import seaborn as sns
#Import the datasets

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
#Lets check the dataset

df_train.info()

df_test.info()
#At first we check our target variable

sns.distplot(df_train['SalePrice'])

print('Skewness: %f', df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#We see that the target variable SalePrice has a right-skewed distribution. We need to log transform this variable

#so that it becomes normally distributed. A normally distributed (or close to normal) target variable helps in better

#modeling the relationship between target and independent variables. In addition,linear algorithms assume constant

#variance in the error term. 

df_train['SalePrice_Log'] = np.log(df_train['SalePrice'])

print('Skewness: %f', df_train['SalePrice_Log'].skew())

print("Kurtosis: %f" % df_train['SalePrice_Log'].kurt())

sns.distplot(df_train['SalePrice_Log'], color ='blue')
#Lets drop the  SalePrice column

df_train.drop('SalePrice', axis =1, inplace = True)
#Now seperate the dataset into numurical and categorical variable

num_train = df_train.select_dtypes(include = [np.number])

cat_train = df_train.select_dtypes(exclude = [np.number])
#Lets check the numurical variables first

num_train.describe()
#Lets check the correlation and heat map

corr = num_train.corr()

colormap = sns.diverging_palette(220, 10, as_cmap = True)

plt.figure(figsize = (20,16))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            annot=True,fmt='.2f',linewidths=0.30,

            cmap = colormap, linecolor='white')

plt.title('Correlation of df Features', y = 1.05, size=15)
#Lets look the correlation score

print (corr['SalePrice_Log'].sort_values(ascending=False), '\n')
#Lets check OverallQual in datits

df_train['OverallQual'].unique()  #We clearly see the variable well scaled 1 to 10
#Now we check the relation with SalePrice_Log variable

sns.jointplot(x=df_train['OverallQual'], y=df_train['SalePrice_Log'], color = 'deeppink')
#Now Check the GrLivArea variable is also highly correlated with SalePrice_Log

sns.jointplot(x=df_train['GrLivArea'], y=df_train['SalePrice_Log'], color = 'green')
# Now remove outliers

df_train.drop(df_train[df_train['GrLivArea'] > 4000].index, inplace=True)

df_train.shape 
#Lets check the missing value of train set

df_train.isnull().sum()

#Now check the null value of test set

df_test.isnull().sum()
#We can see almost same columns of both train and test set have missing values. So we join the datasets

all_data = df_train.append(df_test)

all_data.shape
#Lets clean the dataset. Note that we don't need 'ID' to predict the value of the house. Lets drop the columns

all_data.drop('Id', axis =1, inplace = True)
#Lets ckeck again the missing value

all_data.isnull().sum()
##Now seperate the dataset into numurical and categorical variable

num_data = all_data.select_dtypes(include = [np.number])

cat_data = all_data.select_dtypes(exclude = [np.number])
#Lets deal with the missing data of numurical variables first

num_data.isnull().sum()
#We note the LotFrontage has almost 20% missing value, we replace the nan value with median

all_data['LotFrontage'] = all_data['LotFrontage'].fillna(all_data['LotFrontage'].median())

#We also replace GarageYrBlt, GarageArea and GarageCars with '0', (Since No garage = no cars in such in garage)

all_data["GarageYrBlt"] = all_data["GarageYrBlt"].fillna(0)

all_data["GarageArea"] = all_data["GarageArea"].fillna(0)

all_data['GarageCars'] = all_data['GarageCars'].fillna(0)
#BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath replace with '0',

#(as '0' means no basement)

all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(0) 

all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(0) 

all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(0) 

all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(0)

all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(0) 

all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(0)   

#Now deal with MasVnrArea 

all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
#Now our all missing value of the numrical variable imputed.

#Lets deal with missing value of the catagorical variable

cat_data.isnull().sum()
#We note that Alley, PoolQC, Fence, MiscFeature have over 90% missing value both train & test set.

#lets drop these the columns

all_data.drop('Alley', axis = 1, inplace = True)

all_data.drop('PoolQC', axis = 1, inplace = True)

all_data.drop('Fence', axis = 1, inplace = True)

all_data.drop('MiscFeature', axis = 1, inplace = True)
#FireplaceQu : data description says NA means "no fireplace"

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
#GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None

for col in ('GarageType','GarageFinish','GarageQual','GarageCond'):

    all_data[col] = all_data[col].fillna('None')


#BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related

#features, NaN means that there is no basement.

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')
#We apply same process for other catagorical variable

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
#MSZoning (The general zoning classification):'RL' is by far the most common value. So we can fill in 

#missing values with mode value

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
#Same process apply for 'Electrical', 'KitchenQual', 'Exterior1st',Exterior2nd', 'SaleType'variable

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['Functional'] = all_data['Functional'].fillna(all_data['Functional']).mode()[0]   
#Utilities : For this categorical feature all records are "AllPub",except for one "NoSeWa" and 2 NA . 

#Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. 

#We can then safely remove it.

all_data = all_data.drop(['Utilities'], axis=1)
#Now lets check the whether any missing value remaining or not

all_data.isnull().sum()
#Now lets encode the catagorical variable

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
all_data['BldgType'] = labelencoder.fit_transform(all_data['BldgType'])

all_data['BsmtCond'] = labelencoder.fit_transform(all_data['BsmtCond'])

all_data['BsmtExposure'] = labelencoder.fit_transform(all_data['BsmtExposure'])

all_data['BsmtFinType1'] = labelencoder.fit_transform(all_data['BsmtFinType1'])

all_data['BsmtFinType2'] = labelencoder.fit_transform(all_data['BsmtFinType2'])

all_data['BsmtQual'] = labelencoder.fit_transform(all_data['BsmtQual'])

all_data['CentralAir'] = labelencoder.fit_transform(all_data['CentralAir'])

all_data['Condition1'] = labelencoder.fit_transform(all_data['Condition1'])

all_data['Condition2'] = labelencoder.fit_transform(all_data['Condition2'])

all_data['Electrical'] = labelencoder.fit_transform(all_data['Electrical'])

all_data['ExterCond'] = labelencoder.fit_transform(all_data['ExterCond'])

all_data['ExterQual'] = labelencoder.fit_transform(all_data['ExterQual'])

all_data['Exterior1st'] = labelencoder.fit_transform(all_data['Exterior1st'])

all_data['Exterior2nd'] = labelencoder.fit_transform(all_data['Exterior2nd'])

all_data['FireplaceQu'] = labelencoder.fit_transform(all_data['FireplaceQu'])

all_data['Foundation'] = labelencoder.fit_transform(all_data['Foundation'])

all_data['Functional'] = labelencoder.fit_transform(all_data['Functional'])

all_data['GarageCond'] = labelencoder.fit_transform(all_data['GarageCond'])

all_data['GarageFinish'] = labelencoder.fit_transform(all_data['GarageFinish'])

all_data['GarageQual'] = labelencoder.fit_transform(all_data['GarageQual'])

all_data['GarageType'] = labelencoder.fit_transform(all_data['GarageType'])

all_data['Heating'] = labelencoder.fit_transform(all_data['Heating'])

all_data['HeatingQC'] = labelencoder.fit_transform(all_data['HeatingQC'])

all_data['HouseStyle'] = labelencoder.fit_transform(all_data['HouseStyle'])

all_data['KitchenQual'] = labelencoder.fit_transform(all_data['KitchenQual'])

all_data['LandContour'] = labelencoder.fit_transform(all_data['LandContour'])

all_data['LandSlope'] = labelencoder.fit_transform(all_data['LandSlope'])

all_data['LotConfig'] = labelencoder.fit_transform(all_data['LotConfig'])

all_data['LotShape'] = labelencoder.fit_transform(all_data['LotShape'])

all_data['MSZoning'] = labelencoder.fit_transform(all_data['MSZoning'])

all_data['MasVnrType'] = labelencoder.fit_transform(all_data['MasVnrType'])

all_data['Neighborhood'] = labelencoder.fit_transform(all_data['Neighborhood'])

all_data['PavedDrive'] = labelencoder.fit_transform(all_data['PavedDrive'])

all_data['RoofMatl'] = labelencoder.fit_transform(all_data['RoofMatl'])

all_data['RoofStyle'] = labelencoder.fit_transform(all_data['RoofStyle'])

all_data['SaleCondition'] = labelencoder.fit_transform(all_data['SaleCondition'])

all_data['SaleType'] = labelencoder.fit_transform(all_data['SaleType'])

all_data['Street'] = labelencoder.fit_transform(all_data['Street'])

#get numeric features

numeric_features = [f for f in all_data.columns if all_data[f].dtype != object]
#transform the numeric features using log(x + 1)

from scipy.stats import skew

skewed = all_data[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))

skewed = skewed[skewed > 0.75]

skewed = skewed.index

all_data[skewed] = np.log1p(all_data[skewed])
#Lets check the correlation and heat map

corr = all_data.corr()

colormap = sns.diverging_palette(220, 10, as_cmap = True)

plt.figure(figsize = (20,16))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            annot=True,fmt='.2f',linewidths=0.30,

            cmap = colormap, linecolor='white')

plt.title('Correlation of df Features', y = 1.05, size=15)
#Lets look the correlation score

print (corr['SalePrice_Log'].sort_values(ascending=False), '\n')
#Now lets check top five correlated features. First we check Overall Qual relation with SalePrice_Log variable

sns.jointplot(x=all_data['OverallQual'], y=all_data['SalePrice_Log'], color = 'deeppink')
#Now Check the GrLivArea

sns.jointplot(x=all_data['GrLivArea'], y=all_data['SalePrice_Log'], color = 'green')

#Now Check the GarageCars

sns.jointplot(x=all_data['GarageCars'], y=all_data['SalePrice_Log'], color = 'red')
#Now Check the GarageArea

sns.jointplot(x=all_data['GarageArea'], y=all_data['SalePrice_Log'], color = 'skyblue')
#Now Check the TotalBsmtSF

sns.jointplot(x=all_data['TotalBsmtSF'], y=all_data['SalePrice_Log'], color = 'purple')
#Note that there are some features nagetively corelated with SalePrice_Log.Lets remove the all features which are

#nagetively correlated with SalePrice_Log

all_data = all_data.drop(['Functional'], axis=1)

all_data = all_data.drop(['ExterQual'], axis=1)

all_data = all_data.drop(['BsmtQual'], axis=1)

all_data = all_data.drop(['KitchenQual'], axis=1)

all_data = all_data.drop(['GarageType'], axis=1)

all_data = all_data.drop(['HeatingQC'], axis=1)

all_data = all_data.drop(['GarageFinish'], axis=1)

all_data = all_data.drop(['LotShape'], axis=1)

all_data = all_data.drop(['EnclosedPorch'], axis=1)

all_data = all_data.drop(['MSZoning'], axis=1)

all_data = all_data.drop(['KitchenAbvGr'], axis=1)

all_data = all_data.drop(['Heating'], axis=1)

all_data = all_data.drop(['BsmtFinType1'], axis=1)

all_data = all_data.drop(['BldgType'], axis=1)

all_data = all_data.drop(['MiscVal'], axis=1)

all_data = all_data.drop(['LotConfig'], axis=1)

all_data = all_data.drop(['LowQualFinSF'], axis=1)

all_data = all_data.drop(['FireplaceQu'], axis=1)

all_data = all_data.drop(['SaleType'], axis=1)

all_data = all_data.drop(['OverallCond'], axis=1)

all_data = all_data.drop(['YrSold'], axis=1)

all_data = all_data.drop(['BsmtFinSF2'], axis=1)

all_data = all_data.drop(['MSSubClass'], axis=1)

all_data = all_data.drop(['BsmtHalfBath'], axis=1)

all_data = all_data.drop(['MasVnrType'], axis=1)

all_data = all_data.drop(['BsmtExposure'], axis=1)

all_data.shape
#create new data

train_new = all_data[all_data['SalePrice_Log'].notnull()]

test_new = all_data[all_data['SalePrice_Log'].isnull()]

test_new = test_new.drop(['SalePrice_Log'], axis = 1)

train_new.shape

test_new.shape
#Now creat ML model for our dataset

#First creat our matrix of features and target varible

x = train_new.drop(['SalePrice_Log'], axis = 1)

x_train = x.iloc[:,:].values

y_train= train_new.iloc[:, 41].values

x_test = test_new.iloc[:,:].values
#Lets use the Lasso regression

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso

lasso = Lasso()
# list of alphas to tune

params = {'alpha': [0.001, 0.002, 0.003, 0.005, 0.006, 0.007, 0.008, 0.009]}

# cross validation

grid_search = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = 10, 

                        return_train_score=True,

                        verbose = 1)            

grid_search.fit(x_train, y_train) 
#checking the value of optimum number of parameters

print(grid_search.best_params_)

print(grid_search.best_score_)

cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=1000]

cv_results
#Fitting Lasso Regression to the tranning set

lasso = Lasso(alpha = 0.001, max_iter = 1000)

lasso.fit(x_train, y_train)

#Lets check the accuracy 

accuracy_train = lasso.score(x_train, y_train)

print(accuracy_train)
##Now predicting the test set result

y_pred = lasso.predict(x_test)
#convert back from logarithmic values to SalePrice

y_pred =np.expm1(y_pred) 
#Sumbmission the result

sub = pd.DataFrame()

sub['Id'] = df_test['Id']

sub['SalePrice'] = y_pred

sub.to_csv('submission.csv', index=False)