# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import statsmodels

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor



from sklearn.feature_selection import RFE



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn import metrics



# hide warnings

import warnings

warnings.filterwarnings('ignore')
# read the data

housing = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

housing.head()
#checking shape of dataset

housing.shape
#checking info of dataset

housing.info()
import pandas_profiling

#profile_report = pandas_profiling.ProfileReport(housing)

#profile_report
numeric_data = housing.select_dtypes(include = ['float64','int64'])

numeric_data.head()
#checking percentage of null values in each column



df_missing=pd.DataFrame((round(100*(housing.isnull().sum()/len(housing.index)), 2)), columns=['missing'])

df_missing.sort_values(by=['missing'], ascending=False).head(20)
#reading data dictionary



f = open("/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt", "r")

print(f.read())
#addressing NaN values based on data dictionary



# In column 'PoolQC' (Pool quality), NaN stands for No Pool

housing['PoolQC'] = housing['PoolQC'].fillna('No_Pool')



# In column 'MiscFeature' (Miscellaneous Features), NaN stands for None, meaning the house has no miscellaneous features.

housing['MiscFeature'] = housing['MiscFeature'].fillna('None')



# In column 'Alley', NaN stands for No Alley Access as per the data dictionary

housing['Alley'] = housing['Alley'].fillna('No_Alley_Access')



# In column 'Fence' (Fence Quality), NaN stands for No Fence as per the data dictionary

housing['Fence'] = housing['Fence'].fillna('No_Fence')



# In column 'FireplaceQu' (FireplaceQu Quality), NaN stands for No Fireplace as per the data dictionary

housing['FireplaceQu'] = housing['FireplaceQu'].fillna('No_Fireplace')



# LotFrontage stands for Linear feet of street connected to property, there is no explanation to impute this in data dictionary

# Let's consider imputing it with median of the lotFrontage of houses in the same neighbourhood



# Group data by neighborhood and impute missing value with median LotFrontage of all the neighborhood

housing["LotFrontage"] = housing.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



# In column 'GarageYrBlt' (Gargae Year Built), NaN stands for houses with no garage, let's impute with 0

housing['GarageYrBlt'] = housing['GarageYrBlt'].fillna(0)



# 'GarageType', 'GarageFinish', 'GarageQual' (Garage Quality) and 'GarageCond'(Garage Condition)

# Missing values signify no garage as per data dictionary.let's impute NaN values here with No Garage

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    housing[col] = housing[col].fillna('No_Garage')



#BsmtFinType1, BsmtFinType2 (Rating of basement finished area),

#BsmtExposure (Basement Exposure), BsmtQual(Basement Quality), BsmtCond (Basement Conidtion)

#These are all parameter related to basement. A NaN value probably signifies that the house does not have a basement.

for col in ('BsmtFinType1', 'BsmtFinType2', 'BsmtExposure', 'BsmtQual','BsmtCond'):

    housing[col] = housing[col].fillna('No_Basement')

    

# In column 'MasVnrType' (Masonry veneer type), let's impute it with mode "None"

housing['MasVnrType'] = housing['MasVnrType'].fillna('None')



# In column 'MasVnrType' (Masonry veneer type), let's impute it with mode 0 corresponding to None

housing['MasVnrArea'] = housing['MasVnrArea'].fillna(0)



# In column 'Electrical' (Electrical system), let's impute NaN with "Other"

housing['Electrical'] = housing['Electrical'].fillna("Other")
#checking percentage of null values in each column



df_missing=pd.DataFrame((round(100*(housing.isnull().sum()/len(housing.index)), 2)), columns=['missing'])

df_missing.sort_values(by=['missing'], ascending=False).head(20)
# Let us first visualize the spread of Target Variable 'Sale Price'

from scipy.stats import norm

sns.distplot(housing['SalePrice'], fit=norm)

plt.show()
## "MSSubClass" is a numeric column but it should actually be categorical as per the data dictionary, so let's convert that.



housing=housing.replace({'MSSubClass' : { 20 : '1-STORY 1946 & NEWER ALL STYLES', 

                                          30:'1-STORY 1945 & OLDER',

                                          40:'1-STORY W/FINISHED ATTIC ALL AGES',

                                          45:'1-1/2 STORY - UNFINISHED ALL AGES',

                                          50:'1-1/2 STORY FINISHED ALL AGES',

                                          60:'2-STORY 1946 & NEWER',

                                          70:'2-STORY 1945 & OLDER',

                                          75:'2-1/2 STORY ALL AGES',

                                          80:'SPLIT OR MULTI-LEVEL',

                                          85:'SPLIT FOYER',

                                          90:'DUPLEX - ALL STYLES AND AGES',

                                         120:'1-STORY PUD (Planned Unit Development) - 1946 & NEWER',

                                         150:'1-1/2 STORY PUD - ALL AGES',

                                         160:'2-STORY PUD - 1946 & NEWER',

                                         180:'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',

                                         190:'2 FAMILY CONVERSION - ALL STYLES AND AGES'}})
numeric_data = housing.select_dtypes(include = ['float64','int64'])

numeric_data.columns
#function to plot scatter plot numeric variables with price



def pp(w,x,y,z):

    sns.pairplot(housing, x_vars=[w,x,y,z], y_vars='SalePrice',height=4, aspect=1, kind='scatter')

    plt.show()



pp('LotFrontage', 'LotArea', 'OverallQual','OverallCond')

pp('YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'GrLivArea')

pp('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF')
#function to plot scatter plot numeric variables with price



def pp(w,x,y,z):

    sns.pairplot(housing, x_vars=[w,x,y,z], y_vars='SalePrice',height=4, aspect=1, kind='scatter')

    plt.show()



pp('1stFlrSF', '2ndFlrSF','LowQualFinSF','MSSubClass')

pp('BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath')

pp('BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd','Fireplaces')
#function to plot scatter plot numeric variables with price



def pp(w,x,y,z):

    sns.pairplot(housing, x_vars=[w,x,y,z], y_vars='SalePrice',height=4, aspect=1, kind='scatter')

    plt.show()



pp('GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF')

pp('OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch')

pp('PoolArea','MiscVal', 'MoSold', 'YrSold')
# label encode ordinal features where there is order in categories



housing = housing.replace({  "Alley":        {"No_Alley_Access" : 0, "Grvl" : 1, "Pave" : 2},

                       "BsmtCond":     {"No_Basement" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "BsmtExposure": {"No_Basement" : 0, "No" : 2, "Mn" : 2, "Av": 3, 

                                        "Gd" : 4},

                       "BsmtFinType1": {"No_Basement" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, 

                                        "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},

                       "BsmtFinType2": {"No_Basement" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, 

                                        "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtQual":     {"No_Basement" : 0, "Po" : 1, "Fa" : 2, "TA": 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "CentralAir":   {"None" : 0, "N" : 1, "Y" : 2},

                       "ExterCond":    {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, 

                                        "Gd": 4, "Ex" : 5},

                       "ExterQual":    {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, 

                                        "Gd": 4, "Ex" : 5},

                       "Fence":        {"No_Fence" : 0, "MnWw" : 1, "GdWo" : 2, "MnPrv": 3, 

                                        "GdPrv" : 4},

                       "FireplaceQu":  {"No_Fireplace" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "Functional":   {"None" : 0, "Sal" : 1, "Sev" : 2, "Maj2" : 3, 

                                        "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, 

                                        "Typ" : 8},

                       "GarageCond":   {"No_Garage" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "GarageQual":   {"No_Garage" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "GarageFinish": {"No_Garage" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3},

                       "HeatingQC":    {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "KitchenQual":  {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "LandContour":  {"None" : 0, "Low" : 1, "HLS" : 2, "Bnk" : 3, 

                                        "Lvl" : 4},

                       "LandSlope":    {"None" : 0, "Sev" : 1, "Mod" : 2, "Gtl" : 3},

                       "LotShape":     {"None" : 0, "IR3" : 1, "IR2" : 2, "IR1" : 3, 

                                        "Reg" : 4},

                       "PavedDrive":   {"None" : 0, "N" : 0, "P" : 1, "Y" : 2},

                       "PoolQC":       {"No_Pool" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, 

                                        "Ex" : 4},

                       "Street":       {"None" : 0, "Grvl" : 1, "Pave" : 2},

                       "Utilities":    {"None" : 0, "ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, 

                                        "AllPub" : 4}}

                     )



housing.BsmtCond = housing.BsmtCond.astype(int)
## FUNCTION TO PLOT CHARTS



def plot_charts(var1,var2,label_rotation):

    plt.figure(figsize=(12, 10))   

    plt.subplot(2,2,1)

    plt.title('Count Plot of '+ var1)

    plt1=sns.countplot(housing[var1], palette=("husl"))

    plt1.set(xlabel = '%s'%var1, ylabel='Count of'+ '%s'%var1)

    if(label_rotation):

        plt1.set_xticklabels(plt1.get_xticklabels(),rotation=90)

        

    plt.subplot(2,2,2)

    plt.title(var1+' vs Price')

    plt2=sns.boxplot(x=housing[var1], y=housing.SalePrice, palette=("husl"))

    if(label_rotation):

        plt2.set_xticklabels(plt2.get_xticklabels(),rotation=90)

    

    plt.subplot(2,2,3)

    plt.title('Count Plot of '+ var2)

    plt3=sns.countplot(housing[var2], palette=("husl"))

    plt3.set(xlabel = '%s'%var2, ylabel='Count of'+ '%s'%var2)

    if(label_rotation):

        plt3.set_xticklabels(plt3.get_xticklabels(),rotation=90)

    

    plt.subplot(2,2,4)

    plt.title(var2+' vs Price')

    plt4=sns.boxplot(x=housing[var2], y=housing.SalePrice, palette=("husl"))

    if(label_rotation):

        plt4.set_xticklabels(plt4.get_xticklabels(),rotation=90)

        

    plt.show()
categorical_features=housing.select_dtypes(include='object')

categorical_features.columns
plot_charts('MSZoning', 'Street', label_rotation=False)
housing['Street'].value_counts(dropna=False)
plot_charts('LotShape','Alley',label_rotation=False)
housing['Street'].value_counts(dropna=False)
plot_charts('LandContour','LotConfig',label_rotation=False)
plot_charts('LandSlope','BldgType',label_rotation=False)
plot_charts('RoofStyle', 'RoofMatl',label_rotation=True)
plot_charts('SaleType', 'SaleCondition',label_rotation=False)
# DATA PREPARATION
#changing months to categorical

import calendar



housing['MonthSold'] = housing['MoSold'].apply(lambda x: calendar.month_name[x])

housing=housing.drop(['MoSold'], axis=1)
#changing data type of Gararge yr built to int from float

housing['GarageYrBlt'] = housing['GarageYrBlt'].astype(int)
#DERIVED VARIABLES which might make more sense than year



housing['Age'] = housing['YrSold'] - housing['YearBuilt']

housing['Remod_Age'] = housing['YrSold'] - housing['YearRemodAdd']

housing['Garage_Age'] = housing['YrSold'] - housing['GarageYrBlt']

housing.drop(['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold'],1, inplace = True)
numeric_data = housing.select_dtypes(include = ['float64','int64'])

numeric_data.columns
#OUTLIER TREATMENT



def remove_outliers(dtf, numl_list):

    for j in numl_list:

        Q1 = dtf[j].quantile(0.05)

        Q3 = dtf[j].quantile(0.95)

        IQR = Q3 - Q1       

        dtf = dtf[(dtf[j] >= Q1-2.5*IQR) & (dtf[j] <= Q3+2.5*IQR)]

    return dtf
numeric_data_list=list(numeric_data.columns)
housing=remove_outliers(housing,numeric_data_list)
# Outlier treatment on the variable Sale Price

plt.figure(figsize=(4,3))

plt.boxplot(housing['SalePrice'])

plt.show()
housing.shape
cor = numeric_data.corr()

plt.figure(figsize=(20,20))



sns.heatmap(cor, annot=True)

plt.show()
# we drop Id (not relevant)

corr = housing.drop(["Id"], axis=1).select_dtypes(include="number").corr()



plt.figure(figsize=(16,16));

corr["SalePrice"].sort_values(ascending=True)[:-1].plot(kind="barh")

plt.title("Correlation of numerical features to SalePrice")

plt.xlabel("Correlation to SalePrice")

plt.tight_layout()

plt.show()
#saleprice correlation matrix

plt.figure(figsize=(7,7))

k = 15 #number of variables for heatmap

cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(housing[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, cmap='viridis',yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#converting binary variables to numeric by mapping to 0 and 1



housing['Street'] = housing['Street'].apply(lambda x: 1 if x == 'Pave' else 0 )

housing['CentralAir'] = housing['CentralAir'].apply(lambda x : 1 if x == 'Y' else 0)
#converting binary variables to numeric by mapping to 0 and 1



housing['PavedDrive'] = housing['PavedDrive'].apply(lambda x : 1 if x == 'Y' else 0)
df = housing.drop(['Id'],axis=1)

housing_categorical = df.select_dtypes(include=['object'])

housing_categorical.head()
# convert into dummies

housing_dummies = pd.get_dummies(housing_categorical, drop_first=True)

housing_dummies.head()
#dropping original categorical columns

df = df.drop(list(housing_categorical.columns), axis=1)
#concatenating dummy columns to original dataframe

df = pd.concat([df,housing_dummies], axis=1)
df.shape
#train_test_split

df_train,df_test=train_test_split(df,train_size=0.70, random_state=100)

df_train.shape
y_train = np.log(df_train.SalePrice)

X_train = df_train.drop("SalePrice",1)



y_test= np.log(df_test.SalePrice)

X_test = df_test.drop("SalePrice",1)
num_vars=X_train.select_dtypes(include=['int64','float64']).columns
num_vars
scaler = StandardScaler()

X_train[num_vars] = scaler.fit_transform(X_train[num_vars])

X_test[num_vars] = scaler.transform(X_test[num_vars])
#linear regression model
lm=LinearRegression()

lm.fit(X_train,y_train)



rfe = RFE(lm,20)

rfe=rfe.fit(X_train,y_train)



col=X_train.columns[rfe.support_]

col
X_train_new=X_train[col]



X_train_new = sm.add_constant(X_train_new)



#create first model

lr=sm.OLS(y_train,X_train_new)



#fit the model

lr_model=lr.fit()



lr_model.summary()
#RIDGE REGULARIZATION



# list of alphas to tune





params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100]}





ridge = Ridge()



# cross validation

folds = 5

model_cv = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

model_cv.fit(X_train, y_train) 
print(model_cv.best_params_)

print(model_cv.best_score_)
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=100]

cv_results
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

plt.figure(figsize=(16,5))



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper right')

plt.show()
#final ridge model

alpha = 10

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
#lets predict the R-squared value of test and train data

y_train_pred = ridge.predict(X_train)

print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = ridge.predict(X_test)

print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
from sklearn.metrics import mean_squared_error

print ('RMSE is: \n', mean_squared_error(y_test, y_test_pred))
# Ridge model parameters

model_parameters_1 = list(ridge.coef_)

model_parameters_1.insert(0, ridge.intercept_)

model_parameters_1 = [round(x, 3) for x in model_parameters_1]

cols = X_train.columns

cols = cols.insert(0, "constant")

list(zip(cols, model_parameters_1))
#lasso

params = {'alpha': [0.00005, 0.0001, 0.001, 0.008, 0.01]}

lasso = Lasso()



# cross validation

model_cv_l = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model_cv_l.fit(X_train, y_train)
# cv results

cv_results_l = pd.DataFrame(model_cv_l.cv_results_)
#checking the value of optimum number of parameters

print(model_cv_l.best_params_)

print(model_cv_l.best_score_)
#final lasso model

alpha = 0.001



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train, y_train) 
#lets predict the R-squared value of test and train data

y_train_pred = lasso.predict(X_train)

print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
#lets predict the R-squared value of test and train data

y_test_pred = lasso.predict(X_test)

print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
from sklearn.metrics import mean_squared_error

print ('RMSE is: \n', mean_squared_error(y_test, y_test_pred))
# Lasso model parameters

model_parameters_1 = list(lasso.coef_)

model_parameters_1.insert(0, lasso.intercept_)

model_parameters_1 = [round(x, 3) for x in model_parameters_1]

cols = X_train.columns

cols = cols.insert(0, "constant")

list(zip(cols, model_parameters_1))
test_data=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test_data.head()
df_missing=pd.DataFrame((round(100*(test_data.isnull().sum()/len(test_data.index)), 2)), columns=['missing'])

df_missing.sort_values(by=['missing'], ascending=False).head(20)
#addressing NaN values based on data dictionary



# In column 'PoolQC' (Pool quality), NaN stands for No Pool

test_data['PoolQC'] = test_data['PoolQC'].fillna('No_Pool')



# In column 'MiscFeature' (Miscellaneous Features), NaN stands for None, meaning the house has no miscellaneous features.

test_data['MiscFeature'] = test_data['MiscFeature'].fillna('None')



# In column 'Alley', NaN stands for No Alley Access as per the data dictionary

test_data['Alley'] = test_data['Alley'].fillna('No_Alley_Access')



# In column 'Fence' (Fence Quality), NaN stands for No Fence as per the data dictionary

test_data['Fence'] = test_data['Fence'].fillna('No_Fence')



# In column 'FireplaceQu' (FireplaceQu Quality), NaN stands for No Fireplace as per the data dictionary

test_data['FireplaceQu'] = test_data['FireplaceQu'].fillna('No_Fireplace')



# LotFrontage stands for Linear feet of street connected to property, there is no explanation to impute this in data dictionary

# Let's consider imputing it with median of the lotFrontage of houses in the same neighbourhood



# Group data by neighborhood and impute missing value with median LotFrontage of all the neighborhood

test_data["LotFrontage"] = test_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



# In column 'GarageYrBlt' (Gargae Year Built), NaN stands for houses with no garage, let's impute with 0

test_data['GarageYrBlt'] = test_data['GarageYrBlt'].fillna(0)



# 'GarageType', 'GarageFinish', 'GarageQual' (Garage Quality) and 'GarageCond'(Garage Condition)

# Missing values signify no garage as per data dictionary.let's impute NaN values here with No Garage

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    test_data[col] = test_data[col].fillna('No_Garage')



#BsmtFinType1, BsmtFinType2 (Rating of basement finished area),

#BsmtExposure (Basement Exposure), BsmtQual(Basement Quality), BsmtCond (Basement Conidtion)

#These are all parameter related to basement. A NaN value probably signifies that the house does not have a basement.

for col in ('BsmtFinType1', 'BsmtFinType2', 'BsmtExposure', 'BsmtQual','BsmtCond'):

    test_data[col] = test_data[col].fillna('No_Basement')

    

# In column 'MasVnrType' (Masonry veneer type), let's impute it with mode "None"

test_data['MasVnrType'] = test_data['MasVnrType'].fillna('None')



# In column 'MasVnrType' (Masonry veneer type), let's impute it with mode 0 corresponding to None

test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(0)



# In column 'Electrical' (Electrical system), let's impute NaN with "Other"

test_data['Electrical'] = test_data['Electrical'].fillna("Other")
df_missing=pd.DataFrame((round(100*(test_data.isnull().sum()/len(test_data.index)), 2)), columns=['missing'])

df_missing.sort_values(by=['missing'], ascending=False).head(20)
catgl_feats = test_data.dtypes[test_data.dtypes == 'object'].index

numrl_feats = test_data.dtypes[test_data.dtypes != 'object'].index
Nan_cols = []



cols = test_data.columns

for i in cols:

    if (test_data[i].isnull().sum()/len(df))*100 > 0:

        Nan_cols.append(i)
cat_treat_list = []

num_treat_list = []



for i in Nan_cols:

    if i in catgl_feats:

        cat_treat_list.append(i)

    else:

        num_treat_list.append(i)
cat_treat_list
num_treat_list
for i in cat_treat_list:

    test_data[i].fillna(test_data[i].mode()[0], inplace = True)
for i in num_treat_list:

    test_data[i].fillna(0, inplace = True)
df_missing=pd.DataFrame((round(100*(test_data.isnull().sum()/len(test_data.index)), 2)), columns=['missing'])

df_missing.sort_values(by=['missing'], ascending=False).head(20)
# label encode ordinal features where there is order in categories



test_data = test_data.replace({  "Alley":        {"No_Alley_Access" : 0, "Grvl" : 1, "Pave" : 2},

                       "BsmtCond":     {"No_Basement" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "BsmtExposure": {"No_Basement" : 0, "No" : 2, "Mn" : 2, "Av": 3, 

                                        "Gd" : 4},

                       "BsmtFinType1": {"No_Basement" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, 

                                        "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},

                       "BsmtFinType2": {"No_Basement" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, 

                                        "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtQual":     {"No_Basement" : 0, "Po" : 1, "Fa" : 2, "TA": 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "CentralAir":   {"None" : 0, "N" : 1, "Y" : 2},

                       "ExterCond":    {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, 

                                        "Gd": 4, "Ex" : 5},

                       "ExterQual":    {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, 

                                        "Gd": 4, "Ex" : 5},

                       "Fence":        {"No_Fence" : 0, "MnWw" : 1, "GdWo" : 2, "MnPrv": 3, 

                                        "GdPrv" : 4},

                       "FireplaceQu":  {"No_Fireplace" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "Functional":   {"None" : 0, "Sal" : 1, "Sev" : 2, "Maj2" : 3, 

                                        "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, 

                                        "Typ" : 8},

                       "GarageCond":   {"No_Garage" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "GarageQual":   {"No_Garage" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "GarageFinish": {"No_Garage" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3},

                       "HeatingQC":    {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "KitchenQual":  {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "LandContour":  {"None" : 0, "Low" : 1, "HLS" : 2, "Bnk" : 3, 

                                        "Lvl" : 4},

                       "LandSlope":    {"None" : 0, "Sev" : 1, "Mod" : 2, "Gtl" : 3},

                       "LotShape":     {"None" : 0, "IR3" : 1, "IR2" : 2, "IR1" : 3, 

                                        "Reg" : 4},

                       "PavedDrive":   {"None" : 0, "N" : 0, "P" : 1, "Y" : 2},

                       "PoolQC":       {"No_Pool" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, 

                                        "Ex" : 4},

                       "Street":       {"None" : 0, "Grvl" : 1, "Pave" : 2},

                       "Utilities":    {"None" : 0, "ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, 

                                        "AllPub" : 4}}

                     )



test_data.BsmtCond = test_data.BsmtCond.astype(int)
test_data.GarageQual = test_data.GarageQual.astype(int)
## "MSSubClass" is a numeric column but it should actually be categorical as per the data dictionary, so let's convert that.



test_data=test_data.replace({'MSSubClass' : { 20 : '1-STORY 1946 & NEWER ALL STYLES', 

                                          30:'1-STORY 1945 & OLDER',

                                          40:'1-STORY W/FINISHED ATTIC ALL AGES',

                                          45:'1-1/2 STORY - UNFINISHED ALL AGES',

                                          50:'1-1/2 STORY FINISHED ALL AGES',

                                          60:'2-STORY 1946 & NEWER',

                                          70:'2-STORY 1945 & OLDER',

                                          75:'2-1/2 STORY ALL AGES',

                                          80:'SPLIT OR MULTI-LEVEL',

                                          85:'SPLIT FOYER',

                                          90:'DUPLEX - ALL STYLES AND AGES',

                                         120:'1-STORY PUD (Planned Unit Development) - 1946 & NEWER',

                                         150:'1-1/2 STORY PUD - ALL AGES',

                                         160:'2-STORY PUD - 1946 & NEWER',

                                         180:'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',

                                         190:'2 FAMILY CONVERSION - ALL STYLES AND AGES'}})
#changing months to categorical

import calendar



test_data['MonthSold'] = test_data['MoSold'].apply(lambda x: calendar.month_name[x])

test_data=test_data.drop(['MoSold'], axis=1)
#changing data type of Gararge yr built to int from float

test_data['GarageYrBlt'] = test_data['GarageYrBlt'].astype(int)
#DERIVED VARIABLES which might make more sense than year



test_data['Age'] = test_data['YrSold'] - test_data['YearBuilt']

test_data['Remod_Age'] = test_data['YrSold'] - test_data['YearRemodAdd']

test_data['Garage_Age'] = test_data['YrSold'] - test_data['GarageYrBlt']

test_data.drop(['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold'],1, inplace = True)
#converting binary variables to numeric by mapping to 0 and 1



test_data['Street'] = test_data['Street'].apply(lambda x: 1 if x == 'Pave' else 0 )

test_data['CentralAir'] = test_data['CentralAir'].apply(lambda x : 1 if x == 'Y' else 0)
#converting binary variables to numeric by mapping to 0 and 1



test_data['PavedDrive'] = test_data['PavedDrive'].apply(lambda x : 1 if x == 'Y' else 0)
test_data_X=test_data.drop('Id',1)
test_data_numerical=test_data_X.select_dtypes(include=['int32','int64','float64']).columns
test_data_numerical
test_data_categorical = test_data_X.select_dtypes(include=['object'])

test_data_categorical.head()
# convert into dummies

test_data_dummies = pd.get_dummies(test_data_categorical, drop_first=True)

test_data_dummies.head()
#dropping original categorical columns

df_test = test_data_X.drop(list(test_data_categorical.columns), axis=1)
#concatenating dummy columns to original dataframe

df = pd.concat([df_test,test_data_dummies], axis=1)
df[test_data_numerical].shape
df[test_data_numerical] = scaler.transform(df[test_data_numerical])
#lets predict the R-squared value of test and train data

y_test_predicted = lasso.predict(df)
y_test_predicted
final_predictions = np.exp(y_test_predicted)
final_predictions
salespriceprediction= pd.DataFrame({'Id': test_data['Id'] ,'SalePrice': final_predictions })
salespriceprediction.to_csv("salespriceprediction.csv",index=False)