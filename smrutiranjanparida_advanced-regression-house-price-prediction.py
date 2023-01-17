# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Libraries for model building
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Reading the train and Test data Set

# Reading the data
house_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
house_df .head()
# finding the shape of the dataset
house_df.shape
house_df.info()
# converting the data type for catagorical columns which are wrongly classified as numeric columns
column = ["YrSold","MSSubClass","MoSold","OverallQual","OverallCond"]
for col in column:
    house_df[col]= house_df[col].astype('object')
    test_data[col]= test_data[col].astype('object')
# Checking the missing data
missing_values = pd.DataFrame(round((house_df.isnull().sum()/len(house_df)*100),2))
missing_values.columns=["Missing_Value%"]
missing_values = missing_values[(missing_values["Missing_Value%"])>0].sort_values("Missing_Value%",ascending =False)
missing_values
# Dropping the columns were the % of NA values are more than 90%
house_df = house_df.drop(["Alley","PoolQC","Fence","MiscFeature"], axis =1)
# Dropping the column ID as this is the ID for the property and can not be used for model building
house_df = house_df.drop("Id", axis =1)
# As NA is used to represent none values creating replacing NA with  a different catagory
house_df['FireplaceQu'] = house_df['FireplaceQu'].fillna("NoFire")

# treatment of Testdata
test_data = test_data.drop(["Alley","PoolQC","Fence","MiscFeature"], axis =1)
test_data['FireplaceQu'] = test_data['FireplaceQu'].fillna("NoFire")
# Imputing the missing values with the median
house_df["LotFrontage"]=house_df["LotFrontage"].fillna(house_df["LotFrontage"].median())

# Imputing the missing values for garage built with Yearbuilt
house_df["GarageYrBlt"]  = house_df["GarageYrBlt"].fillna(house_df["YearBuilt"])
# Imputing wrongly labeled values as missing values NA means not available
for items in ['GarageQual','GarageFinish','GarageType','GarageCond']:
    house_df[items] = house_df[items].fillna("Nogarage")

for items in ["BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2"]:
    house_df[items] = house_df[items].fillna("NoBsmt")
    
# treatment of Testdata
test_data ["LotFrontage"]=test_data["LotFrontage"].fillna(test_data["LotFrontage"].median())
test_data["GarageYrBlt"]  = test_data["GarageYrBlt"].fillna(test_data["YearBuilt"])
for items in ['GarageQual','GarageFinish','GarageType','GarageCond']:
    test_data[items] = test_data[items].fillna("Nogarage")
for items in ["BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2"]:
    test_data[items] = test_data[items].fillna("NoBsmt")
# Drop the data containing null values.
house_df = house_df.dropna(axis =0)
# adding new features as age of the building 
house_df['Age_building'] = house_df['YearBuilt'].max() - house_df['YearBuilt']
house_df['Age_Remodel'] = house_df['YearRemodAdd'].max() - house_df['YearRemodAdd']
house_df['Age_Garage'] = house_df['GarageYrBlt'].max() - house_df['GarageYrBlt']
house_df = house_df.drop(['YearBuilt','YearRemodAdd','GarageYrBlt'],axis =1)
house_df['TotalArea'] = house_df['1stFlrSF']+house_df['2ndFlrSF']+house_df['TotalBsmtSF']
house_df['TotalBath'] = house_df['BsmtFullBath']+0.5*house_df['BsmtHalfBath']+house_df['FullBath']+0.5*house_df['HalfBath']

# Creating the features in test_data
test_data['Age_building'] = test_data['YearBuilt'].max() - test_data['YearBuilt']
test_data['Age_Remodel'] = test_data['YearRemodAdd'].max() - test_data['YearRemodAdd']
test_data['Age_Garage'] = test_data['GarageYrBlt'].max() - test_data['GarageYrBlt']
test_data = test_data.drop(['YearBuilt','YearRemodAdd','GarageYrBlt'],axis =1)
test_data['TotalArea'] = test_data['1stFlrSF']+test_data['2ndFlrSF']+test_data['TotalBsmtSF']
test_data['TotalBath'] = test_data['BsmtFullBath']+0.5*test_data['BsmtHalfBath']+test_data['FullBath']+0.5*test_data['HalfBath']

# Finding the outliers in data set 
columns  = list(house_df.columns)
num_col = [col for col in columns if house_df[col].dtype!='object']
house_df[num_col].quantile([0.05,0.25,0.75,0.9,0.95]) 
# dropping columns which has >90 % of data with sme value
house_df.drop(["LowQualFinSF","BsmtHalfBath","KitchenAbvGr","3SsnPorch","ScreenPorch","PoolArea","MiscVal"],axis =1, 
              inplace =True)
test_data.drop(["LowQualFinSF","BsmtHalfBath","KitchenAbvGr","3SsnPorch","ScreenPorch","PoolArea","MiscVal"],axis =1, 
              inplace =True)
# Finding the outliers in data set 
columns  = list(house_df.columns)
num_col = [col for col in columns if house_df[col].dtype!='object']
for item in num_col:
    Q1 = house_df[item].quantile(0.05)
    Q3 = house_df[item].quantile(0.95)
    IQR = Q3-Q1
    house_df = house_df[(house_df[item]>(Q1-1.5*IQR)) & (house_df[item]<(Q3+1.5*IQR))]
# converting the the data which has very high inequality
house_df=house_df.drop(['Utilities','Street', 'Condition2', 'RoofMatl','GarageCond','GarageQual','Functional',
                        'Heating','BsmtFinType2','LandSlope','LandContour'],axis =1)
test_data=test_data.drop(['Utilities','Street', 'Condition2', 'RoofMatl','GarageCond','GarageQual','Functional',
                        'Heating','BsmtFinType2','LandSlope','LandContour'],axis =1)

# Understanding the target variable
plt.figure(figsize = (8,8))
sns.distplot(house_df["SalePrice"])
# Log transformation of the target column for normalizing 
house_df["SalePrice"] = np.log1p(house_df["SalePrice"])
# Visualization of the Numeric columns
def num_plot(item):
    plt.figure(figsize = (10,4))
    plt.subplot(1,3,1)
    plt.boxplot(house_df[item])
    plt.title(item)
    plt.subplot(1,3,2)
    plt.hist(house_df[item])
    plt.title(item)
    plt.subplot(1,3,3)
    plt.scatter(x=item, y = 'SalePrice', data = house_df)
    plt.tight_layout()  
# Visualization of the catagorical columns    
def cat_visual(item):
    plt.figure(figsize = (10,4))
    plt.subplot(1,2,1)
    plt.title(item)
    sns.boxplot(x = item, y ="SalePrice", data = house_df)
    plt.xticks(rotation =45)
    data = pd.DataFrame(round(house_df[item].value_counts()/len(house_df[item])*100,2)).reset_index()
    data.columns = ["Cat","Perc"]
    plt.subplot(1,2,2)
    plt.title(item)
    plt.xticks(rotation =45)
    sns.barplot(x="Cat", y ="Perc",data = data)
    plt.tight_layout()
for items in ["LotFrontage","LotArea","TotalArea","TotalBath","GrLivArea","GarageArea"]:
    num_plot(items)
# Visualization for catagorical columns
for items in ["OverallQual","ExterQual","BsmtQual"]:
    cat_visual(items)
# checking the correlation matrix 
plt.figure(figsize = (10,10))
sns.heatmap(house_df.corr())
# removing features that have very high correlation
house_df = house_df.drop(["GarageCars", "Age_Garage","Age_Remodel"],axis =1)
test_data = test_data.drop(["GarageCars", "Age_Garage","Age_Remodel"],axis =1)
# removing column yr sold as theere is no significnt chsnge in price
house_df=house_df.drop('YrSold',axis =1)
test_data = test_data.drop('YrSold', axis =1)
#Converting the values to new values by combining with very small data to infer
house_df['Condition1'] = house_df['Condition1'].replace(['RRAe', 'RRNe', 'RRNn','PosA'], 'Other_cond')
house_df['RoofStyle'] = house_df['RoofStyle'].replace(['Flat','Gambrel','Mansard','Shed'], 'Other_roof')
house_df['Exterior1st'] = house_df['Exterior1st'].replace(['CBlock','AsphShn','ImStucc','BrkComm','Stone','Stucco','AsbShng',
                                                           'WdShing'], 'Other_ext')
house_df['Exterior2nd'] = house_df['Exterior2nd'].replace(['AsbShng','AsphShn','BrkComm','BrkFace','CBlock','ImStucc',
                                                           'Plywood','PreCast','Stone','Stucco','WdShing'],'Other')
house_df['SaleType'] = house_df['SaleType'].replace(['CWD','VWD','Con','ConLw','ConLI','ConLD'], 'Oth')
house_df['SaleCondition'] = house_df['SaleCondition'].replace(['AdjLand','Alloca','Family'], 'Other_cond')
house_df['GarageType'] = house_df['GarageType'].replace(['2Types','Basment','CarPort','NA'], 'Other_type')
house_df['HouseStyle'] = house_df['HouseStyle'].replace(['1.5Unf','2.5Unf','2.5Fin'], 'Other')
house_df['LotShape'] = house_df['LotShape'].replace(['IR2','IR3'], 'IR2_3')
house_df['MSZoning'] = house_df['MSZoning'].replace(['C (all)','RH'], 'Other')
house_df['Electrical'] = house_df['Electrical'].replace(['FuseF','FuseF','Mix'], 'Other')
house_df['OverallCond'] =house_df['OverallCond'].replace ({'1':'<4','2':'<4','3':'<4','4':'4-5','5':'4:5','6':'6-8',
                                                          '7':'6-8','8':'6-8'})
house_df['OverallQual'] =house_df['OverallQual'].replace ({'1':'<=3','2':'<=3','3':'<=3'})

# converting testdata

test_data['Condition1'] = test_data['Condition1'].replace(['RRAe', 'RRNe', 'RRNn','PosA'], 'Other_cond')
test_data['RoofStyle'] = test_data['RoofStyle'].replace(['Flat','Gambrel','Mansard','Shed'], 'Other_roof')
test_data['Exterior1st'] = test_data['Exterior1st'].replace(['CBlock','AsphShn','ImStucc','BrkComm','Stone','Stucco','AsbShng',
                                                           'WdShing'], 'Other_ext')
test_data['Exterior2nd'] = test_data['Exterior2nd'].replace(['AsbShng','AsphShn','BrkComm','BrkFace','CBlock','ImStucc',
                                                           'Plywood','PreCast','Stone','Stucco','WdShing'],'Other')
test_data['SaleType'] = test_data['SaleType'].replace(['CWD','VWD','Con','ConLw','ConLI','ConLD'], 'Oth')
test_data['SaleCondition'] = test_data['SaleCondition'].replace(['AdjLand','Alloca','Family'], 'Other_cond')
test_data['GarageType'] = test_data['GarageType'].replace(['2Types','Basment','CarPort','NA'], 'Other_type')
test_data['HouseStyle'] = test_data['HouseStyle'].replace(['1.5Unf','2.5Unf','2.5Fin'], 'Other')
test_data['LotShape'] = test_data['LotShape'].replace(['IR2','IR3'], 'IR2_3')
test_data['MSZoning'] = test_data['MSZoning'].replace(['C (all)','RH'], 'Other')
test_data['Electrical'] = test_data['Electrical'].replace(['FuseF','FuseF','Mix'], 'Other')
test_data['OverallCond'] =test_data['OverallCond'].replace ({'1':'<4','2':'<4','3':'<4','4':'4-5','5':'4:5','6':'6-8',
                                                          '7':'6-8','8':'6-8'})
test_data['OverallQual'] =test_data['OverallQual'].replace ({'1':'<=3','2':'<=3','3':'<=3'})
test_data.info()

# Creating Dummy variables for the Catagorical variables
cat_col  = [col for col in house_df.columns if house_df[col].dtype=='object']
# get dummy variables for catagorical values
cat = pd.get_dummies(house_df[cat_col],drop_first = True)
cat_test =pd.get_dummies(test_data[cat_col],drop_first = True)
house_price = house_df.drop(cat_col, axis =1)
house_price = pd.concat([house_price,cat],axis=1)

# Dataset for model Building
test_data = test_data.drop(cat_col, axis =1)
test_data = pd.concat([test_data,cat_test],axis=1)
test_data.head()
# scaling the features
from sklearn.preprocessing import StandardScaler
# Instantiate scaler object
columns =['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea',
          'BsmtFullBath','FullBath','HalfBath','BedroomAbvGr','TotRmsAbvGrd','Fireplaces','GarageArea','WoodDeckSF','OpenPorchSF',
          'EnclosedPorch','Age_building','TotalArea','TotalBath']

scalerX = StandardScaler()
scalery = StandardScaler()
X_train = house_price.drop("SalePrice",axis =1)
y_train  = house_price[["SalePrice"]]
# Fit and transform on the train set
X_train[columns]= scalerX .fit_transform(X_train[columns])
test_data[columns]= scalerX.transform(test_data[columns])
y_train= scalery.fit_transform(y_train)
# list of alphas to tune
params = {'alpha': [0.1, 1,2,3,4,5,6,10,50,100,200,300,500]}

ridge = Ridge()
# cross validation
model_cv_r= GridSearchCV(estimator = ridge, param_grid = params, scoring= 'neg_mean_absolute_error',cv = 5, 
                        return_train_score=True,verbose = 1)            
model_cv_r .fit(X_train, y_train) 
# summarize the results of the grid search
cv_results_r = pd.DataFrame(model_cv_r.cv_results_)

print(model_cv_r.best_score_)
print(model_cv_r.best_estimator_.alpha)
# plotting mean test and train scoes with alpha 
cv_results_r['param_alpha'] = cv_results_r['param_alpha'].astype('float32')

# plotting
plt.figure(figsize=(15,8))
plt.plot(cv_results_r['param_alpha'], cv_results_r['mean_train_score'])
plt.plot(cv_results_r['param_alpha'], cv_results_r['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()
model_r = Ridge(alpha=5)
model_r.fit(X_train, y_train)
print(model_r.coef_)
modelr = pd.DataFrame()

modelr["paramters"] = list(X_train.columns)
modelr["coefficient"] = model_r.coef_[0]
modelr = modelr.sort_values(by ="coefficient", ascending =False).reset_index()
plt.figure(figsize=(20,10))
sns.barplot(modelr["paramters"],modelr["coefficient"])
modell = modelr[modelr["coefficient"]!=0]
plt.xticks(rotation = 90)
y_pred_rg = model_r.predict(X_train)
# Model Matrices
print ("R Square:",r2_score(y_train, y_pred_rg))
print ("RMSE:",np.sqrt(mean_squared_error(y_train, y_pred_rg)))
plt.scatter(y_train, y_pred_rg)
# Validation of the model
error  = y_train-y_pred_rg
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.title("Error Terms distribution for Ridge Regression")
sns.distplot(error)
plt.subplot(1,2,2)
plt.title("Error Terms for Ridge Regression")
plt.scatter(y_train,error)
plt.axhline(y=0,color='r')
plt.tight_layout()
## Lasso Regression

lasso = Lasso()
params = {'alpha': [0.00001,0.0001,0.0005,0.0009,0.001,0.002,0.003, 0.005,0.008,0.01,0.05]}

# cross validation
model_cvl = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = 5, 
                        return_train_score=True,
                        verbose = 1)            

model_cvl.fit(X_train, y_train)
print(model_cvl.best_score_)
print(model_cvl.best_estimator_.alpha)
cv_results_l= pd.DataFrame(model_cvl.cv_results_)

# plotting mean test and train scoes with alpha 
cv_results_l['param_alpha'] = cv_results_l['param_alpha'].astype('float32')

# plotting
plt.figure(figsize=(15,8))
plt.plot(cv_results_l['param_alpha'], cv_results_l['mean_train_score'])
plt.plot(cv_results_l['param_alpha'], cv_results_l['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()
model_l = Lasso(alpha=0.001)
model_l.fit(X_train, y_train)
print(model_l.coef_)
modell = pd.DataFrame()
modell["paramters"] = list(X_train.columns)
modell["coefficient"] = model_l.coef_
modell = modell[modell["coefficient"]!=0]
modell = modell.sort_values(by ="coefficient", ascending =False).reset_index()
plt.figure(figsize=(20,10))
sns.barplot(modell["paramters"],modell["coefficient"])
plt.xticks(rotation = 90)
y_pred_l = model_l.predict(X_train)
print ("R Square:",r2_score(y_train, y_pred_l))
print ("RMSE:",np.sqrt(mean_squared_error(y_train, y_pred_l)))
# Validation of the model
y_pred_ls = y_pred_l.reshape(-1,1)
error  = y_train-y_pred_ls
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.title("Error Terms distribution for Ridge Regression")
sns.distplot(error)
plt.subplot(1,2,2)
plt.title("Error Terms for Ridge Regression")
plt.scatter(y_train,error)
plt.axhline(y=0,color='r')
plt.tight_layout()
error.shape
# Checking the missing data in test dataset
missing_values = pd.DataFrame(round((test_data.isnull().sum()/len(X_test_scale)*100),2))
missing_values.columns=["Missing_Value%"]
missing_values = missing_values[(missing_values["Missing_Value%"])>0].sort_values("Missing_Value%",ascending =False)
missing_values
# Imputing the missing data for Test datase with median values
def impute_median(item):
    test_data[item] = test_data[item].fillna(test_data[item].median())
for item in (["MasVnrArea","BsmtFullBath","TotalBath","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","GarageArea","TotalArea"]):
    impute_median(item)
X_test = test_data[X_train.columns]
y_pred = model_l.predict(X_test)
test_data["SalesPrice"] = y_pred
test_data["SalesPrice"]=scalery.inverse_transform(test_data["SalesPrice"])
test_data["SalesPrice"] = np.expm1(test_data["SalesPrice"])
submission  = test_data[["Id","SalesPrice"]]
submission.to_csv("submission.csv")