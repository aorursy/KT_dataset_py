import pandas as pd
import numpy as np
data = pd.read_csv('train.csv')
data.head()
data.isnull().sum()
data.describe()
data.info()
testdata = pd.read_csv('test.csv')
testdata
testdata.info()
# drop columns that have lots of NaN like MiscFeature, Fence, PoolQC and Alley
dropcolumn = ['MiscFeature', 'Fence', 'PoolQC', 'Alley']
data.drop(columns = dropcolumn, inplace=True)
testdata.drop(columns=dropcolumn, inplace= True)

new  = data.replace(np.nan,' ', regex= True)
new


# convert categorical into numeric data 
for i in new[['FireplaceQu','GarageCond','GarageQual','FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtCond', 'BsmtQual', 'ExterCond','ExterQual']]:
    new[i] = new[i].replace(['Ex', 'Gd','TA','Fa','Po', 'NA'],['5','4','3','2','1','0'])
for i in new[['BsmtFinType2','BsmtFinType1']]:
    new[i]= new[i].replace(['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'], ['5','4','3','2','1','0','0'])
new['BsmtExposure'] = new['BsmtExposure'].replace(['Gd', 'Av', 'Mn', 'No', 'NA'], ['3','2','1', '0', '0'])

# convert data type from object to float
list1 = new[['GarageCond','GarageQual','FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtCond', 'BsmtQual', 'ExterCond','ExterQual', 'BsmtFinType2','BsmtFinType1', 'BsmtExposure']]
for i in list1:    
    new[i]= pd.to_numeric(new[i],errors='coerce')
    print(new[i])



new.info()
plt.scatter(new['GrLivArea'], new['SalePrice'], alpha = 0.1, label = "ID")               # x在前,y在后
plt.xlabel("Above ground living area square feet")                
plt.ylabel("SalePrice")          
plt.title("GrLivArea vs SalePrice") 
plt.legend()
# delete outliers
new = new.drop(new[(new['GrLivArea']>4000) & (new['SalePrice']<300000)].index)

#Check the graphic again
plt.scatter(new['GrLivArea'], new['SalePrice'], alpha = 0.1, label = "ID")               # x在前,y在后
plt.xlabel("Above ground living area square feet")                
plt.ylabel("SalePrice")          
plt.title("GrLivArea vs SalePrice") 
plt.legend()
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(new["SalePrice"])
plt.ylabel("Density")
plt.title("Distribution of SalePrice")
print(new["SalePrice"].describe())
new["LotFrontage"]= pd.to_numeric(new["LotFrontage"],errors='coerce')

sns.distplot(new["LotFrontage"])
plt.ylabel("Density")
plt.title("Distribution of LotFrontage")
print(new["LotFrontage"].describe())



new['SalePrice']= np.log(new['SalePrice'])
new['SalePrice']
plt.scatter(new['LotFrontage'], new['SalePrice'], alpha = 0.1, label = "ID")               # x在前,y在后
plt.xlabel("Linear feet of street connected to property")                
plt.ylabel("SalePrice")          
plt.title("LotFrontage vs SalePrice") 
plt.legend()
sns.distplot(new["LotArea"])
plt.ylabel("Density")
plt.title("Distribution of LotArea")
print(new["LotArea"].describe())
plt.scatter(new['LotArea'], new['SalePrice'], alpha = 0.1, label = "ID")               # x在前,y在后
plt.xlabel("LotArea")                
plt.ylabel("SalePrice")          
plt.title("LotArea vs SalePrice") 
plt.legend()

sns.distplot(new["YearBuilt"])
plt.ylabel("Density")
plt.title("Distribution of YearBuilt")
print(new["YearBuilt"].describe())
plt.scatter(new['YearBuilt'], new['SalePrice'], alpha = 0.1, label = "ID")               # x在前,y在后
plt.xlabel("YearBuilt")                
plt.ylabel("SalePrice")          
plt.title("YearBuilt vs SalePrice") 
plt.legend()

sns.distplot(new["GrLivArea"])
plt.ylabel("Density")
plt.title("Distribution of GrLivArea")
print(new["GrLivArea"].describe())
plt.scatter(new['GrLivArea'], new['SalePrice'], alpha = 0.1, label = "ID")               # x在前,y在后
plt.xlabel("Above ground living area square feet")                
plt.ylabel("SalePrice")          
plt.title("GrLivArea vs SalePrice") 
plt.legend()

sns.distplot(new["GarageArea"])
plt.ylabel("Density")
plt.title("Distribution of GrLivArea")
print(new["GarageArea"].describe())

plt.scatter(new['GarageArea'], new['SalePrice'], alpha = 0.1, label = "ID")               # x在前,y在后
plt.xlabel("GarageArea")                
plt.ylabel("SalePrice")          
plt.title("GarageArea vs SalePrice") 
plt.legend()

plt.show()
new['GarageYrBlt'] = new['GarageYrBlt'].astype(str)
plt.scatter(new['GarageYrBlt'], new['SalePrice'], alpha = 0.1, label = "ID")               # x在前,y在后
plt.xlabel("GarageYrBlt")                
plt.ylabel("SalePrice")          
plt.title("GarageYrBlt vs SalePrice") 
plt.legend()

plt.show()

sns.distplot(new["1stFlrSF"])
plt.ylabel("Density")
plt.title("Distribution of 1stFlrSF")
print(new["1stFlrSF"].describe())


plt.scatter(new['1stFlrSF'], new['SalePrice'], alpha = 0.1, label = "ID")               # x在前,y在后
plt.xlabel("1stFlrSF")                
plt.ylabel("SalePrice")          
plt.title("1stFlrSF vs SalePrice") 
plt.legend()

plt.show()
sns.distplot(new["2ndFlrSF"])
plt.ylabel("Density")
plt.title("Distribution of 2ndFlrSF")
print(new["2ndFlrSF"].describe())

plt.scatter(new['2ndFlrSF'], new['SalePrice'], alpha = 0.1, label = "ID")               # x在前,y在后
plt.xlabel("2ndFlrSF")                
plt.ylabel("SalePrice")          
plt.title("2ndFlrSF vs SalePrice") 
plt.legend()

plt.show()
sns.distplot(new["TotalBsmtSF"])
plt.ylabel("Density")
plt.title("Distribution of TotalBsmtSF")
print(new["TotalBsmtSF"].describe())
    
plt.scatter(new['TotalBsmtSF'], new['SalePrice'], alpha = 0.1, label = "ID")               # x在前,y在后
plt.xlabel("TotalBsmtSF")                
plt.ylabel("SalePrice")          
plt.title("TotalBsmtSF vs SalePrice") 
plt.legend()

plt.show()
sns.distplot(new["BsmtFinSF1"])
plt.ylabel("Density")
plt.title("Distribution of BsmtFinSF1")
print(new["BsmtFinSF1"].describe())
  
plt.scatter(new['BsmtFinSF1'], new['SalePrice'], alpha = 0.1, label = "ID")               # x在前,y在后
plt.xlabel("BsmtFinSF1")                
plt.ylabel("SalePrice")          
plt.title("BsmtFinSF1 vs SalePrice") 
plt.legend()

plt.show()
plt.scatter(new['BsmtFinSF2'], new['SalePrice'], alpha = 0.1, label = "ID")               # x在前,y在后
plt.xlabel("BsmtFinSF2")                
plt.ylabel("SalePrice")          
plt.title("BsmtFinSF2 vs SalePrice") 
plt.legend()

plt.show()
plt.scatter(new['BsmtUnfSF'], new['SalePrice'], alpha = 0.1, label = "ID")               # x在前,y在后
plt.xlabel("BsmtUnfSF")                
plt.ylabel("SalePrice")          
plt.title("BsmtUnfSF vs SalePrice") 
plt.legend()

plt.show()
plt.scatter(new['OpenPorchSF'], new['SalePrice'], alpha = 0.1, label = "ID")               # x在前,y在后
plt.xlabel("OpenPorchSF")                
plt.ylabel("SalePrice")          
plt.title("OpenPorchSF vs SalePrice") 
plt.legend()

plt.show()
new['EnclosedPorch'].value_counts()
plt.scatter(new['EnclosedPorch'], new['SalePrice'], alpha = 0.1, label = "ID")               # x在前,y在后
plt.xlabel("EnclosedPorch")                
plt.ylabel("SalePrice")          
plt.title("EnclosedPorch vs SalePrice") 
plt.legend()

plt.show()
import seaborn as sns
sns.boxplot(x = 'OverallQual', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()
sns.boxplot(x = 'OverallCond', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()

sns.boxplot(x = 'ExterQual', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()

sns.boxplot(x = 'ExterCond', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()

# Heating analysis
sns.boxplot(x = 'HeatingQC', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()
sns.boxplot(x = 'Heating', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()

#Grav is expensive, so it will lower sale price
new['KitchenAbvGr'].value_counts()
# most of the houses only have 1 kitchen
sns.boxplot(x = 'KitchenAbvGr', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()

sns.boxplot(x = 'KitchenQual', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()
# Fireplace analysis
sns.boxplot(x = 'FireplaceQu', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()

sns.boxplot(x = 'Fireplaces', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()

# clearly, more firplaces and fireplace quality lead to a higher price
# Garage analysis
sns.boxplot(x = 'GarageCond', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()
sns.boxplot(x = 'GarageCars', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()
# the more cars the garage can accommodate, the higher price, but not 4 
sns.boxplot(x = 'GarageQual', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()

new['GarageFinish'] = new['GarageFinish'].replace(' ','0')
new['GarageFinish'].value_counts().index
sns.boxplot(x = 'GarageFinish', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()

# garage facility shows that the higher the quality is, the higher sale price is 
# does not necessarily come with a more car capacity, garage condition, and quality
# basement analysis 
sns.boxplot(x = 'BsmtQual', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()

sns.boxplot(x = 'BsmtCond', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()

sns.boxplot(x = 'BsmtExposure', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()

sns.boxplot(x = 'BsmtFinType1', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()
sns.boxplot(x = 'BsmtFinType2', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()


sns.boxplot(x = 'BsmtFullBath', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()
# there is a little difference that more basement full bath comes with a higher sale price
sns.boxplot(x = 'BsmtHalfBath', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()
# every feature for basement shows that a higher score with a higher price
# some features such as BsmtFullBath and BsmtHalfBath have meager influence on sale price
sns.boxplot(x = 'FullBath', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()
sns.boxplot(x = 'HalfBath', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()
sns.boxplot(x = 'BedroomAbvGr', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()
sns.boxplot(x = 'TotRmsAbvGrd', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()
new= pd.get_dummies(new, columns =['CentralAir'], drop_first = True  )

sns.boxplot(x = 'CentralAir_Y', y = 'SalePrice', data = new, palette = 'Blues')
plt.tight_layout()

plt.show()


# see the correlation between features 
import seaborn as sns
f, ax = plt.subplots(figsize = (40,40))
mask = np.zeros_like(new.corr(), dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap = True)
sns.heatmap(new.corr(), annot = True, cmap = cmap, mask = mask, center = 0, square = True)
plt.show()


cor = np.abs(new.corr(method='spearman')['SalePrice']).sort_values(ascending = False)
cor
# high correlation (>0.69) we might not use them in the regression at the same time
# ExterQual and OverallQual
# BsmtQual and YearBuilt
# BsmtFinSF1 and BsmtFintype1
# BsmtFinSF2 and BsmtFintype2
# 1stFlrSF and TotalBsmtSF 
# KitchenQual and ExterQual
# TotRmsAbvGrd and GrLivArea 
# GrLivArea = 1stFlrSF + 2ndFlrSF, we use GrLivArea only 
# GarageCars and GarageArea
# we use OverallQual, GrLivArea, GarageCars,  KitchenQual, BsmtQual ,
# FullBath,TotalBsmtSF , Fireplaces ,HeatingQC, OpenPorchSF, LotArea as our X 
# Benchmark model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# delete outliers
data = data.drop(data[(data['GrLivArea']>4000) & (data['SalePrice']<300000)].index)

x = data[['OverallQual', 'GrLivArea', 'GarageCars',  'KitchenQual', 'BsmtQual' ,'FullBath','TotalBsmtSF' , 'Fireplaces' ,'HeatingQC', 'OpenPorchSF', 'LotArea']]
x = x.fillna(0)
y = data['SalePrice']

print(x)
print(y)


for i in x:
       x = x.replace(['Ex', 'Gd','TA','Fa','Po', 'NA'],['5','4','3','2','1','0'])

# convert data type from object to float
list1 = data[['OverallQual', 'GrLivArea', 'GarageCars',  'KitchenQual', 'BsmtQual' ,'FullBath','TotalBsmtSF' , 'Fireplaces' ,'HeatingQC', 'OpenPorchSF', 'LotArea']]
for i in list1:    
    data[i]= pd.to_numeric(data[i],errors='coerce')
    print(data[i])
    
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2, random_state = 0)
print(xtrain)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

xtrain_bench = xtrain
xtest_bench = xtest

np.any(np.isnan(xtrain_bench))
np.any(np.isnan(ytrain))

np.all(np.isfinite(xtrain_bench))
np.all(np.isfinite(ytrain))
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain_bench, ytrain)
ypred_bench = model.predict(xtest_bench)
print(ypred_bench)
# coefficient of determination

from sklearn.metrics import r2_score
print(r2_score(ytest,ypred_bench))
compar1 = pd.DataFrame({'Actual': ytest, 'Predicted': ypred_bench})
compar1
from sklearn.metrics import mean_squared_error, mean_absolute_error

test_mse = mean_squared_error(ytest,ypred_bench)
test_rmse = np.sqrt(test_mse)

test_mae = mean_absolute_error(ytest, ypred_bench)
print(f"the test RMSE is: {test_rmse}")
print(f"the test MAE is: {test_mae}")
print(bench_model.coef_)
print(bench_model.intercept_)


# Computing the accuracy with k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = bench_model, X = xtrain_bench, y = ytrain, cv = 10)
print(accuracies)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
# get a bench_model summary
xtrain_bench = np.append(arr = np.ones((1166,1)).astype(int) , values =xtrain_bench, axis = 1 )

import statsmodels.api as sm
x_opt = np.array(xtrain_bench[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11]], dtype=float)

regressor_OLS = sm.OLS(endog = ytrain, exog = x_opt).fit()
regressor_OLS.summary()

# maybe we can discard x10 and see if the prediction score will increase
######################################
# we drop x10 (OpenPorchSF) since it #
# is statistically insignificant     #
######################################


# x10 -> -0.7
print(xtrain[0])

xtrain_2 = np.delete(xtrain, np.s_[9], axis =1 )
print(xtrain_2)
xtest_2 = np.delete(xtest, np.s_[9], axis =1 )
print(xtest_2)
model_2 = LinearRegression()
model_2.fit(xtrain_2, ytrain)
ypred_2 = model_2.predict(xtest_2)
print(r2_score(ytest,ypred_2))
compar2 = pd.DataFrame({'Actual': ytest, 'Predicted': ypred_2})
compar2
test_mse = mean_squared_error(ytest,ypred_2)
test_rmse = np.sqrt(test_mse)

test_mae = mean_absolute_error(ytest, ypred_2)
print(f"the test RMSE is: {test_rmse}")
print(f"the test MAE is: {test_mae}")
print(model_2.coef_)
print(model_2.intercept_)

# Computing the accuracy with k-Fold Cross Validation
accuracies = cross_val_score(estimator = model_2, X = xtrain_2, y = ytrain, cv = 10)
print(accuracies)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
# get a model_2 summary
xtrain_2 = np.append(arr = np.ones((1166,1)).astype(int) , values =xtrain_2, axis = 1 )

import statsmodels.api as sm
x_opt = np.array(xtrain_2[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10]], dtype=float)
regressor_OLS = sm.OLS(endog = ytrain, exog = x_opt).fit()
regressor_OLS.summary()

######################################
# we drop x6 (FullBath) since it #
# is statistically insignificant     #
######################################

print(xtrain_2[0])
xtrain_3 = np.delete(xtrain_2, np.s_[0], axis =1 )

xtrain_3 = np.delete(xtrain_3, np.s_[5], axis =1 )
xtest_3 = np.delete(xtest_2, np.s_[0], axis =1 )

xtest_3 = np.delete(xtest_2, np.s_[5], axis =1 )

model_3 = LinearRegression()
model_3.fit(xtrain_3, ytrain)
ypred_3 = model_3.predict(xtest_3)
print(r2_score(ytest,ypred_3))
compar3 = pd.DataFrame({'Actual': ytest, 'Predicted': ypred_3})
compar3
test_mse = mean_squared_error(ytest,ypred_3)
test_rmse = np.sqrt(test_mse)

test_mae = mean_absolute_error(ytest, ypred_3)
print(f"the test RMSE is: {test_rmse}")
print(f"the test MAE is: {test_mae}")
print(model_3.coef_)
print(model_3.intercept_)

# Computing the accuracy with k-Fold Cross Validation
accuracies = cross_val_score(estimator = model_3, X = xtrain_3, y = ytrain, cv = 10)
print(accuracies)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
# Highest accuracy score in k fold validation 81.07%
# get model_3 summary
xtrain_3 = np.append(arr = np.ones((1166,1)).astype(int) , values =xtrain_3, axis = 1 )

import statsmodels.api as sm
x_opt = np.array(xtrain_3[:, [0, 1, 2, 3, 4, 5,6,7,8,9]], dtype=float)
regressor_OLS = sm.OLS(endog = ytrain, exog = x_opt).fit()
regressor_OLS.summary()
# an unit increase in OverallQual, GrLivArea, KitchenQual and TotalBsmtSF
# will have larger and positive impact in sale price
# we can tell that people prefer a bigger ground and basement size and great 
# overall and kitchen quality


# call test data
testdata = pd.read_csv('test.csv')
testdata
# we use model 3 to predict
# first we do some feature selection and transformation
# replace NaN with blank
test_newdata = testdata.replace(np.nan,' ', regex= True)
test_newdata= test_newdata[['OverallQual', 'GrLivArea', 'GarageCars',  'KitchenQual', 'BsmtQual' ,'TotalBsmtSF' , 'Fireplaces' ,'HeatingQC', 'LotArea']]
print(test_newdata)


for i in test_newdata[['KitchenQual', 'BsmtQual'  ,'HeatingQC']]:
    test_newdata[i] = test_newdata[i].replace(['Ex', 'Gd','TA','Fa','Po', 'NA'],['5','4','3','2','1','0'])
print(test_newdata.info())
# convert data type from object(str) to float
list1 = test_newdata[['OverallQual', 'GrLivArea', 'GarageCars',  'KitchenQual', 'BsmtQual' ,'TotalBsmtSF' , 'Fireplaces' ,'HeatingQC', 'LotArea']]
for i in list1:    
    test_newdata[i]= pd.to_numeric(test_newdata[i],errors='coerce')
    print(test_newdata[i])
# for features that have na, we replace with mode (GarageCars, KitchenQual, BsmtQual )
# for TotalBsmtSF, we replace na with mean

# replace NA with mode
test_newdata['KitchenQual'] = test_newdata['KitchenQual'].fillna(test_newdata['KitchenQual'].mode()[0])
test_newdata['GarageCars'] = test_newdata['GarageCars'].fillna(test_newdata['GarageCars'].mode()[0])
test_newdata['BsmtQual'] = test_newdata['BsmtQual'].fillna(test_newdata['BsmtQual'].mode()[0])

test_newdata['TotalBsmtSF'].mean()
test_newdata['TotalBsmtSF'] = test_newdata['TotalBsmtSF'].fillna(1046.1179698216736)

print(test_newdata.info())
test_newdata.isna().sum()
sc = StandardScaler()

test_newdata = sc.fit_transform(test_newdata)

prediction = model_3.predict(test_newdata)
print(prediction.reshape(-1,1))
prediction = pd.DataFrame(prediction)
prediction.columns= ['SalePrice']
prediction = prediction.round(2)

ID= pd.DataFrame(testdata['Id'])
ID
submission = pd.concat([ID,prediction], axis = 1)
submission.head()
submission.to_csv('Submission_new.csv', index = False)



