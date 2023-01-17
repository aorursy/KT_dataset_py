import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge,Lasso
warnings.filterwarnings('ignore')
%matplotlib inline
from scipy.stats import skew

print("ok")
train= pd.read_csv('../input/train.csv')
train_data = pd.read_csv('../input/train.csv')
print (train.head())
#print (train.columns)
#print(train.shape)
test= pd.read_csv('../input/test.csv')
submission_ID = test['Id']
#print (test.head())
#print (test.columns)
print(test.shape)
train.drop("Id",axis=1,inplace = True)
print(train.shape)
test.drop("Id",axis=1,inplace=True)
print (test.shape)
#Check For Outliers



plt.scatter(train['GrLivArea'],train['SalePrice'],c = "Red")



plt.xlabel("GrLivArea")



plt.ylabel("SalePrice")



plt.show()






#Removing Outliers



train = train[train['GrLivArea']<4000]
test = test[test['GrLivArea']<4000]

plt.scatter(train['GrLivArea'],train['SalePrice'],c = "Red")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()
print (train.shape)



print (test.shape)



#Checking the distribution of plot (The distribution is right skewed )


sns.distplot(train['SalePrice'], fit=norm )
#Normalizing the Y values 



train['SalePrice'] = np.log1p(train['SalePrice'])




#print(train['SalePrice'])


#Y value after normalisation 

sns.distplot(train['SalePrice'] , fit=norm)

plt.ylabel('frequency')


#for i in train.columns :

#   print  ( "the feature is " + str(i) + "the no. of nulls is " + str(train[i].isnull().sum()) )


print (train.shape)

print (test.shape)


for i in ['BsmtQual',"BsmtCond","BsmtExposure","FireplaceQu","Fence","BsmtFinType1","BsmtFinType2","GarageType","GarageFinish","MiscFeature","GarageQual","GarageCond"]:

    
    train[i] = train[i].fillna("No")
    
    test[i] = test[i].fillna("No")
    #print(train[i].isnull().sum())
for i in ['BsmtFullBath',"LotFrontage",'BsmtFinSF1','TotalBsmtSF','BsmtFinSF2','BsmtHalfBath','BsmtUnfSF','Fireplaces',"MiscFeature","MasVnrArea",'GarageArea',"GarageCars",'HalfBath','LotFrontage','MiscVal','MasVnrArea',"PoolQC",'ScreenPorch','TotRmsAbvGrd','WoodDeckSF'] :
    train[i] = train[i].fillna(0) 
    test[i] = test[i].fillna(0)
   # print(train[i].isnull().sum())
for i in ["Condition1","Condition2"] :
    train[i] = train[i].fillna("Norm")
    test[i] = test[i].fillna("Norm")
    print(train[i].isnull().sum())
for i in ["Alley","MasVnrType"] :

    train[i] = train[i].fillna("None")
    
    test[i] = test[i].fillna("None")
    
    print(train[i].isnull().sum())
    
    print(test[i].isnull().sum())
    

print (train.shape)
print (test.shape)
for i in ["HeatingQC","KitchenQual"] :
    train[i] = train[i].fillna("TA")
    test[i] = test[i].fillna("TA")

    print(test[i].isnull().sum())
    print(train[i].isnull().sum())
test['MSZoning'] = (test['MSZoning'].fillna("RL"))
(test[['Utilities','Exterior1st','Exterior2nd','SaleType','Functional']].mode())
test['Utilities'] = test['Utilities'].fillna("AllPub")

test['Exterior1st'] = test['Exterior1st'].fillna("VinylSd")

test['Exterior2nd'] = test['Exterior2nd'].fillna("VinylSd")

test['SaleType']= test['SaleType'].fillna("WD")

test['Functional'] = test['Functional'].fillna("Typ")


print (train.shape)

print (test.shape)


print(train.isnull().sum())
print(test.isnull().sum())
train.drop('GarageYrBlt',axis = 1,inplace = True)

test.drop('GarageYrBlt',axis = 1,inplace = True)


train['SalePrice'] = np.log1p(train['SalePrice'])

y = train['SalePrice']


train.drop('SalePrice',axis = 1,inplace = True)

train['MSSubClass'] = train['MSSubClass'].apply(str)
train['OverallCond'] = train['OverallCond'].astype(str)
train['YrSold'] = train['YrSold'].astype(str)
train['MoSold'] = train['MoSold'].astype(str)
train.shape
test['MSSubClass'] = test['MSSubClass'].apply(str)
test['OverallCond'] = test['OverallCond'].astype(str)
test['YrSold'] = test['YrSold'].astype(str)
test['MoSold'] = test['MoSold'].astype(str)
print(train.shape)
print (test.shape)
from sklearn.preprocessing import LabelEncoder

cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
       'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
      'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
     'YrSold', 'MoSold']
for i in cols :
    lb = LabelEncoder()
    lb.fit(list(train[i].values) )
    train[i] = lb.transform(list(train[i].values))
print(train['BsmtQual'].unique())

for i in cols :
    lb = LabelEncoder()
    lb.fit(list(test[i].values) )
    test[i] = lb.transform(list(test[i].values))
print (test['BsmtQual'].unique())
print ("The correlation to target variable is ")
corr_train = (train.corr())
corr_test = test.corr()
print (corr_train['SalePrice'].sort_values(ascending = False))
print (corr_test['SalePrice'].sort_values(ascending = False))
numerical_features_train = train.select_dtypes(exclude = ['object'])
numerical_features_test = test.select_dtypes(exclude = ['object'])

print (numerical_features_train.isnull().sum())
skewed = numerical_features_train.apply(lambda x : skew(x))
skewed = skewed[abs(skewed) > 0.5]
skewed_features = skewed.index
numerical_features_train[skewed_features] = np.log1p(numerical_features_train[skewed_features])
print (numerical_features_train.columns)

skewed = numerical_features_test.apply(lambda x : skew(x))
skewed = skewed[abs(skewed) > 0.5]
skewed_features = skewed.index
numerical_features_test[skewed_features] = np.log1p(numerical_features_test[skewed_features])
print (numerical_features_test.columns)
categorical_features_train = train.select_dtypes(include = ['object'])
categorical_features_test = test.select_dtypes(include = ['object'])
print(categorical_features_train)
print (categorical_features_train.isnull().sum())
categorical_features_train.isnull().sum()
#categorical_features_train['Electrical'] = categorical_features_train['Electrical'].fillna("SBrkr")
print (categorical_features_train.isnull().sum())
print("the length of df is ")
#print (len((categorical_features_train).columns))

categorical_features_test.isnull().sum()
#categorical_features_test['Electrical'] = categorical_features_test['Electrical'].fillna("SBrkr")
print (categorical_features_test.isnull().sum())
print ("The length of df is")
##print (len((categorical_features_test).columns))
train_objs_num = len(categorical_features_train)
dataset = pd.concat(objs=[categorical_features_train, categorical_features_test], axis=0)
dataset = pd.get_dummies(dataset)
categorical_features_train = (dataset[:train_objs_num])
categorical_features_test = (dataset[train_objs_num:])
print (categorical_features_train.columns)
print (categorical_features_test.columns)
for i in categorical_features_train.columns:
        print ((len((categorical_features_train[i].unique()))))
#print ("line break ")
for i in categorical_features_test.columns:
        print ((len((categorical_features_test[i].unique()))))
train = pd.concat([numerical_features_train,categorical_features_train],axis= 1)
print (train.shape)
test = pd.concat([numerical_features_test,categorical_features_test],axis= 1)
print (test.shape)
X_train,X_test,Y_train,Y_test = train_test_split(train,y, test_size = 0.2,random_state = 42)
print (X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
#print (X_train.head(10))
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV,ElasticNetCV,LinearRegression
from sklearn.model_selection import cross_val_score

def cv_error_train(mod):
    rmse_train = np.sqrt(-cross_val_score(mod, X_train, Y_train,scoring= "neg_mean_squared_error" ,cv = 10))
    return (rmse_train)
def cv_error_test(mod):
    rmse_test = np.sqrt(-cross_val_score(mod, X_test, Y_test,scoring = "neg_mean_squared_error" ,cv = 10))
    return (rmse_test)
#Linear_Regression_Without_Regularization :

model = LinearRegression()
train_model = model.fit(X_train,Y_train)
test_model = model.fit(X_test,Y_test)

print (cv_error_train(train_model).mean())
print (cv_error_test(test_model).mean())

y_train_predict = train_model.predict(X_train)
y_test_predict = test_model.predict(X_test)

#plotting the residual sum of squares 

plt.scatter(y_train_predict,y_train_predict - Y_train,c = "blue",label = "Training data")
plt.scatter(y_test_predict,y_test_predict - Y_test,c = "red",label = "Test data",)

plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")

plt.show()

plt.scatter(y_train_predict,Y_train,c = "blue",label = "Training data")
plt.scatter(y_test_predict,Y_test,c = "red",label = "Test data",)

plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real Value")

plt.show()
y_predict = train_model.predict(test)
print (y_predict)
#Ridge Regression 

ridge = RidgeCV(alphas= [0.01,0.03,0.05,0.1,0.3,0.5,1.0,3.0,5.0,10.0,15.0,30.0,50.0,100.0])
ridge.fit(X_train,Y_train)

print(ridge.alpha_)
print (cv_error_train(ridge).mean())
print (cv_error_test(ridge).mean())

y_train_predict = ridge.predict(X_train)
y_test_predict = ridge.predict(X_test)
#plotting the residual sum of squares 

plt.scatter(y_train_predict,y_train_predict - Y_train,c = "blue",label = "Training data")
plt.scatter(y_test_predict,y_test_predict - Y_test,c = "red",label = "Test data",)

plt.title("Ridge regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")

plt.show()

plt.scatter(y_train_predict,Y_train,c = "blue",label = "Training data")
plt.scatter(y_test_predict,Y_test,c = "red",label = "Test data",)

plt.title("Ridge regression")
plt.xlabel("Predicted values")
plt.ylabel("Real Value")

plt.show()
y_predict_ridge = ridge.predict(test)
print (y_predict_ridge)
coeff = pd.Series(ridge.coef_,index= X_train.columns)
print (coeff.sort_values(ascending = False))
print( "The ridge model has eliminated " + str(sum(coeff == 0)) + " features")
print ("The ridge model has kept " + str(sum(coeff != 0)) + " features")
top_bottom_10 = pd.concat([coeff.sort_values().head(10),coeff.sort_values().tail(10)])

top_bottom_10.plot(kind = "barh")
lasso = LassoCV(alphas= [0.0001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5,1.0,3.0,5.0,10.0,30.0,50.0],cv = 10,max_iter = 50000)
lasso.fit(X_train,Y_train)

print(lasso.alpha_)
print (cv_error_train(lasso).mean())
print (cv_error_test(lasso).mean())

y_train_predict = lasso.predict(X_train)
y_test_predict = lasso.predict(X_test)
#plotting the residual sum of squares 

plt.scatter(y_train_predict,y_train_predict - Y_train,c = "blue",label = "Training data")
plt.scatter(y_test_predict,y_test_predict - Y_test,c = "red",label = "Test data",)

plt.title("Lasso regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")

plt.show()

plt.scatter(y_train_predict,Y_train,c = "blue",label = "Training data")
plt.scatter(y_test_predict,Y_test,c = "red",label = "Test data",)

plt.title("Lasso regression")
plt.xlabel("Predicted values")
plt.ylabel("Real Value")

plt.show()
y_test_predict = lasso.predict(X_test)
coef = pd.Series(lasso.coef_,index= X_train.columns)
print ("The lasso model has kept " + str(sum(coef != 0)) + " features")
print ("The lasso model has eliminated " + str(sum(coef == 0)) + " features")
top_bottom_10 = pd.concat([coeff.sort_values().head(10),coeff.sort_values().tail(10)])

top_bottom_10.plot(kind = "barh")
y_predict_lasso = lasso.predict(test)
print (y_predict)
elastic_cv = ElasticNetCV( l1_ratio = [0.1,0.3,0.5,0.7,1.0],
                            alphas= [0.0001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5,1.0,3.0,5.0,10.0,30.0,50.0],
                           cv = 10)

elastic_cv.fit(X_train,Y_train)

print(elastic_cv.alpha_)
print (cv_error_train(elastic_cv).mean())
print (cv_error_test(elastic_cv).mean())

y_train_predict = elastic_cv.predict(X_train)
y_test_predict = elastic_cv.predict(X_test)
#plotting the residual sum of squares 

plt.scatter(y_train_predict,y_train_predict - Y_train,c = "blue",label = "Training data")
plt.scatter(y_test_predict,y_test_predict - Y_test,c = "red",label = "Test data",)

plt.title("Elastic_CV")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")

plt.show()

plt.scatter(y_train_predict,Y_train,c = "blue",label = "Training data")
plt.scatter(y_test_predict,Y_test,c = "red",label = "Test data",)

plt.title("Elastic_CV")
plt.xlabel("Predicted values")
plt.ylabel("Real Value")

plt.show()
y_predict = elastic_cv.predict(test)
print (y_predict)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=30, n_estimators=500, max_features = 100, oob_score=True, n_jobs=-1, random_state=1234)
rf.fit(X_train,Y_train)

print (cv_error_train(rf).mean())
print (cv_error_test(rf).mean())
y_predict_rf = rf.predict(test)
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(X_train, Y_train)

print (cv_error_train(rf).mean())
print (cv_error_test(rf).mean())
y_predict_gbr = gbr.predict(test)
import xgboost as xgb
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)
model_xgb.fit(X_train, Y_train)


model_xgb.fit(X_train, Y_train)

print (cv_error_train(model_xgb).mean())
print (cv_error_test(model_xgb).mean())
import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

lgb_model.fit(X_train, Y_train)

print (cv_error_train(lgb_model).mean())
print (cv_error_test(lgb_model).mean())
y_predict_lgb = lgb_model.predict(test)
Final_price = np.expm1(y_predict_lgb)
Final_price_1 = np.expm1(Final_price)
sub = pd.DataFrame()
sub['Id'] = submission_ID
sub = sub[sub['Id'] != 2550]
sub['SalePrice'] = Final_price_1.round(2)
sub.to_csv('submission.csv',index=False)
print (sub)