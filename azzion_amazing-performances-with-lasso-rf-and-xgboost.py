import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
df.head()
test.head(5)
# total number of missing cells in our dataframe
df.isnull().sum().max()
# Run the below code to get the count % of NaN values
#df.isnull().sum()/1460
df['SalePrice'].describe()
sns.distplot(df['SalePrice'])
print("Kurtosis : "+str(df['SalePrice'].kurt()))
print("Skew : "+str(df['SalePrice'].skew()))
plt.scatter(x=df['LotArea'],y=df['SalePrice'])
plt.show()
plt.scatter(x=df['GrLivArea'],y=df['SalePrice'])
plt.show()
plt.scatter(x=df['TotRmsAbvGrd'],y=df['SalePrice'])
plt.show()
f, ax = plt.subplots(figsize=(12, 9))
sns.boxplot(x=df['OverallQual'],y=df['SalePrice'])
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corrmat,vmax=.8,square='True')
#corrmat['SalePrice'].nlargest(10)
cols = corrmat.nlargest(10,'SalePrice')['SalePrice'].index
zoomcorr = np.corrcoef(df[cols].values.T)
f, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(zoomcorr,vmax=.8,square=True,annot=True,xticklabels=cols.values, yticklabels=cols.values,annot_kws={'size': 10})
sns.pairplot(df[cols], size = 2.5)
plt.show()
total = df.isnull().sum().sort_values(ascending=False)
percent = (total/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data.sort_values('Percent',ascending=False).head(20)
# drop missing data from our records
df = df.drop(missing_data[missing_data['Total']>1].index,1)
df = df.drop(df.loc[df['Electrical'].isnull()].index,axis=0)
# testing missing values in HOLD OUT TEST DATA 
total = test.isnull().sum().sort_values(ascending=False)
percent = (total/test.isnull().count()).sort_values(ascending=False)
missing_data1 = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data1.sort_values('Percent',ascending=False).head(20)
# Dropping data from the test set to keep a balance 
test = test.drop('PoolQC',axis=1)
test = test.drop('MiscFeature',axis=1)
test = test.drop('Alley',axis=1)
test = test.drop('Fence',axis=1)
test = test.drop('FireplaceQu',axis=1)
test = test.drop('LotFrontage',axis=1)
test = test.drop('GarageFinish',axis=1)
test = test.drop('GarageQual',axis=1)
test = test.drop('GarageType',axis=1)
test = test.drop('GarageYrBlt',axis=1)
test = test.drop('GarageCond',axis=1)
test = test.drop('BsmtExposure',axis=1)
test = test.drop('BsmtFinType2',axis=1)
test = test.drop('BsmtQual',axis=1)
test = test.drop('BsmtCond',axis=1)
test = test.drop('BsmtFinType1',axis=1)
test = test.drop('MasVnrType',axis=1)
test = test.drop('MasVnrArea',axis=1)
#check if there is any missing cell left
df.isnull().sum().max(),test.isnull().sum().max()
df.dtypes.value_counts(),test.dtypes.value_counts()
test = test.fillna(test.median())
#histogram and normal probability plot
from scipy.stats import norm
from scipy import stats
sns.distplot(df['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['SalePrice'], plot=plt)
#df['SalePrice'] = np.log(df['SalePrice'])
#histogram and normal probability plot
#from scipy.stats import norm
#from scipy import stats
#sns.distplot(df['SalePrice'], fit=norm);
#fig = plt.figure()
#res = stats.probplot(df['SalePrice'], plot=plt)
df = df.drop(['Street','Utilities','Condition2','RoofMatl','Heating','Id'], axis=1)
test = test.drop(['Street','Utilities','Condition2','RoofMatl','Heating','Id'], axis=1)
bonf_outlier = [88,462,523,588,632,968,1298,1324]
df = df.drop(bonf_outlier)
train = pd.get_dummies(df,drop_first=True)# use drop_first=True to not include 1st dummy variable this allows us to not keep all dummies in the dataset
cols = train.columns
cols
test_new = pd.get_dummies(test,drop_first=True)
col1 = test_new.columns
col1
cols.difference(col1)
train = train.drop('Electrical_Mix',axis=1)
train = train.drop('Exterior1st_ImStucc',axis=1)
train = train.drop('Exterior1st_Stone',axis=1)
train = train.drop('Exterior2nd_Other',axis=1)
train = train.drop('HouseStyle_2.5Fin',axis=1)
cols = train.columns
col1 = test_new.columns
col1.difference(cols)
#checking number of column types in our new training data set
train.dtypes.value_counts(),test_new.dtypes.value_counts() # so our categorical variables got replaced by the new dummy variables 
df['SaleCondition'].unique() #Just check the unique dummy values
# Creating Training and Target set 
Y = train['SalePrice']
X = train
X = X.drop(['SalePrice'],axis=1)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_train_org, X_test_org, y_train, y_test = train_test_split(X,Y, test_size = 0.07, random_state = 10)

print("The shape of Target set is :"+str(X_train_org.shape))
print("The shape of feature set is :"+str(y_train.shape))
print("The shape of Target set is :"+str(X_test_org.shape))
print("The shape of feature set is :"+str(y_test.shape))
# Running linear regression on the data 
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train_org,y_train)

y_predict = lr.predict(X_test_org)
lr.score(X_test_org,y_test)
coeffecients = pd.DataFrame(lr.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients.head(5)
from sklearn import metrics
import numpy as np
predictions = lr.predict(X_test_org)
print('House prediction dataset')
print('linear model intercept: {}'
     .format(lr.intercept_))
print('R-squared score (training): {:.3f}'
     .format(lr.score(X_train_org, y_train)))
print('R-squared score (test): {:.3f}'
     .format(lr.score(X_test_org, y_test)))
print('MAE for test data set:', metrics.mean_absolute_error(y_test, predictions))
print('MSE for test data set :', metrics.mean_squared_error(y_test, predictions))
print('RMSE for test data set:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
y_predict_df = pd.DataFrame(lr.predict(X_test_org))
y_predict_df.head(5)
from sklearn.linear_model import Ridge

lridge = Ridge(alpha=20.0).fit(X_train_org,y_train)
print("The intercept value is :"+str(lridge.intercept_))
print("The Training set R2 score value is :"+str(lridge.score(X_train_org,y_train)))
print("The Test set R2 score value is :"+str(lridge.score(X_test_org,y_test)))
print("The total coefficients with non zero value :"+str(np.sum(lridge.coef_ !=0)))
y_lridge_prediction = pd.DataFrame(lridge.predict(X_test_org))
y_lridge_prediction.head(5)
print("The MAE to test data :"+str(metrics.mean_absolute_error(y_test,y_lridge_prediction)))
print("The MSE to test data :"+str(metrics.mean_squared_error(y_test,y_lridge_prediction)))
print("The RMSE to test data :"+str(np.sqrt(metrics.mean_squared_error(y_test,y_lridge_prediction))))
print("Evaluating for different alpha values")
for alpha_val in [1,10,20,100,500,1000]:
    ridgereg = Ridge(alpha=alpha_val).fit(X_train_org,y_train)
    r2train = ridgereg.score(X_train_org,y_train)
    r2test  = ridgereg.score(X_test_org,y_test)
    print("Alpha : {}\ R2 score training set : {:.2f} R2 score test set : {:.2f} Non null Coeff : {}\n".format(alpha_val,r2train,r2test,np.sum(ridgereg.coef_!=0)))
# Cross Validation with K Folds 

from sklearn.model_selection import cross_val_score

scores = cross_val_score(lridge,X_train_org,y_train,cv=5)
print("The Score is :"+str(scores))
print("The Mean score is :"+str(scores.mean()))

# Using stratifiedKFold 
from  sklearn.model_selection import StratifiedKFold

scoress = cross_val_score(lridge,X_train_org,y_train,cv=StratifiedKFold(3,random_state=10,shuffle=True))
print("The Stratified Cross-val Score is :"+str(scoress))
print("The Mean Stratified Cross-val score is :"+str(scoress.mean()))
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=80).fit(X_train_org,y_train)
print("Intercept :"+str(lasso.intercept_))
print("R2 score from training set :"+str(lasso.score(X_train_org,y_train)))
print("R2 score from test set :"+str(lasso.score(X_test_org,y_test)))
print("Total Coef without non zero :"+str(np.sum(lasso.coef_!=0)))
y_lasso_prediction = lasso.predict(X_test_org)
print("MAE for test set : "+str(metrics.mean_absolute_error(y_test,y_lasso_prediction)))
print("MSE for test set : "+str(metrics.mean_squared_error(y_test,y_lasso_prediction)))
print("RMSE for test set : "+str(np.sqrt(metrics.mean_squared_error(y_test,y_lasso_prediction))))
print("Trying out different alpha values : ")
for lasso_alpha_val in [1,3,5,10,25,40,60,80,100,200,400,800]:
    lassoreg = Lasso(alpha=lasso_alpha_val).fit(X_train_org,y_train)
    r2scrtrain = lassoreg.score(X_train_org,y_train)
    r2scrtest = lassoreg.score(X_test_org,y_test)
    print("Alpha : {}\ R2 score training set : {:.2f} R2 score test set : {:.2f} Non null Coeff : {}\n".format(lasso_alpha_val,r2scrtrain,r2scrtest,np.sum(lassoreg.coef_!=0)))


predicted_values = pd.DataFrame(y_lasso_prediction)
predicted_values.head(5)
    
cv_score = cross_val_score(lasso,X_train_org,y_train,cv=5)
print("The Cross val scores for Lasso is :"+str(cv_score))
print("The mean cross score efor Lasso is :"+str(cv_score.mean()))

cv_score_strat = cross_val_score(lasso,X_train_org,y_train,cv=StratifiedKFold(5,shuffle=True,random_state=10))
print("The Cross val scores for Lasso is :"+str(cv_score_strat))
print("The mean cross score for Lasso is :"+str(cv_score_strat.mean()))
# Filling the test set data
test_new = test_new.fillna(method='ffill',axis=1)
test_new.isnull().sum().max()
test_new.head()
X_train_org.head()
lassoreg2 = Lasso(alpha=72).fit(X_train_org,y_train)
print("Intercept :"+str(lassoreg2.intercept_))
print("R2 score from training set :"+str(lassoreg2.score(X_train_org,y_train)))
print("R2 score from test set :"+str(lassoreg2.score(X_test_org,y_test)))
print("Total Coef without non zero :"+str(np.sum(lassoreg2.coef_!=0)))
cv_score = cross_val_score(lassoreg2,X_train_org,y_train,cv=5)
print("CV score for Lasso Reg 2 :"+str(cv_score.mean()))
lasso_preds = lassoreg2.predict(test_new)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(max_depth=30, n_estimators=500, max_features = 100, oob_score=True, n_jobs=-1, random_state=1234)
forest_model.fit(X_train_org, y_train)
melb_preds = forest_model.predict(X_test_org)
print("R2 score from training set :"+str(forest_model.score(X_train_org,y_train)))
print("R2 score from test set :"+str(forest_model.score(X_test_org,y_test)))
print("RMSE from test set :"+str(np.sqrt(metrics.mean_squared_error(melb_preds,y_test))))
rm_cv_score = cross_val_score(forest_model,X_train_org,y_train,cv=5,n_jobs=-1)
print('CV score for Random Forrest :'+str(rm_cv_score.mean()))
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.fit_transform(X_test_org)
test_new = scaler.fit_transform(test_new)

from xgboost.sklearn import XGBRegressor
xgb_test1 = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)
xgb_test1.fit(X_train,y_train)
xgb_cv_score1 = cross_val_score(xgb_test1,X_train,y_train, cv = 5, n_jobs = -1)
xgb_preds1 = xgb_test1.predict(X_test)
print("RMSE from test set :"+str(np.sqrt(metrics.mean_squared_error(xgb_preds1,y_test))))
print('CV score for   :'+str(xgb_cv_score1.mean()))
xgb_preds = xgb_test1.predict(test_new)
preds = (xgb_preds+lasso_preds)/2
preds
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice':preds })
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
my_submission.head()