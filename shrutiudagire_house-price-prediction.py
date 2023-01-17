import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df_test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
df.shape,df_test.shape
null_values=df.isnull().sum().to_frame()
null_values.columns=['Count of null values']
null_values[null_values['Count of null values']>0]
null_values_test=df_test.isnull().sum().to_frame()
null_values_test.columns=['Count of null values']
null_values_test[null_values_test['Count of null values']>0]
df_combine=pd.concat([df,df_test],axis=0)
df_combine.shape
null=df_combine.isnull().sum().to_frame()
null.columns=['Count of null values']
null[null['Count of null values']>0]
df_combine.loc[df_combine['Alley'].isnull(),'Alley']='No alley access'
df_combine.loc[df_combine['LotFrontage'].isnull(),'LotFrontage']=df_combine['LotFrontage'].median()
df_combine.loc[df_combine['MasVnrType'].isnull(),'MasVnrType']='Other'
df_combine.loc[df_combine['MasVnrArea'].isnull(),'MasVnrArea']=0
df_combine.loc[df_combine['BsmtQual'].isnull(),'BsmtQual']='No Basement'
df_combine.loc[df_combine['BsmtCond'].isnull(),'BsmtCond']='No Basement'
df_combine.loc[df_combine['BsmtExposure'].isnull(),'BsmtExposure']='No Basement'
df_combine.loc[df_combine['BsmtFinType1'].isnull(),'BsmtFinType1']='No Basement'
df_combine.loc[df_combine['BsmtFinType2'].isnull(),'BsmtFinType2']='No Basement'
df_combine.loc[df_combine['FireplaceQu'].isnull(),'FireplaceQu']='No Fireplace'
df_combine.loc[df_combine['GarageType'].isnull(),'GarageType']='No Garage'
df_combine.loc[df_combine['GarageYrBlt'].isnull(),'GarageYrBlt']=0
df_combine.loc[df_combine['GarageFinish'].isnull(),'GarageFinish']='No Garage'
df_combine.loc[df_combine['GarageQual'].isnull(),'GarageQual']='No Garage'
df_combine.loc[df_combine['GarageCond'].isnull(),'GarageCond']='No Garage'
df_combine.loc[df_combine['Electrical'].isnull(),'Electrical']='SBrkr'
null=df_combine.isnull().sum().to_frame()
null.columns=['Count of null values']
null[null['Count of null values']>0]
#A large values are missing for PoolQC hence deleting the column
df_combine['PoolQC'].value_counts()
df_combine.drop(['Id','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
null=df_combine.isnull().sum().to_frame()
null.columns=['Count of null values']
null[null['Count of null values']>0]
df_combine=df_combine.reset_index()
df_combine.drop('index',axis=1,inplace=True)
df_combine
df_combine.loc[df_combine['BsmtFinSF1'].isnull() | df_combine['BsmtFinSF2'].isnull(),'BsmtFinSF1']=0
df_combine.loc[df_combine['BsmtFinSF1'].isnull() | df_combine['BsmtFinSF2'].isnull(),'BsmtFinSF2']=0
df_combine.loc[df_combine['BsmtFullBath'].isnull(),'BsmtFullBath']=df_combine['BsmtFullBath'].mode()[0]
df_combine.loc[df_combine['BsmtHalfBath'].isnull(),'BsmtHalfBath']=df_combine['BsmtHalfBath'].mode()[0]
df_combine.loc[df_combine['BsmtUnfSF'].isnull(),'BsmtUnfSF']=df_combine['BsmtUnfSF'].median()
df_combine.loc[df_combine['Exterior1st'].isnull(),'Exterior1st']=df_combine['Exterior1st'].mode()[0]
df_combine.loc[df_combine['Exterior2nd'].isnull(),'Exterior2nd']=df_combine['Exterior2nd'].mode()[0]
df_combine.loc[df_combine['Functional'].isnull(),'Functional']=df_combine['Functional'].mode()[0]
df_combine.loc[df_combine['GarageArea'].isnull(),'GarageArea']=0
df_combine.loc[df_combine['GarageCars'].isnull(),'GarageCars']=df_combine['GarageCars'].mode()[0]
df_combine.loc[df_combine['KitchenQual'].isnull(),'KitchenQual']=df_combine['KitchenQual'].mode()[0]
df_combine.loc[df_combine['MSZoning'].isnull(),'MSZoning']=df_combine['MSZoning'].mode()[0]
df_combine.loc[df_combine['SaleType'].isnull(),'SaleType']=df_combine['SaleType'].mode()[0]
df_combine.loc[df_combine['TotalBsmtSF'].isnull(),'TotalBsmtSF']=0
df_combine.loc[df_combine['Utilities'].isnull(),'Utilities']=df_combine['Utilities'].mode()[0]
df=df_combine.iloc[0:1460,:].copy(deep=True)
df_test=df_combine.iloc[1460:,:].copy(deep=True)
df.shape,df_test.shape
null_values=df.isnull().sum().to_frame()
null_values.columns=['Count of null values']
null_values[null_values['Count of null values']>0]
null_values_test=df_test.isnull().sum().to_frame()
null_values_test.columns=['Count of null values']
null_values_test[null_values_test['Count of null values']>0]
df.head()
numeric=['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd','BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'GarageYrBlt', 'GarageArea',
        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'MiscVal']

cat=['MSSubClass', 'MSZoning','Street', 'Alley','LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','GarageCars',
       'BedroomAbvGr', 'KitchenAbvGr','TotRmsAbvGrd',
       'OverallQual', 'OverallCond', 'RoofStyle','RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', 'KitchenQual','Functional', 'FireplaceQu', 'GarageType',
       'GarageFinish', 'GarageQual', 'GarageCond', 'MoSold', 'YrSold','Fireplaces',
       'PavedDrive', 'SaleType','SaleCondition']
len(numeric),len(cat)
for i in cat:
    plt.figure(figsize=(20,5))
    plt.subplot(1,2,1)
    plt.title(i)
    ((df[i].value_counts()/df.shape[0])*100).plot(kind='bar')
    plt.subplot(1,2,2)
#     df.boxplot(column='SalePrice',by=i)
    sns.boxplot(df[i],df['SalePrice'])
    plt.xticks(rotation=90)
    plt.show()

for i in numeric:
    sns.scatterplot(df[i],df['SalePrice'])
    plt.show()
corrdf=pd.DataFrame(df.corr()['SalePrice'])
corrdf
sns.boxplot(df['SalePrice'])
df['SalePrice'].plot(kind='kde')
# lets apply log transformation as it is right skewed
df['SalePrice']=np.log(df['SalePrice'])
sns.boxplot(df['SalePrice'])
df['SalePrice'].skew()
df['SalePrice'].plot(kind='kde')
df_combine=pd.concat([df,df_test],axis=0)
df_combine=df_combine.reset_index()
df_combine.drop('index',axis=1,inplace=True)
df_combine
df_dummies1=pd.DataFrame()
for i in cat:
    df_dummies=pd.DataFrame()
    df_dummies=pd.get_dummies(df_combine[i],prefix=i,drop_first=True)
    df_dummies1=pd.concat([df_dummies1,df_dummies],axis=1)
    
df_dummies1.head()
for i in cat:
    df_combine.drop(i,axis=1,inplace=True)
df_combine.columns
df_final=pd.concat([df_dummies1,df_combine],axis=1)
df_final
df=df_final.iloc[0:1460,:].copy(deep=True)
df_test=df_final.iloc[1460:,:].copy(deep=True)
df.shape,df_test.shape
X=df.drop('SalePrice',axis=1)
Y=df['SalePrice']
from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()
x1=minmax.fit_transform(X)
df_multi_scaled=pd.DataFrame(x1,columns=X.columns)
df_multi_scaled.head()
X_test=df_test.drop('SalePrice',axis=1)
Y_test=df_test['SalePrice']
x2=minmax.transform(X_test)
df_multi_scaled_test=pd.DataFrame(x2,columns=X_test.columns)
# df_multi_scaled_test.head()
import warnings 
warnings.filterwarnings('ignore')
import statsmodels.api as sm

X_constant = sm.add_constant(df_multi_scaled)
lin_reg = sm.OLS(Y,X_constant).fit()
lin_reg.summary()
cols = list(df_multi_scaled.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = df_multi_scaled[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(Y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

len(selected_features_BE)
dfnew=df_multi_scaled[selected_features_BE]
dftestnew=df_multi_scaled_test[selected_features_BE]
df_combine=pd.concat([dfnew,dftestnew])
df_combine=df_combine.reset_index()
df_combine.drop('index',axis=1,inplace=True)
df_combine
from statsmodels.stats.outliers_influence import variance_inflation_factor
[variance_inflation_factor(df_combine.values, j) for j in range(1, df_combine.shape[1])]
def calculate_vif(x):
    thresh = 5.0
    output = pd.DataFrame()
    k = x.shape[1]
    vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
    for i in range(1,k):
        a = np.argmax(vif)
        if vif[a] <= thresh :
            break
        if i == 1 :          
            output = x.drop(x.columns[a], axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
        elif i > 1 :
            output = output.drop(output.columns[a],axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
    return(output)
Xnew=calculate_vif(df_combine)
Xnew.head()
from statsmodels.stats.outliers_influence import variance_inflation_factor
[variance_inflation_factor(Xnew.values, j) for j in range(1, Xnew.shape[1])]
df=Xnew.iloc[0:1460,:].copy(deep=True)
df_test=Xnew.iloc[1460:,:].copy(deep=True)
df.shape,df_test.shape
import warnings 
warnings.filterwarnings('ignore')
import statsmodels.api as sm

X_constant = sm.add_constant(df)
lin_reg = sm.OLS(Y,X_constant).fit()
lin_reg.summary()
cols = list(df.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = df[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(Y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

df=df[selected_features_BE]
df_test=df_test[selected_features_BE]

from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.linear_model import Ridge,ElasticNet,Lasso
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score
variance=[]
means=[]
for n in np.arange(0.001,1,0.001):
    ridge=Ridge(alpha=n,random_state=0)
    kfold = model_selection.KFold(shuffle=True,n_splits=3,random_state=0)
    scores=model_selection.cross_val_score(ridge,df,Y,cv=kfold,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    means.append(np.mean(rmse))
    variance.append(np.std(rmse,ddof=1))
x_axis=np.arange(0.001,1,0.001)
plt.plot(x_axis,variance) 

np.argmin(variance),means[np.argmin(variance)],variance[np.argmin(variance)]
variance_lasso=[]
means_lasso=[]
for n in np.arange(0.001,1,0.0001):
    lasso=Lasso(alpha=n,random_state=0)
    kfold = model_selection.KFold(shuffle=True,n_splits=3,random_state=0)
    scores=cross_val_score(lasso,df,Y,cv=kfold,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    means_lasso.append(np.mean(rmse))
    variance_lasso.append(np.std(rmse,ddof=1))
x_axis=np.arange(0.001,1,0.0001)
plt.plot(x_axis,variance_lasso) 

np.argmin(variance_lasso),means_lasso[np.argmin(variance_lasso)],variance_lasso[np.argmin(variance_lasso)]
np.argmin(means_lasso),means_lasso[np.argmin(means_lasso)],variance_lasso[np.argmin(means_lasso)]
from sklearn.linear_model import ElasticNetCV, ElasticNet
cv_model = ElasticNetCV(l1_ratio=np.arange(0.001,1,0.001),n_jobs=-1, random_state=0)
cv_model.fit(df,Y)
print('Optimal alpha: %.8f'%cv_model.alpha_)
print('Optimal l1_ratio: %.3f'%cv_model.l1_ratio_)
print('Number of iterations %d'%cv_model.n_iter_)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
#3)Decision tree
DT=DecisionTreeRegressor()
from sklearn.model_selection import RandomizedSearchCV
params={'max_depth':np.arange(5,150),
       'min_samples_leaf':np.arange(5,50),
       'min_samples_split':np.arange(5,50)
       }
gsearch=RandomizedSearchCV(DT,param_distributions=params,cv=3,scoring='neg_mean_squared_error',random_state=0)
gsearch.fit(df,Y)
gsearch.best_params_
# 3)tunning k value
knn=KNeighborsRegressor()
knn_params={'n_neighbors':np.arange(1,100),'weights':['uniform','distance']}
randomsearch=RandomizedSearchCV(knn,knn_params,cv=3,scoring='neg_mean_squared_error',random_state=0)
randomsearch.fit(df,Y)
randomsearch.best_params_
means_knn=[]
variance_knn=[]
for n in np.arange(1,100):
    KNN=KNeighborsRegressor(weights='distance',n_neighbors=n)
    kfold = model_selection.KFold(shuffle=True,n_splits=3,random_state=0)
    value=model_selection.cross_val_score(KNN,df,Y,cv=kfold,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(value))
    variance_knn.append(np.std(rmse,ddof=1))
    means_knn.append(np.mean(rmse))
    
x_axis=np.arange(len(means_knn))
plt.plot(x_axis,means_knn)
np.argmin(means_knn),means_knn[np.argmin(means_knn)],variance_knn[np.argmin(means_knn)]
np.argmin(variance_knn),means_knn[np.argmin(variance_knn)],variance_knn[np.argmin(variance_knn)]

variance_rf=[]
means_rf=[]
for n in np.arange(1,100):
    RF=RandomForestRegressor(criterion='mse',n_estimators=n,random_state=0)
    kfold = model_selection.KFold(shuffle=True,n_splits=3,random_state=0)
    scores=cross_val_score(RF,df,Y,cv=kfold,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    means_rf.append(np.mean(rmse))
    variance_rf.append(np.std(rmse,ddof=1))
    

x_axis=np.arange(len(variance_rf))
plt.plot(x_axis,variance_rf) 
np.argmin(variance_rf),means_rf[ np.argmin(variance_rf)],np.min(variance_rf)
np.argmin(means_rf),means_rf[np.argmin(means_rf)],variance_rf[np.argmin(means_rf)]
from sklearn.ensemble import BaggingRegressor
means_Bag_DT=[]
variance_Bag_DT=[]
for n in np.arange(1,200):
    Bag=BaggingRegressor(n_estimators=n,random_state=0)
    kfold = model_selection.KFold(shuffle=True,n_splits=3,random_state=0)
    scores=cross_val_score(Bag,df,Y,cv=kfold,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    means_Bag_DT.append(np.mean(rmse))
    variance_Bag_DT.append(np.std(rmse,ddof=1))
x_axis=np.arange(len(variance_Bag_DT))
plt.plot(x_axis,means_Bag_DT)
np.argmin(variance_Bag_DT),means_Bag_DT[np.argmin(variance_Bag_DT)],variance_Bag_DT[np.argmin(variance_Bag_DT)]
np.argmin(means_Bag_DT),means_Bag_DT[np.argmin(means_Bag_DT)],variance_Bag_DT[np.argmin(means_Bag_DT)]
knn_cust=KNeighborsRegressor(weights='uniform',n_neighbors=6,n_jobs=-1)
means_Bag_knngrid=[]
variance_Bag_knngrid=[]
for n in np.arange(1,50):
    Bag=BaggingRegressor(base_estimator=knn_cust,n_estimators=n,random_state=0)
    kfold = model_selection.KFold(shuffle=True,n_splits=3,random_state=0)
    scores=cross_val_score(Bag,df,Y,cv=kfold,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    means_Bag_knngrid.append(np.mean(rmse))
    variance_Bag_knngrid.append(np.std(rmse,ddof=1))
x_axis=np.arange(len(variance_Bag_knngrid))
plt.plot(x_axis,variance_Bag_knngrid)
np.argmin(variance_Bag_knngrid),means_Bag_knngrid[np.argmin(variance_Bag_knngrid)],variance_Bag_knngrid[np.argmin(variance_Bag_DT)]
np.argmin(means_Bag_knngrid),means_Bag_knngrid[np.argmin(means_Bag_knngrid)],variance_Bag_knngrid[np.argmin(means_Bag_knngrid)]
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,VotingRegressor

rmse_ada_DT=[]
variance_ada_DT=[]
for n in np.arange(1,100):
    AB=AdaBoostRegressor(n_estimators=n,random_state=0)
    kfold = model_selection.KFold(shuffle=True,n_splits=3,random_state=0)
    scores=cross_val_score(AB,df,Y,cv=kfold,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    rmse_ada_DT.append(np.mean(rmse))
    variance_ada_DT.append((np.std(rmse,ddof=1)))
x_axis=np.arange(len(rmse_ada_DT))
plt.plot(x_axis,rmse_ada_DT)
np.argmin(rmse_ada_DT),rmse_ada_DT[np.argmin(rmse_ada_DT)],variance_ada_DT[np.argmin(rmse_ada_DT)]
np.argmin(variance_ada_DT),rmse_ada_DT[np.argmin(variance_ada_DT)],variance_ada_DT[np.argmin(variance_ada_DT)]
# RandomForest=RandomForestRegressor(n_estimators=6,random_state=0,n_jobs=-1)
# rmse_ada_RandomForest=[]
# variance_ada_RandomForest=[]
# for n in np.arange(1,100):
#     AB=AdaBoostRegressor(base_estimator=RandomForest,n_estimators=n,random_state=0)
#     kfold = model_selection.KFold(shuffle=True,n_splits=3,random_state=0)
#     scores=cross_val_score(AB,df,Y,cv=kfold,scoring='neg_mean_squared_error')
#     rmse=np.sqrt(np.abs(scores))
#     rmse_ada_RandomForest.append(np.mean(rmse))
#     variance_ada_RandomForest.append((np.std(rmse,ddof=1)))
# x_axis=np.arange(len(rmse_ada_RandomForest))
# plt.plot(x_axis,rmse_ada_RandomForest)
# np.argmin(rmse_ada_RandomForest),rmse_ada_RandomForest[np.argmin(rmse_ada_RandomForest)],variance_ada_RandomForest[np.argmin(rmse_ada_RandomForest)]
# np.argmin(variance_ada_RandomForest),rmse_ada_RandomForest[np.argmin(variance_ada_RandomForest)],variance_ada_RandomForest[np.argmin(variance_ada_RandomForest)]

# Best estimator for RandomForest is 98
LR=LinearRegression()
rmse_ada_GB=[]
variance_ada_GB=[]
for n in np.arange(1,260):
    GB=GradientBoostingRegressor(n_estimators=n,random_state=0)
    kfold = model_selection.KFold(shuffle=True,n_splits=3,random_state=0)
    scores=cross_val_score(GB,df,Y,cv=kfold,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    rmse_ada_GB.append(np.mean(rmse))
    variance_ada_GB.append((np.std(rmse,ddof=1)))
x_axis=np.arange(len(rmse_ada_GB))
plt.plot(x_axis,rmse_ada_GB)
np.argmin(rmse_ada_GB),rmse_ada_GB[np.argmin(rmse_ada_GB)],variance_ada_GB[np.argmin(rmse_ada_GB)]
np.argmin(variance_ada_GB),rmse_ada_GB[np.argmin(variance_ada_GB)],variance_ada_GB[np.argmin(variance_ada_GB)]
ridge=Ridge(alpha=0.107,random_state=0)
elasticnet = ElasticNet(l1_ratio=0.059, alpha =0.00178473,  fit_intercept=True)
FGDT=DecisionTreeRegressor()
DT=DecisionTreeRegressor(min_samples_split=30,min_samples_leaf=9,max_depth=79,random_state=0)
RandomForest=RandomForestRegressor(n_estimators=99,random_state=0,n_jobs=-1)
knn_grid=KNeighborsRegressor(weights='uniform',n_neighbors=7,n_jobs=-1)
knn_cust=KNeighborsRegressor(weights='distance',n_neighbors=6,n_jobs=-1)
Bag_dt=BaggingRegressor(n_estimators=6,random_state=0)
boost_dt=AdaBoostRegressor(n_estimators=49,random_state=0)
boost_randomForest=AdaBoostRegressor(base_estimator=RandomForest,n_estimators=98,random_state=0)
gradientboost=GradientBoostingRegressor(n_estimators=259,random_state=0)
stacked_1 = VotingRegressor(estimators = [('Elasticnet', elasticnet),('Gradientboost',gradientboost), ('AdaBoost Default', boost_dt)])
stacked_2 = VotingRegressor(estimators = [('Elasticnet', elasticnet),('Gradientboost',gradientboost), ('Ridge', ridge)])



models = []
models.append(('Ridge', ridge))
models.append(('ElasticNet',elasticnet))
models.append(('FGDT',FGDT))
models.append(('DT',DT))
models.append(('RandomForest',RandomForest))
models.append(('KNN GRID',knn_grid))
models.append(('KNN cust',knn_cust))
models.append(('Bagged DT',Bag_dt))
models.append(('ADABoost DT',boost_dt))
models.append(('ADABoost RF',boost_randomForest))
models.append(('Gradient Boosting',gradientboost))
models.append(('Stacking-elastic net,gradient boost and AdaBoostDT',stacked_1))
models.append(('Stacking-elastic net,gradient boost and Ridge',stacked_2))

# After removing multicollinearity lets perform linear regression and then RIDGE and  LASSO
means=[]
rmse=[]
names=[]
variance=[]
df_result=pd.DataFrame()
for name,model in models:
    kfold=model_selection.KFold(shuffle=True,n_splits=3,random_state=0)
    cv_result=model_selection.cross_val_score(model,df,Y,cv=kfold,scoring='neg_mean_squared_error')
    value=np.sqrt(np.abs(cv_result))
    print(value)
    means.append(value)
    names.append(name)
    rmse.append(np.mean(value))
    variance.append(np.std((value),ddof=1))
df_result['Names']=names
df_result['RMSE']=rmse
df_result['Variance']=variance
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(means)
ax.set_xticklabels(names)
plt.xticks(rotation=90)
plt.show()
df_result

df_result.sort_values(by='RMSE')
df_result.sort_values(by='Variance')
stacked_2.fit(df,Y)
predicted_values=stacked_2.predict(df_test)
pred=np.exp(predicted_values)
result=pd.DataFrame(pred)
result
