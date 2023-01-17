

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.legend_handler import HandlerBase
import seaborn as sns
import missingno as msno
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
%matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import the necessary modelling algos.

#classifiaction.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV,ElasticNet
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#model selection
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#evaluation metrics
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classification

from scipy import stats
from scipy.stats import norm, skew   # specifically for staistics
train=pd.read_csv(r'../input/train.csv')
test=pd.read_csv(r'../input/test.csv')
train.head(10)
#test.head(10)

df=train.copy()
#df.head(10)
df.shape
df.drop(['Id'],axis=1,inplace=True)
test.drop(['Id'],axis=1,inplace=True)
df.index # the indices of the rows.
df.columns 
df.isnull().any()
msno.matrix(df) # just to visulaize. 

cat_df=df.select_dtypes(include='object')
cat_df.head(10)
cat_df.shape
cat_df.columns   # list of the categorical columns.
num_df=df.select_dtypes(include='number')
num_df.shape
num_df.columns # list of numeric columns.
nan_df=df.loc[:, df.isna().any()]
nan_df.shape
nan_df.columns   # list of columns with missing values.
all_data=pd.concat([train,test])
print(all_data.shape)
all_data = all_data.reset_index(drop=True)
all_data.head(10)
print(all_data.loc[1461:,'SalePrice'])  
# note that it is Nan for the values in test set as expected. so we drop it here for now.
all_data.drop(['SalePrice'],axis=1,inplace=True)

# analyzing the target variable ie 'Saleprice'
sns.distplot(a=df['SalePrice'],color='#ff4125',axlabel=False).set_title('Sale Price')
#Get also the qq-plot (the quantile-quantile plot)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
df['SalePrice']=np.log1p(df['SalePrice']) 
# now again see the distribution.
sns.distplot(a=df['SalePrice'],color='#ff4125',axlabel=False).set_title('log(1+SalePrice)')  # better.


cor_mat= df[:].corr()
cor_with_tar=cor_mat.sort_values(['SalePrice'],ascending=False)
print("The most relevant features (numeric) for the target are :")
cor_with_tar.SalePrice
# using a corelation map to visualize features with high corelation.
cor_mat= df[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath',
             'YearBuilt','YearRemodAdd','GarageYrBlt','TotRmsAbvGrd','SalePrice']].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)

# some inference section.
def plot_num(feature):
    fig,axes=plt.subplots(1,2)
    sns.boxplot(data=df,x=feature,ax=axes[0])
    sns.distplot(a=df[feature],ax=axes[1],color='#ff4125')
    fig.set_size_inches(15,5)
plot_num('GrLivArea')
plot_num('GarageArea')
plot_num('TotalBsmtSF') 
def plot_cat(feature):
  sns.countplot(data=df,x=feature)
  ax=sns.countplot(data=df,x=feature)
   
plot_cat('OverallQual')
plot_cat('FullBath')
plot_cat('YearBuilt')
plot_cat('TotRmsAbvGrd') # most of the houses have 5-7 rooms above the grd floor.
plot_cat('GarageCars')
sns.factorplot(data=df,x='Neighborhood',kind='count',size=10,aspect=1.5)
fig, ax = plt.subplots()
ax.scatter(x = df['GrLivArea'], y = df['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()
df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<13)].index) # removing some outliers on lower right side.
# again checking
fig, ax = plt.subplots()
ax.scatter(x = df['GrLivArea'], y = df['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()
# garage area
fig, ax = plt.subplots()
ax.scatter(x =(df['GarageArea']), y = df['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GarageArea')
plt.show()
# can try to fremove the points with gargae rea > than 1200.
# basment area
fig, ax = plt.subplots()
ax.scatter(x =(df['TotalBsmtSF']), y = df['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('TotalBsmtSF')
plt.show()   # check >3000 can leave here.
#overall qual
sns.factorplot(data=df,x='OverallQual',y='SalePrice',kind='box',size=5,aspect=1.5)
#garage cars
sns.factorplot(data=df,x='GarageCars',y='SalePrice',kind='box',size=5,aspect=1.5)
#no of rooms
sns.factorplot(data=df,x='TotRmsAbvGrd',y='SalePrice',kind='bar',size=5,aspect=1.5) # increasing rooms imply increasing SalePrice as expected.
#neighborhood
sns.factorplot(data=df,x='Neighborhood',y='SalePrice',kind='box',size=10,aspect=1.5)
#sale conditioin
sns.factorplot(data=df,x='SaleCondition',y='SalePrice',kind='box',size=10,aspect=1.5)

nan_all_data = (all_data.isnull().sum())
nan_all_data= nan_all_data.drop(nan_all_data[nan_all_data== 0].index).sort_values(ascending=False)
nan_all_data
miss_df = pd.DataFrame({'Missing Ratio' :nan_all_data})
miss_df

#delet some features withvery high number of missing values.  
all_data.drop(['PoolQC','Alley','Fence','Id','MiscFeature'],axis=1,inplace=True)

test.drop(['PoolQC','Alley','Fence','MiscFeature'],axis=1,inplace=True)
df.drop(['PoolQC','Alley','Fence','MiscFeature'],axis=1,inplace=True)
# FireplaceQu
# it is useful but many of the values nearly half are missing makes no sense to fill half of the values. so deleting this
all_data.drop(['FireplaceQu'],axis=1,inplace=True)
test.drop(['FireplaceQu'],axis=1,inplace=True)
df.drop(['FireplaceQu'],axis=1,inplace=True)

#Lot Frontage
print(df['LotFrontage'].dtype)
plt.scatter(x=np.log1p(df['LotFrontage']),y=df['SalePrice'])
cr=df.corr()
print(df['LotFrontage'].describe())
print("The corelation of the LotFrontage with the Target : " , cr.loc['LotFrontage','SalePrice'])

all_data['LotFrontage'].fillna(np.mean(all_data['LotFrontage']),inplace=True)
all_data['LotFrontage'].isna().sum()
#Garage  related features.
# these features eg like garage qual,cond,finish,type seems to be important and relevant for buying car. 
# hence I will not drop these features insted i will fill them with the 'none' for categorical and 0 for numeric as nan here implies that there is no garage.

all_data['GarageYrBlt'].fillna(0,inplace=True)
print(all_data['GarageYrBlt'].isnull().sum())

all_data['GarageArea'].fillna(0,inplace=True)
print(all_data['GarageArea'].isnull().sum())

all_data['GarageCars'].fillna(0,inplace=True)
print(all_data['GarageCars'].isnull().sum())

all_data['GarageQual'].fillna('None',inplace=True)   # creating a separate category 'none' which means no garage.
print(all_data['GarageQual'].isnull().sum())

all_data['GarageFinish'].fillna('None',inplace=True)   # creating a separate category 'none' which means no garage.
print(all_data['GarageFinish'].isnull().sum())

all_data['GarageCond'].fillna('None',inplace=True)   # creating a separate category 'none' which means no garage.
print(all_data['GarageCond'].isnull().sum())

all_data['GarageType'].fillna('None',inplace=True)   # creating a separate category 'none' which means no garage.
print(all_data['GarageType'].isnull().sum())


# basement related features.
#missing values are likely zero for having no basement

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col].fillna(0,inplace=True)
    
# for categorical features we will create a separate class 'none' as before.

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col].fillna('None',inplace=True)
    
print(all_data['TotalBsmtSF'].isnull().sum())


# MasVnrArea 0 and MasVnrType 'None'.
all_data['MasVnrArea'].fillna(0,inplace=True)
print(all_data['MasVnrArea'].isnull().sum())

all_data['MasVnrType'].fillna('None',inplace=True)
print(all_data['MasVnrType'].isnull().sum())
#MSZoning.
# Here nan does not mean no so I will with the most common one ie the mode.
all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0],inplace=True)
print(all_data['MSZoning'].isnull().sum())
# utilities
sns.factorplot(data=df,kind='box',x='Utilities',y='SalePrice',size=5,aspect=1.5)
all_data.drop(['Utilities'],axis=1,inplace=True)
#functional
# fill with mode
all_data['Functional'].fillna(all_data['Functional'].mode()[0],inplace=True)
print(all_data['Functional'].isnull().sum())
# other rem columns rae all cat like kitchen qual etc.. and so filled with mode.
for col in ['SaleType','KitchenQual','Exterior2nd','Exterior1st','Electrical']:
  all_data[col].fillna(all_data[col].mode()[0],inplace=True)
  print(all_data[col].isnull().sum())
nan_all_data = (all_data.isnull().sum())
nan_all_data= nan_all_data.drop(nan_all_data[nan_all_data== 0].index).sort_values(ascending=False)
nan_all_data
miss_df = pd.DataFrame({'Missing Ratio' :nan_all_data})
miss_df


all_data.shape

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.50]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

for col in all_data.columns:
    if(all_data[col].dtype == 'object'):
        le=LabelEncoder()
        all_data[col]=le.fit_transform(all_data[col])
train=all_data.loc[:(df.shape)[0]+2,:]
test=all_data.loc[(df.shape)[0]+2:,:]
train['SalePrice']=df['SalePrice']
train['SalePrice'].fillna(np.mean(train['SalePrice']),inplace=True)
train.shape
print(train['SalePrice'].isnull().sum())
print(train.shape)
print(test.shape)
x_train,x_test,y_train,y_test=train_test_split(train.drop(['SalePrice'],axis=1),train['SalePrice'],test_size=0.20,random_state=42)

reg_lin=LinearRegression()
reg_lin.fit(x_train,y_train)
pred=reg_lin.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred)))
reg_lasso=Lasso()
reg_lasso.fit(x_train,y_train)
pred=reg_lasso.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred)))
params_dict={'alpha':[0.001, 0.005, 0.01,0.05,0.1,0.5,1]}
reg_lasso_CV=GridSearchCV(estimator=Lasso(),param_grid=params_dict,scoring='neg_mean_squared_error',cv=10)
reg_lasso_CV.fit(x_train,y_train)
pred=reg_lasso_CV.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred)))
reg_lasso_CV.best_params_
reg_ridge=Ridge()
reg_ridge.fit(x_train,y_train)
pred=reg_ridge.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred)))
params_dict={'alpha':[0.1, 0.15, 0.20,0.25,0.30,0.35,0.4,0.45,0.50,0.55,0.60]}
reg_ridge_CV=GridSearchCV(estimator=Ridge(),param_grid=params_dict,scoring='neg_mean_squared_error',cv=10)
reg_ridge_CV.fit(x_train,y_train)
pred=reg_ridge_CV.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred)))
reg_ridge_CV.best_params_
#the params are tuned with grid searchCV.

reg_gb=GradientBoostingRegressor(n_estimators=2000,learning_rate=0.05,max_depth=3,min_samples_split=10,max_features='sqrt',subsample=0.75 ,loss='huber')
reg_gb.fit(x_train,y_train)
pred=reg_gb.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred)))
import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(x_train,y_train)
pred=model_xgb.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred)))
# predictions on the test set.
 
pred=reg_gb.predict(test)
pred_act=np.exp(pred)
pred_act=pred_act-1
len(pred_act)
test_id=[]
for i in range (1461,2920):
    test_id.append(i)
d={'Id':test_id,'SalePrice':pred_act}
ans_df=pd.DataFrame(d)
ans_df.head(10)
ans_df.to_csv('answer.csv',index=False)
