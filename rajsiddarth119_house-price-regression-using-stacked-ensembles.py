import pandas as pd

train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.head(5)
print("train shape is ",train.shape)

print("test shape is ",test.shape)
#Dropping ID

train_data=train.drop(['Id'],axis=1)

test_data=test.drop(['Id'],axis=1)
import matplotlib.pyplot as plt

%matplotlib inline

fig,ax=plt.subplots()

ax.scatter(train_data['GrLivArea'],train_data['SalePrice'])

plt.ylabel('SalePrice')

plt.xlabel('GrLivArea')
#Removing outliers

train_data=train_data.drop(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)].index)
fig,ax=plt.subplots()

ax.scatter(train_data['GrLivArea'],train_data['SalePrice'])

plt.ylabel('SalePrice')

plt.xlabel('GrLivArea')
train_data['SalePrice'].hist()
import seaborn as sns

from scipy.stats import norm

sns.distplot(train_data['SalePrice'],fit=norm)

mean,std=norm.fit(train_data['SalePrice'])

print("mean= ",mean, "and stdev= ",std)
#qq plots

from scipy.stats import probplot

probplot(train['SalePrice'],plot=plt)

plt.show()
#Log transformation

import numpy as np

train_data['SalePrice']=np.log1p(train_data['SalePrice'])

sns.distplot(train_data['SalePrice'],fit=norm)
probplot(train_data['SalePrice'],plot=plt)

plt.show()
#Storing indixes for later

ntrain=train_data.shape[0]

ntest=test_data.shape[0]
ntrain
#Concatanating data

y_train=train_data['SalePrice'].values

train_temp=train_data.drop(['SalePrice'],axis=1)

all_data=pd.concat([train_temp,test_data])

all_data.shape
missing_data=pd.DataFrame(round(all_data.isnull().sum()*100/all_data.shape[0],2),columns=['Missing_fraction']).reset_index()

temp=missing_data.sort_values(['Missing_fraction'],ascending=False).head(20)
f,ax=plt.subplots(figsize=(8,8))

plt.xticks(rotation='90')

sns.barplot(x='index',y='Missing_fraction',data=temp)

plt.xlabel('Features')

plt.title('Percent missing data by feature')
plt.subplots(figsize=(7,7))

sns.heatmap(all_data.corr())
#All features are filled according to their descriptions

#For example if the hose does not have a basement then it is filled as None,and the

#corresponding values of the basement such as Basement square footage are filled with 0.

#Similar approach to other features such as garage

columns_None=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu',

             'GarageType','GarageFinish','GarageQual','GarageCond',

             'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

             'BsmtFinType2','MasVnrType','MSSubClass']



columns_Zero=['GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1',

             'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath',

             'MasVnrArea']



for i in columns_None:

    all_data[i]=all_data[i].fillna("None")



for j in columns_Zero:

    all_data[j]=all_data[j].fillna(0)
all_data['LotFrontage']=all_data.groupby(['Neighborhood'])['LotFrontage'].transform(lambda x:x.fillna(x.median()))
all_data['MSZoning'].value_counts()
all_data['MSZoning'].isnull().sum()
#Filling the missing values with mode

all_data['MSZoning']=all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data['Utilities'].value_counts()
#Only one value.Hence dropping utilities

all_data=all_data.drop(['Utilities'],axis=1)
#Functional : data description says NA means typical

all_data['Functional'].value_counts()
all_data['Functional'].isnull().sum()
all_data['Functional']=all_data['Functional'].fillna("Typ")
all_data['Electrical'].isnull().sum()
all_data['Electrical'].value_counts()
#Filling with mode

all_data['Electrical']=all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'].value_counts()
all_data['KitchenQual'].isnull().sum()
all_data['Exterior1st'].isnull().sum()
all_data['Exterior2nd'].isnull().sum()
all_data['SaleType'].isnull().sum()
for i in ['KitchenQual','Exterior1st','Exterior2nd','SaleType']:

    all_data[i]=all_data[i].fillna(all_data[i].mode()[0])
#checking if any missing values are remaining

all_data.isnull().sum().sum()
#Categorical Variables

for categ in ['MSSubClass','OverallCond','YrSold','MoSold']:

    all_data[categ]=all_data[categ].astype('str')
#Label Encoding

columns=('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



from sklearn.preprocessing import LabelEncoder



for i in columns:

    le=LabelEncoder()

    le.fit(list(all_data[i].values))    

    all_data[i]=le.transform(list(all_data[i].values))
all_data['MoSold'].value_counts()
all_data['Total_SF']=all_data['TotalBsmtSF']+all_data['1stFlrSF']+all_data['2ndFlrSF']
numeric_features=all_data.dtypes[all_data.dtypes!=object].index
from scipy.stats import skew

temp=pd.DataFrame(all_data[numeric_features].apply(lambda x :abs(x.skew())).sort_values(ascending=False),columns=['Skewness']).reset_index()

temp=temp.rename(columns={'index':'features'})

temp.head(10)
#(x^lambda-1/lambda)

skewed_features=temp[temp['Skewness']>0.75]['features'].values

from scipy.special import boxcox1p

for i in skewed_features:

    all_data[i]=boxcox1p(all_data[i],[0.15])
all_data[skewed_features].skew()
all_data=pd.get_dummies(all_data)

all_data.shape
train=all_data[:ntrain]

test=all_data[ntrain:]
from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import Lasso

from sklearn.model_selection import cross_val_score 

#train=all_data[:ntrain]

#test=all_data[ntrain:]



#Hyperparameter tuning for Lasso

rmse=[]

for i in [0.0001,0.001,0.01,0.1]:

    model_lasso=make_pipeline(RobustScaler(),Lasso(alpha=i,random_state=42))

    rmse.append(np.sqrt(-cross_val_score(model_lasso,train,y_train,cv=5,

                 scoring='neg_mean_squared_error')).mean())

print("rmse for lasso is ",min(rmse))    
from sklearn.linear_model import ElasticNet

rmse=[]

model_enet=make_pipeline(RobustScaler(),ElasticNet(alpha=0.0005,l1_ratio=0.9,random_state=42))

rmse.append(np.sqrt(-cross_val_score(model_enet,train,y_train,cv=5,

                 scoring='neg_mean_squared_error')).mean())

print("rmse for enet is ",min(rmse)) 
from sklearn.kernel_ridge import KernelRidge



rmse=[]

model_kridge=make_pipeline(RobustScaler(),KernelRidge(alpha=0.6,kernel='polynomial',degree=2,coef0=2.5))

rmse.append(np.sqrt(-cross_val_score(model_kridge,train,y_train,cv=5,

                 scoring='neg_mean_squared_error')).mean())

print("rmse for kridge is ",min(rmse)) 
from sklearn.ensemble import GradientBoostingRegressor

#Using Huber loss to make it robust to outliers

model_gboost=GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =42)

rmse=[]

rmse.append(np.sqrt(-cross_val_score(model_gboost,train.values,y_train,cv=5,

                 scoring='neg_mean_squared_error')).mean())

print("rmse for Gradient Boosting is ",min(rmse)) 
from xgboost import XGBRegressor

model_xgboost = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1, nthread = -1)

rmse=[]

rmse.append(np.sqrt(-cross_val_score(model_xgboost,train,y_train,cv=5,

                 scoring='neg_mean_squared_error')).mean())

print("rmse for Xgboost is ",min(rmse)) 
from sklearn.base import clone

class averagedmodels():

    

    """ Instantiating the model with various base learners"""

    """ Give base learners as a list"""

    def __init__(self,models):

        self.models=models

    

    def fit(self,X,y):

        self.models_=[clone(x) for x in self.models]

        for model in self.models_:

            model.fit(X,y)

        return self

    

    def predict(self,X):

        predictions=np.column_stack([model.predict(X) for model in self.models_])

        return np.mean(predictions,axis=1)

          
#Instantiating class averaged models

averaged_models = averagedmodels(models = [model_enet,model_xgboost,

                                          model_gboost,model_kridge,model_lasso])

averaged_models.fit(train.values,y_train)

#axis=0 columns,axis=1 row

temp=np.sqrt(np.mean(pow((averaged_models.predict(train.values)-y_train),2)))

print("rmse from averaged models is ",temp)
from sklearn.model_selection import KFold

def Stackedregression(base_models,train,test,meta_learner,nfolds):

    """This function takes train and test data and returns two tables B and C on which we then 

    perform Stacked regression using a metalearner""" 

    kf=KFold(n_splits=nfolds,shuffle=False,random_state=42)

    """We define an empty array table and fill the appropriate values with the help of indices"""

    table_B=np.zeros((train.values.shape[0],len(base_models)))

    table_C=np.zeros((test.values.shape[0],len(base_models)))

    

    """Give base models as a list"""

    for index,model in enumerate(base_models):

        for train_index,holdout_index in kf.split(train):

            

            X_train,X_holdout=train.values[train_index],train.values[holdout_index]

            y_traindata,y_holdout=y_train[train_index],y_train[holdout_index]

            

            #building table B for train

            model_regr=clone(model)

            model_regr.fit(X_train,y_traindata)

            y_pred=model_regr.predict(X_holdout)

            table_B[holdout_index,index]=y_pred

        

        #Building table for C    

        y_predtest=model_regr.predict(test.values)

        table_C[test.index,index]=y_predtest

        

   

            

    table_B=pd.DataFrame(table_B,columns=[i for i in range(len(base_models))])

    table_C=pd.DataFrame(table_C,columns=[i for i in range(len(base_models))])

    table_B=pd.concat([table_B,pd.DataFrame(y_train,columns=['Target'])],axis=1)

    

    """Training the model on tabe B and predicting on table C using meta learner"""

    b_train,b_test=table_B.drop(['Target'],axis=1).values,table_B['Target'].values  

    meta_learner=clone(meta_learner)

    meta_learner.fit(b_train,b_test)

    

    

    """Predicting on table C"""

    

    y_testpredict=meta_learner.predict(table_C.values)

    



   

    return table_B,table_C,y_testpredict  

        

    
tab1,tab2,y3=Stackedregression(base_models=[model_lasso,model_enet,model_kridge,model_gboost],

                  train=train,test=test,nfolds=5,meta_learner=model_xgboost)
temp=pd.read_csv("C:/Users/sid/Desktop/Data_Science/Github_Codes/Kaggle_kernels/Project_2/test.csv")

submit=pd.DataFrame()

submit['Id']=temp['Id']

submit['SalePrice']=y3