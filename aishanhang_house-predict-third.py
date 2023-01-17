import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import os
print(os.listdir("../input"))
plt.style.use('ggplot')
#导入包
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline,make_pipeline
from scipy.stats import skew
from sklearn.decomposition import PCA,KernelPCA
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score,GridSearchCV,KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet,SGDRegressor,BayesianRidge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.svm import SVR,LinearSVR
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
submission=pd.read_csv('../input/sample_submission.csv')
#探索可视化
plt.figure(figsize=(15,8))
sns.boxplot(train.YearBuilt,train.SalePrice)
plt.figure(figsize=(12,6))
plt.scatter(x=train.GrLivArea,y=train.SalePrice)
plt.xlabel('GrLivArea',fontsize=13)
plt.ylabel('SalePrice',fontsize=13)
plt.ylim(0,800000)
#显示有两个离群点，删除
#删除离群点
train.drop(train[(train['SalePrice']<300000)&(train['GrLivArea']>4000)].index,inplace=True)
train.shape
full=pd.concat([train,test],ignore_index=True)
full.shape
full.drop(['Id'],axis=1,inplace=True)
full.shape
#数据清洗
aa=full.isnull().sum()
aa[aa>0].sort_values(ascending=False)
#首先输入LotFrontage,根据LotArea 和 Neighborhood 的中位数。由于LotArea是连续特征，使用qcut将其分为10份
full.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean','median','count'])
full['LotAreaCut']=pd.qcut(full.LotArea,10)
full.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean','median','count'])
full['LotFrontage']=full.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x:x.fillna(x.median()))
full['LotFrontage']=full.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x:x.fillna(x.median()))
#由于LotAreaCut和Neighborhood组合还有空值没有被包括，所以单独采用LotAreaCut再分组
#接下来根据数据描述来填充其他缺失值
#这些都是数值型数据
cols=['MasVnrArea','BsmtUnfSF','TotalBsmtSF','GarageCars','BsmtFinSF2','BsmtFinSF1','GarageArea']
for col in cols:
    full[col].fillna(0,inplace=True)
cols1=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageCond','GarageQual','GarageFinish','GarageYrBlt','GarageType','BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1','MasVnrType']
#缺失的是类别数据
for col in cols1:
    full[col].fillna('None',inplace=True)
#根据模式填充
cols2=['MSZoning','BsmtFullBath','BsmtHalfBath','Utilities','Functional','Electrical','KitchenQual','SaleType','Exterior1st','Exterior2nd']
for col in cols2:
    full[col].fillna(full[col].mode()[0],inplace=True)
full.isnull().sum()[full.isnull().sum()>0]
#full['LowQualFinSF'].unique()
#特征工程
#1.转换一些数值特征为类别特征。然后对这些特征使用LabelEncoder 和 get_dummies   ,"LowQualFinSF"
NumStr=['MSSubClass','BsmtFullBath','BsmtHalfBath','HalfBath','BedroomAbvGr','KitchenAbvGr','MoSold','YrSold','YearBuilt','YearRemodAdd','GarageYrBlt']
for col in NumStr:
    full[col]=full[col].astype(str)
full.groupby(['HouseStyle'])[['SalePrice']].agg(['mean','median','count'])
def map_values():
    full["oMSSubClass"] = full.MSSubClass.map({'180':1, 
                                        '30':2, '45':2, 
                                        '190':3, '50':3, '90':3, 
                                        '85':4, '40':4, '160':4, 
                                        '70':5, '20':5, '75':5, '80':5, '150':5,
                                        '120': 6, '60':6})
    
    full["oMSZoning"] = full.MSZoning.map({'C (all)':1, 'RH':2, 'RM':2, 'RL':3, 'FV':4})
    
    full["oNeighborhood"] = full.Neighborhood.map({'MeadowV':1,
                                               'IDOTRR':2, 'BrDale':2,
                                               'OldTown':3, 'Edwards':3, 'BrkSide':3,
                                               'Sawyer':4, 'Blueste':4, 'SWISU':4, 'NAmes':4,
                                               'NPkVill':5, 'Mitchel':5,
                                               'SawyerW':6, 'Gilbert':6, 'NWAmes':6,
                                               'Blmngtn':7, 'CollgCr':7, 'ClearCr':7, 'Crawfor':7,
                                               'Veenker':8, 'Somerst':8, 'Timber':8,
                                               'StoneBr':9,
                                               'NoRidge':10, 'NridgHt':10})
    
    full["oCondition1"] = full.Condition1.map({'Artery':1,
                                           'Feedr':2, 'RRAe':2,
                                           'Norm':3, 'RRAn':3,
                                           'PosN':4, 'RRNe':4,
                                           'PosA':5 ,'RRNn':5})
    
    full["oBldgType"] = full.BldgType.map({'2fmCon':1, 'Duplex':1, 'Twnhs':1, '1Fam':2, 'TwnhsE':2})
    
    full["oHouseStyle"] = full.HouseStyle.map({'1.5Unf':1, 
                                           '1.5Fin':2, '2.5Unf':2, 'SFoyer':2, 
                                           '1Story':3, 'SLvl':3,
                                           '2Story':4, '2.5Fin':4})
    
    full["oExterior1st"] = full.Exterior1st.map({'BrkComm':1,
                                             'AsphShn':2, 'CBlock':2, 'AsbShng':2,
                                             'WdShing':3, 'Wd Sdng':3, 'MetalSd':3, 'Stucco':3, 'HdBoard':3,
                                             'BrkFace':4, 'Plywood':4,
                                             'VinylSd':5,
                                             'CemntBd':6,
                                             'Stone':7, 'ImStucc':7})
    
    full["oMasVnrType"] = full.MasVnrType.map({'BrkCmn':1, 'None':1, 'BrkFace':2, 'Stone':3})
    
    full["oExterQual"] = full.ExterQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
    
    full["oFoundation"] = full.Foundation.map({'Slab':1, 
                                           'BrkTil':2, 'CBlock':2, 'Stone':2,
                                           'Wood':3, 'PConc':4})
    
    full["oBsmtQual"] = full.BsmtQual.map({'Fa':2, 'None':1, 'TA':3, 'Gd':4, 'Ex':5})
    
    full["oBsmtExposure"] = full.BsmtExposure.map({'None':1, 'No':2, 'Av':3, 'Mn':3, 'Gd':4})
    
    full["oHeating"] = full.Heating.map({'Floor':1, 'Grav':1, 'Wall':2, 'OthW':3, 'GasW':4, 'GasA':5})
    
    full["oHeatingQC"] = full.HeatingQC.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    
    full["oKitchenQual"] = full.KitchenQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
    
    full["oFunctional"] = full.Functional.map({'Maj2':1, 'Maj1':2, 'Min1':2, 'Min2':2, 'Mod':2, 'Sev':2, 'Typ':3})
    
    full["oFireplaceQu"] = full.FireplaceQu.map({'None':1, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    
    full["oGarageType"] = full.GarageType.map({'CarPort':1, 'None':1,
                                           'Detchd':2,
                                           '2Types':3, 'Basment':3,
                                           'Attchd':4, 'BuiltIn':5})
    
    full["oGarageFinish"] = full.GarageFinish.map({'None':1, 'Unf':2, 'RFn':3, 'Fin':4})
    
    full["oPavedDrive"] = full.PavedDrive.map({'N':1, 'P':2, 'Y':3})
    
    full["oSaleType"] = full.SaleType.map({'COD':1, 'ConLD':1, 'ConLI':1, 'ConLw':1, 'Oth':1, 'WD':1,
                                       'CWD':2, 'Con':3, 'New':3})
    
    full["oSaleCondition"] = full.SaleCondition.map({'AdjLand':1, 'Abnorml':2, 'Alloca':2, 'Family':2, 'Normal':3, 'Partial':4})            
                
                        
                        
    
    return "Done!"
map_values()
#丢弃两个不需要特征
full.drop('LotAreaCut',axis=1,inplace=True)
full.drop('SalePrice',axis=1,inplace=True)
#PipeLine 好处：可以很方便的测试不同的特征组合
#标签编码三个‘Year’特征
class labelenc(BaseEstimator,TransformerMixin):
    def __inti__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        lab=LabelEncoder()
        X['YearBuilt']=lab.fit_transform(X['YearBuilt'])
        X['YearRemodAdd']=lab.fit_transform(X['YearRemodAdd'])
        X['GarageYrBlt']=lab.fit_transform(X['GarageYrBlt'])
        return X
# 对有偏度的特征使用log变换，然后get_dummies
class skew_dummies(BaseEstimator,TransformerMixin):
    def __init__(self,skew=0.5):
        self.skew=skew
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        X_numeric=X.select_dtypes(exclude=['object'])
        skewness=X_numeric.apply(lambda x:skew(x))
        skewness_feature=skewness[abs(skewness)>=self.skew].index
        X[skewness_feature]=np.log1p(X[skewness_feature])
        X=pd.get_dummies(X)
        return X
# 建立pipeline,就是将几个函数集合到一起了
pipe=Pipeline([
    ('labenc',labelenc()),
    ('skew_dummies',skew_dummies(skew=1)),
])
#保存最初变量以后使用
full2=full.copy()
data_pipe=pipe.fit_transform(full2)
data_pipe.shape
data_pipe.head()
#使用robustscaler以防还有其他离群点 （强大的缩放器）
scaler=RobustScaler()
n_train=train.shape[0]
X=data_pipe[:n_train]
test_X=data_pipe[n_train:]
y=train.SalePrice

X_scaled=scaler.fit(X).transform(X)
y_log=np.log(train.SalePrice)
test_X_scaled=scaler.transform(test_X)
#特征选择 Lasspo Ridge RandomForest GradientBoosting Tree
lasso=Lasso(alpha=0.001)
lasso.fit(X_scaled,y_log)
FI_lasso=pd.DataFrame({'Feature Importance':lasso.coef_},index=data_pipe.columns)
FI_lasso.sort_values('Feature Importance',ascending=False)
FI_lasso[FI_lasso['Feature Importance']!=0].sort_values('Feature Importance').plot(kind='barh',figsize=(15,25))
plt.xticks(rotation=90)
plt.show()
#根据特征重要性信息，在pipeline中做一些新添一些特征
class add_feature(BaseEstimator, TransformerMixin):
    def __init__(self,additional=1):
        self.additional = additional
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        if self.additional==1:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]   
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
            
        else:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]   
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
            
            X["+_TotalHouse_OverallQual"] = X["TotalHouse"] * X["OverallQual"]
            X["+_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]
            X["+_oMSZoning_TotalHouse"] = X["oMSZoning"] * X["TotalHouse"]
            X["+_oMSZoning_OverallQual"] = X["oMSZoning"] + X["OverallQual"]
            X["+_oMSZoning_YearBuilt"] = X["oMSZoning"] + X["YearBuilt"]
            X["+_oNeighborhood_TotalHouse"] = X["oNeighborhood"] * X["TotalHouse"]
            X["+_oNeighborhood_OverallQual"] = X["oNeighborhood"] + X["OverallQual"]
            X["+_oNeighborhood_YearBuilt"] = X["oNeighborhood"] + X["YearBuilt"]
            X["+_BsmtFinSF1_OverallQual"] = X["BsmtFinSF1"] * X["OverallQual"]
            
            X["-_oFunctional_TotalHouse"] = X["oFunctional"] * X["TotalHouse"]
            X["-_oFunctional_OverallQual"] = X["oFunctional"] + X["OverallQual"]
            X["-_LotArea_OverallQual"] = X["LotArea"] * X["OverallQual"]
            X["-_TotalHouse_LotArea"] = X["TotalHouse"] + X["LotArea"]
            X["-_oCondition1_TotalHouse"] = X["oCondition1"] * X["TotalHouse"]
            X["-_oCondition1_OverallQual"] = X["oCondition1"] + X["OverallQual"]
            
           
            X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
            X["Rooms"] = X["FullBath"]+X["TotRmsAbvGrd"]
            X["PorchArea"] = X["OpenPorchSF"]+X["EnclosedPorch"]+X["3SsnPorch"]+X["ScreenPorch"]
            X["TotalPlace"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"] + X["OpenPorchSF"]+X["EnclosedPorch"]+X["3SsnPorch"]+X["ScreenPorch"]

    
            return X
pipe=Pipeline([
    ('labenc',labelenc()),
    ('add_feature',add_feature(additional=2)),
    ('skew_dummies',skew_dummies(skew=1)),
])
#PCA降维
full_pipe=pipe.fit_transform(full)
full_pipe.shape
n_train=train.shape[0]
X=full_pipe[:n_train]
test_X=full_pipe[n_train:]
y=train.SalePrice

X_scaled=scaler.fit(X).transform(X)
y_log=np.log(train.SalePrice)
test_X_scaled=scaler.transform(test_X)
pca=PCA(n_components=350)
X_scaled=pca.fit_transform(X_scaled)
test_X_scaled=pca.transform(test_X_scaled)
X_scaled.shape, test_X_scaled.shape
#模型和验证
def rmse_cv(model,X,y):
    rmse=np.sqrt(-cross_val_score(model,X,y,scoring='neg_mean_squared_error',cv=5))
    return rmse
models=[LinearRegression(),Ridge(),Lasso(alpha=0.1,max_iter=10000),RandomForestRegressor(),GradientBoostingRegressor(),SVR(),LinearSVR(),ElasticNet(alpha=0.001,max_iter=10000),
       SGDRegressor(max_iter=1000,tol=1e-3),BayesianRidge(),KernelRidge(alpha=0.6,kernel='polynomial',degree=2,coef0=2.5),ExtraTreesRegressor(),XGBRegressor()]
names=['LR','Ridge','Lasso','RF','GBR','SVR','LinSVR','Ela','SGD','Bay','Ker','Extra','Xgb']
for name,model in zip(names,models):
    score=rmse_cv(model,X_scaled,y_log)
    print("{}:{:.6f},{:.4f}".format(name,score.mean(),score.std()))
#超参优化
class grid():
    def __init__(self,model):
        self.model=model
    def grid_get(self,X,y,param_grid):
        grid_search=GridSearchCV(self.model,param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
        grid_search.fit(X,y)
        print(grid_search.best_params_,np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score']=np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
#grid(Lasso()).grid_get(X_scaled,y_log,{'alpha':[0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009]})
#grid(Ridge()).grid_get(X_scaled,y_log,{'alpha':[60]})
#grid(SVR()).grid_get(X_scaled,y_log,{'C':[11,12,13,14],'epsilon':[0.009,0.01,0.02],'gamma':[0.0002,0.0003,0.0004,0.0005],'kernel':['rbf']})
#param_grid={'alpha':[0.1,0.2,0.5],'coef0':[0.8,1.0,14,15,20,25],'degree':[2,3,4],'kernel':['polynomial']}
#grid(KernelRidge()).grid_get(X_scaled,y_log,param_grid)
#grid(ElasticNet()).grid_get(X_scaled,y_log,{'alpha':[0.04,0.05,0.07,0.09],'l1_ratio':[0.04,0.03,0.02,0.01,0]})
#集成方法 根据他们的权重来取均值
class AverageWeight(BaseEstimator,RegressorMixin):
    def __init__(self,mod,weight):
        self.mod=mod
        self.weight=weight
    def fit(self,X,y):
        self.models_=[clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X,y)
        return self
    def predict(self,X):
        w=list()
        pred=np.array([model.predict(X) for model in self.models_])
        for data in range(pred.shape[1]):
            single=[pred[model,data]*weight for model ,weight in zip(range(pred.shape[0]),self.weight)]
            w.append(np.sum(single))
        return w
lasso=Lasso(alpha=0.0005)#0.11222985252930259
ridge=Ridge(alpha=60)# 0.11029403244945726
svr=SVR(C=12,epsilon=0.01,gamma=0.0004,kernel='rbf')#0.10877461055690453{'C': 12, 'epsilon': 0.01, 'gamma': 0.0004, 'kernel': 'rbf'} 0.10816083998052767
ker=KernelRidge(alpha=0.5,coef0=1.0,degree=3,kernel='polynomial')#0.10833058500999447
ela=ElasticNet(alpha=0.05,l1_ratio=0)#0.11029574008657018
bay=BayesianRidge()#Bay:0.110668,0.0060
#根据 gridsearch score 设置权重
w1=0.02
w2=0.2
w3=0.25
w4=0.3
w5=0.03
w6=0.2
#weight_avg=AverageWeight(mod=[lasso,ridge,svr,ker,ela,bay],weight=[w1,w2,w3,w4,w5,w6])
#score=rmse_cv(weight_avg,X_scaled,y_log)
#print(score.mean())
#如果只平均俩个最好的model 结果更好一点
#weight_avg=AverageWeight(mod=[svr,ker],weight=[0.5,0.5])
#score=rmse_cv(weight_avg,X_scaled,y_log)
#print(score.mean())
#堆叠 除了正常的堆叠，还要增加 get_oof 方法，因为之后我将组合堆叠生成的特征和原先的特征
class stacking(BaseEstimator,RegressorMixin,TransformerMixin):
    def __init__(self,mod,meta_model):
        self.mod=mod
        self.meta_model=meta_model
        self.kf=KFold(n_splits=5,random_state=42,shuffle=True)
    def fit(self,X,y):
        self.saved_model=[list() for i in self.mod]
        oof_train=np.zeros((X.shape[0],len(self.mod)))
        for i,model in enumerate(self.mod):
            for train_index,val_index in self.kf.split(X,y):
                renew_model=clone(model)
                renew_model.fit(X[train_index],y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index,i]=renew_model.predict(X[val_index])
        self.meta_model.fit(oof_train,y)
        return self
    def predict(self,X):
        whole_test=np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1) for single_model in self.saved_model])
        return self.meta_model.predict(whole_test)
   
    def get_oof(self,X,y,test_X):
        oof=np.zeros((X.shape[0],len(self.mod)))
        test_single=np.zeros((test_X.shape[0],5))
        test_mean=np.zeros((test_X.shape[0],len(self.mod)))
        for i,model in enumerate(self.mod):
            for j,(train_index,val_index) in enumerate(self.kf.split(X,y)):
                clone_model=clone(model)
                clone_model.fit(X[train_index],y[train_index])
                oof[val_index,i]=clone_model.predict(X[val_index])
                test_single[:,j]=clone_model.predict(test_X)
            test_mean[:,i]=test_single.mean(axis=1)
        return oof,test_mean
a=Imputer().fit_transform(X_scaled)
b=Imputer().fit_transform(y_log.values.reshape(-1,1)).ravel()
a.shape,b.shape,test_X_scaled.shape
stack_model=stacking(mod=[lasso,ridge,svr,ker,ela,bay],meta_model=ker)
#score=rmse_cv(stack_model,a,b)
#print(score.mean())
#现在我们提取堆叠生成的特征，然后和先前特征组合
X_train_stack,X_test_stack=stack_model.get_oof(a,b,test_X_scaled)
X_train_stack.shape,X_test_stack.shape
#X_train_stack[:1],a[:1]
X_train_add=np.hstack((a,X_train_stack))
X_test_add=np.hstack((test_X_scaled,X_test_stack))
X_train_add.shape,X_test_add.shape
#组合特征后的超参优化
#grid(Lasso()).grid_get(X_train_add,b,{'alpha': [0.00001,0.0001,0.0002],'max_iter':[10000]})
#grid(Ridge()).grid_get(X_train_add,b,{'alpha':[0.05,0.07,0.08,0.09,0.1]})
#grid(SVR()).grid_get(X_train_add,b,{'C':[8,9,11],'kernel':["rbf"],"gamma":[0.001,0.002,0.004],"epsilon":[0.009,0.01,0.02]})
#param_grid={'alpha':[0.007,0.009], 'kernel':["polynomial"], 'degree':[2,3],'coef0':[1.5,2.0,3.0,4.0]}
#grid(KernelRidge()).grid_get(X_train_add,b,param_grid)
#grid(ElasticNet()).grid_get(X_train_add,b,{'alpha':[0.00001,0.0001,0.0002,0.0003],'l1_ratio':[0.5,0.6,0.7],'max_iter':[10000]})
lasso = Lasso(alpha=0.0001,max_iter=10000)#{'alpha': 0.0005, 'max_iter': 10000} 0.11129660796478734 {'alpha': 0.0001, 'max_iter': 10000} 0.09226663454775956
ridge = Ridge(alpha=60)#{'alpha': 60} 0.11020174961498394  {'alpha': 0.09} 0.10714499635266204
svr = SVR(gamma= 0.001,kernel='rbf',C=11,epsilon=0.01)#{'C': 13, 'epsilon': 0.009, 'gamma': 0.0004, 'kernel': 'rbf'} 0.10823195916311412 {'C': 11, 'epsilon': 0.01, 'gamma': 0.001, 'kernel': 'rbf'} 0.10590847066839579
ker = KernelRidge(alpha=0.007 ,kernel='polynomial',degree=2 , coef0=4.0)#{'alpha': 0.2, 'coef0': 0.8, 'degree': 3, 'kernel': 'polynomial'} 0.10826973614765958 {'alpha': 0.007, 'coef0': 4.0, 'degree': 2, 'kernel': 'polynomial'} 0.10423535790468955
ela = ElasticNet(alpha=0.0001,l1_ratio=0.6,max_iter=10000)#{'alpha': 0.005, 'l1_ratio': 0.08, 'max_iter': 10000} 0.11117135158453571  {'alpha': 0.0001, 'l1_ratio': 0.6, 'max_iter': 10000} 0.09202893573835882
#bay = BayesianRidge()#Bay: 0.110577, 0.0060
X_train_add.shape, X_test_add.shape
score=rmse_cv(stack_model,X_train_add,b)
print(score.mean())
#提交
stack_model=stacking(mod=[lasso,ridge,svr,ker,ela],meta_model=ela)
stack_model.fit(X_train_add,b)
a.shape,b.shape,X_test_add.shape
submission['SalePrice']=np.exp(stack_model.predict(X_test_add))
submission.head()
'''param_grid={'alpha':[0.0001,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.003,0.004,0.005,0.01]}
grid_search=GridSearchCV(Lasso(),param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
grid_search.fit(X_scaled,y_log)
submission['SalePrice']=np.exp(grid_search.predict(test_scaled))'''
'''print(grid_search.best_params_,np.sqrt(-grid_search.best_score_))
grid_search.cv_results_['mean_test_score']=np.sqrt(-grid_search.cv_results_['mean_test_score'])
print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])'''
submission.to_csv('submission.csv',index=False)