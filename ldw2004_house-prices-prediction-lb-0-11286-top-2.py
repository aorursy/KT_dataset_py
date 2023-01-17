#Final Score 0.11286 top 2%
#Refer to massquantity's code. The link is https://www.kaggle.com/massquantity/all-you-need-is-pca-lb-0-11421-top-4
#I only change some features 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
plt.style.use('ggplot')
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Exploratory Visualization
plt.figure(figsize=(15,8))
sns.boxplot(train.YearBuilt, train.SalePrice)
plt.figure(figsize=(12,6))
plt.scatter(x=train.GrLivArea, y=train.SalePrice)
plt.xlabel("GrLivArea", fontsize=13)
plt.ylabel("SalePrice", fontsize=13)
plt.ylim(0,800000)
train.drop(train[(train["GrLivArea"]>4000)&(train["SalePrice"]<300000)].index,inplace=True)
full=pd.concat([train,test], ignore_index=True)
full.drop(['Id'],axis=1, inplace=True)
full.shape

#Data Cleaning
aa = full.isnull().sum()
aa[aa>0].sort_values(ascending=False)
full.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean','median','count'])
full["LotAreaCut"] = pd.qcut(full.LotArea,10)
full.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean','median','count'])
full['LotFrontage']=full.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
full['LotFrontage']=full.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
full.shape
cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
for col in cols:
    full[col].fillna(0, inplace=True)
full.shape
cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt",
         "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
for col in cols1:
    full[col].fillna("None", inplace=True)
full.shape
cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType",
         "Exterior1st", "Exterior2nd"]
for col in cols2:
    full[col].fillna(full[col].mode()[0], inplace=True)
full.shape
full.isnull().sum()[full.isnull().sum()>0]

#Feature Engineering
NumStr = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold",
          "YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]
for col in NumStr:
    full[col]=full[col].astype(str)
full.shape
full.groupby(['MSSubClass'])[['SalePrice']].agg(['mean','median','count'])
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
full.drop("LotAreaCut",axis=1,inplace=True)
full.drop(['SalePrice'],axis=1,inplace=True)
full.shape
fcol = full.columns.tolist()

#Pipeline
class labelenc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        lab=LabelEncoder()
        X["YearBuilt"] = lab.fit_transform(X["YearBuilt"])
        X["YearRemodAdd"] = lab.fit_transform(X["YearRemodAdd"])
        X["GarageYrBlt"] = lab.fit_transform(X["GarageYrBlt"])
        return X
class skew_dummies(BaseEstimator, TransformerMixin):
    def __init__(self,skew=0.5):
        self.skew = skew
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X_numeric=X.select_dtypes(exclude=["object"])
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= self.skew].index
        X[skewness_features] = np.log1p(X[skewness_features])
        X = pd.get_dummies(X)
        return X
pipe = Pipeline([
    ('labenc', labelenc()),
    ('skew_dummies', skew_dummies(skew=1)),
    ])
full2 = full.copy()
data_pipe = pipe.fit_transform(full2)
data_pipe.shape
data_pipe.head()
scaler = RobustScaler()
n_train=train.shape[0]

X = data_pipe[:n_train]
test_X = data_pipe[n_train:]
y= train.SalePrice

X_scaled = scaler.fit(X).transform(X)
y_log = np.log(train.SalePrice)
test_X_scaled = scaler.transform(test_X)

#Feature Selection
lasso=Lasso(alpha=0.001)
lasso.fit(X_scaled,y_log)
FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=data_pipe.columns)
FI_lasso.sort_values("Feature Importance",ascending=False)
FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
plt.xticks(rotation=90)
plt.show()

#KMeans Features Crosses
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=15, random_state=42).fit(X_scaled.T)

#SSE = []
#Scores = []
#for i in range(5,30,1):
#    clf = KMeans(n_clusters=i, random_state=42).fit(X_scaled.T)
#    SSE.append(clf.inertia_)
#    Scores.append(silhouette_score(X_scaled.T,clf.labels_,metric='euclidean'))

#X = range(5,30,1)
#plt.xlabel('k')
#plt.ylabel('SSE')
#plt.plot(X,SSE,'o-')
#plt.show()
#X = range(5,30,1)
#plt.xlabel('k')
#plt.ylabel('轮廓系数')
#plt.plot(X,Scores,'o-')
#plt.show()
kmres = pd.DataFrame(columns=data_pipe.columns)
kmres.loc[0] = kmeans.labels_
#kmres
#

cls = []
for i in range(0,15):
    l = []
    for j in range(len(data_pipe.columns)):
        if kmres.iat[0,j]==i:
            l.append(data_pipe.columns[j])
    cls.append(l)

#cls
#for i in range(0,15):
#    print(cls[i])
#    print(" ")

class add_feature(BaseEstimator, TransformerMixin):
    def __init__(self,additional=1):
        self.additional = additional
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        if self.additional==1:
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

            
        else:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]   
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
            
            X["*_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]
            #X["+_GrLivArea_LotArea"] = X["GrLivArea"] + X["LotArea"]
            #X["*_GrLivArea_LotArea"] = X["GrLivArea"] * X["LotArea"]
            
            X["+_oMSZoning_TotalHouse"] = X["oMSZoning"] * X["TotalHouse"]
            X["+_oMSZoning_YearBuilt"] = X["oMSZoning"] + X["YearBuilt"]
            #X["+_oMSZoning_oNeighborhood"] = X["oMSZoning"] + X["oNeighborhood"]
            #X["*_oMSZoning_oNeighborhood"] = X["oMSZoning"] * X["oNeighborhood"]
            
            X["+_oNeighborhood_TotalHouse"] = X["oNeighborhood"] * X["TotalHouse"]
            X["+_oNeighborhood_YearBuilt"] = X["oNeighborhood"] + X["YearBuilt"]
            #X["+_oNeighborhood_YearRemodAdd"] = X["oNeighborhood"] + X["YearRemodAdd]
            #X["*_oNeighborhood_YearBuilt"] = X["oNeighborhood"] * X["YearBuilt"]
            #X["*_oNeighborhood_YearRemodAdd"] = X["oNeighborhood"] * X["YearRemodAdd"]
            
            X["+_BsmtFinSF1_OverallQual"] = X["BsmtFinSF1"] * X["OverallQual"]
            
            X["*_oFunctional_TotalHouse"] = X["oFunctional"] * X["TotalHouse"]
            X["+_oFunctional_OverallQual"] = X["oFunctional"] + X["OverallQual"]
            
            X["*_LotArea_OverallQual"] = X["LotArea"] * X["OverallQual"]
            
            X["*_oCondition1_TotalHouse"] = X["oCondition1"] * X["TotalHouse"]
            X["+_oCondition1_OverallQual"] = X["oCondition1"] + X["OverallQual"]
        
            X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
            X["Rooms"] = X["FullBath"]+X["TotRmsAbvGrd"]

            
            #通过KMeans，组合特征
                    
            #X["*_TotalArea_OverallCond"] = X["TotalArea"] * X["OverallCond"]  

            X["+_GrLivArea_TotalBsmtSF"] = X["GrLivArea"] + X["TotalBsmtSF"] 
            X["*_GrLivArea_OverallCond"] = X["GrLivArea"] * X["OverallCond"]

            X["*_1stFlrSF_OverallQual"] = X["1stFlrSF"] * X["OverallQual"]
            X["*_2ndFlrSF_OverallQual"] = X["2ndFlrSF"] * X["OverallQual"] 
            X["*_1stFlrSF_OverallCond"] = X["1stFlrSF"] * X["OverallCond"]
            X["*_2ndFlrSF_OverallCond"] = X["2ndFlrSF"] * X["OverallCond"] 

            X["*_YearBuilt_OverallQual"] = X["YearBuilt"] * X["OverallQual"]
            X["*_YearRemodAdd_OverallQual"] = X["YearRemodAdd"] * X["OverallQual"] 
            X["*_YearBuilt_OverallCond"] = X["YearBuilt"] * X["OverallCond"]
            X["*_YearRemodAdd_OverallCond"] = X["YearRemodAdd"] * X["OverallCond"] 

            #X["*_TotalArea_oNeighborhood"] = X["TotalArea"] * X["oNeighborhood"]  
            X["*_oNeighborhood_OverallQual"] = X["oNeighborhood"] * X["OverallQual"] 
            X["*_oNeighborhood_OverallCond"] = X["oNeighborhood"] * X["OverallCond"] 

            X["*_oSaleCondition_TotalArea"] = X["oSaleCondition"] * X["TotalArea"]
            X["*_oSaleCondition_OverallQual"] = X["oSaleCondition"] * X["OverallQual"]
            X["*_oSaleCondition_OverallCond"] = X["oSaleCondition"] * X["OverallCond"]
            #X["+_oSaleCondition_OverallQual"] = X["oSaleCondition"] + X["OverallQual"]
            X["+_oSaleCondition_OverallCond"] = X["oSaleCondition"] + X["OverallCond"]
            #X["+_oSaleCondition_oMSZoning"] = X["oSaleCondition"] + X["oMSZoning"]

            #X["*_WoodDeckSF_OverallQual"] = X["WoodDeckSF"] * X["OverallQual"]
            #X["*_WoodDeckSF_OverallCond"] = X["WoodDeckSF"] * X["OverallCond"]
            #X["+_WoodDeckSF_OverallQual"] = X["WoodDeckSF"] + X["OverallQual"]
            #X["+_WoodDeckSF_OverallCond"] = X["WoodDeckSF"] + X["OverallCond"]

            #X["+_GarageArea_LotArea"] = X["GarageArea"] + X["LotArea"]
            #X["*_GarageArea_OverallQual"] = X["GarageArea"] * X["OverallQual"]
            #X["*_GarageArea_OverallCond"] = X["GarageArea"] * X["OverallCond"]
            #X["+_GarageArea_OverallQual"] = X["GarageArea"] + X["OverallQual"]
            #X["+_GarageArea_OverallCond"] = X["GarageArea"] + X["OverallCond"]
            
            #X["+_TotalBsmtSF_YearBuilt"] = X["TotalBsmtSF"] + X["YearBuilt"]
            #X["+_TotalBsmtSF_YearRemodAdd"] = X["TotalBsmtSF"] + X["YearRemodAdd"]
            #X["*_TotalBsmtSF_YearBuilt"] = X["TotalBsmtSF"] * X["YearBuilt"]
            #X["*_TotalBsmtSF_YearRemodAdd"] = X["TotalBsmtSF"] * X["YearRemodAdd"]
            
            #X["+_oHouseStyle_BsmtUnfSF"] = X["oHouseStyle"] + X["BsmtUnfSF"]
            #X["*_oHouseStyle_BsmtUnfSF"] = X["oHouseStyle"] * X["BsmtUnfSF"]
            
            return X
pipe = Pipeline([
    ('labenc', labelenc()),
    ('add_feature', add_feature(additional=2)),
    ('skew_dummies', skew_dummies(skew=1)),
    ])

#PCA
full_pipe = pipe.fit_transform(full)
full_pipe.shape
n_train=train.shape[0]
X = full_pipe[:n_train]
test_X = full_pipe[n_train:]
y= train.SalePrice

X_scaled = scaler.fit(X).transform(X)
y_log = np.log(train.SalePrice)
test_X_scaled = scaler.transform(test_X)
pca = PCA(n_components=430)
X_scaled=pca.fit_transform(X_scaled)
test_X_scaled = pca.transform(test_X_scaled)
X_scaled.shape, test_X_scaled.shape

#Modeling & Evaluation
def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse
models = [LinearRegression(),Ridge(),Lasso(alpha=0.01,max_iter=10000),RandomForestRegressor(),GradientBoostingRegressor(),
          SVR(),LinearSVR(),ElasticNet(alpha=0.001,max_iter=10000),SGDRegressor(max_iter=1000,tol=1e-3),
          BayesianRidge(),KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
          ExtraTreesRegressor(),XGBRegressor()]
names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela","SGD","Bay","Ker","Extra","Xgb"]
for name, model in zip(names, models):
    score = rmse_cv(model, X_scaled, y_log)
    print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))

class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
#grid(Lasso()).grid_get(X_scaled,y_log,{'alpha': [0.001,0.002,0.003,0.0004,0.005,0.006,0.007,0.008],'max_iter':[10000]})
#param_grid = {'alpha':[50,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70]}
#grid(Ridge()).grid_get(X_scaled,y_log,param_grid)
#param_grid = {'C':[5,6,7,8,9,10,11,12,13],'kernel':["rbf"],"gamma":[0.0008],"epsilon":[0.007,0.008,0.009]}
#grid(SVR()).grid_get(X_scaled,y_log,param_grid)
#param_grid={'alpha':[0.1,0.2,0.3,0.4,0.5,0.6,0.7], 'kernel':["polynomial"], 'degree':[3],'coef0':[0.5,0.6,0.7,0.8,0.9]}
#grid(KernelRidge()).grid_get(X_scaled,y_log,param_grid)
#grid(ElasticNet()).grid_get(X_scaled,y_log,{'alpha':[0.041,0.042,0.043,0.044,0.045,0.046,0.047,0.048,0.049,0.051,0.053,0.055,0.057],'l1_ratio':[0.0001],'max_iter':[10000]})
#grid(BayesianRidge()).grid_get(X_scaled,y_log,{'n_iter':[300]})
#grid(XGBRegressor(learning_rate=0.02,n_estimators=1000,max_depth=3,min_child_weight=3,subsample=0.8,colsample_bytree=0.8)).grid_get(X_scaled,y_log,{
#    'max_depth':[3,4,5,6,7],'min_child_weight':[1,2,3,4,5]})

#Ensemble Methods
class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self,mod,weight):
        self.mod = mod
        self.weight = weight
        
    def fit(self,X,y):
        self.models_ = [clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X,y)
        return self
    
    def predict(self,X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        # for every data point, single model prediction times weight, then add them together
        for data in range(pred.shape[1]):
            single = [pred[model,data]*weight for model,weight in zip(range(pred.shape[0]),self.weight)]
            w.append(np.sum(single))
        return w
lasso = Lasso(alpha=0.0005,max_iter=10000)
ridge = Ridge(alpha=60)
svr = SVR(gamma= 0.0004,kernel='rbf',C=13,epsilon=0.009)
ker = KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 , coef0=0.8)
ela = ElasticNet(alpha=0.005,l1_ratio=0.08,max_iter=10000)
bay = BayesianRidge()
# assign weights based on their gridsearch score
w1 = 0.02
w2 = 0.2
w3 = 0.25
w4 = 0.3
w5 = 0.03
w6 = 0.2
weight_avg = AverageWeight(mod = [lasso,ridge,svr,ker,ela,bay],weight=[w1,w2,w3,w4,w5,w6])
score = rmse_cv(weight_avg,X_scaled,y_log)
print(score.mean())
weight_avg = AverageWeight(mod = [svr,ker],weight=[0.5,0.5])
score = rmse_cv(weight_avg,X_scaled,y_log)
print(score.mean())

#Stacking
class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self,mod,meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)
        
    def fit(self,X,y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))
        
        for i,model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X,y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index,i] = renew_model.predict(X[val_index])
        
        self.meta_model.fit(oof_train,y)
        return self
    
    def predict(self,X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1) 
                                      for single_model in self.saved_model]) 
        return self.meta_model.predict(whole_test)
    
    def get_oof(self,X,y,test_X):
        oof = np.zeros((X.shape[0],len(self.mod)))
        test_single = np.zeros((test_X.shape[0],5))
        test_mean = np.zeros((test_X.shape[0],len(self.mod)))
        for i,model in enumerate(self.mod):
            for j, (train_index,val_index) in enumerate(self.kf.split(X,y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index],y[train_index])
                oof[val_index,i] = clone_model.predict(X[val_index])
                test_single[:,j] = clone_model.predict(test_X)
            test_mean[:,i] = test_single.mean(axis=1)
        return oof, test_mean
a = Imputer().fit_transform(X_scaled)
b = Imputer().fit_transform(y_log.values.reshape(-1,1)).ravel()
stack_model = stacking(mod=[lasso,svr,ker,ela],meta_model=ker)
score = rmse_cv(stack_model,a,b)
print(score.mean())
stack_model.fit(a,b)
pred = np.exp(stack_model.predict(test_X_scaled))
result=pd.DataFrame({'Id':test.Id, 'SalePrice':pred})
result.to_csv("submission.csv",index=False)