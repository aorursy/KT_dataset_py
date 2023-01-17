# linear algebra
import numpy as np 

# data processing
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

# data visualization
import matplotlib.pyplot as plt 
import plotly.express as px 
import seaborn as sns 
from scipy import stats  

# Prediction models
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

#Evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error 
from hyperopt import hp, tpe, fmin
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.head(10)
train.info()
#correlation map
f,ax=plt.subplots(figsize=(25, 25))
sns.heatmap(train.corr(), annot=True, linewidths=.4, fmt= '.1f',ax=ax)
plt.show()
train.isnull().sum()[train.isnull().sum() > 100] # Features possess dramatically numerous null values that we cannot fill them properly. 
train.drop(["LotFrontage","Alley","FireplaceQu","PoolQC","Fence","MiscFeature"],axis=1,inplace = True) # Dropping them
fig, ax =plt.subplots(1,2,figsize=(14, 6))
sns.scatterplot(x="TotalBsmtSF", y="SalePrice", data=train,ax=ax[0])
sns.scatterplot(x="GrLivArea", y="SalePrice", data=train,ax=ax[1])
fig.show()
train = train[train.TotalBsmtSF < 5000].reset_index(drop = True)
train= train[train["GrLivArea"] < 4500].reset_index(drop = True)
fig_neigh = train.groupby("Neighborhood").median().sort_values(by="SalePrice")
fig = px.box(fig_neigh, x=fig_neigh.index, y="SalePrice",)
fig.show()
def neig_func(model):
    dic = {'NridgHt': 25, 'NoRidge': 24, 'StoneBr': 23,
           'Timber': 22, 'Somerst': 21, 'Veenker': 20, 'Crawfor': 19,
           'ClearCr': 18, 'CollgCr': 17, 'Blmngtn': 16, 'NWAmes': 15,
           'Gilbert': 14, 'SawyerW': 13, 'Mitchel': 12, 'NPkVill': 11,
           'NAmes': 10, 'SWISU': 9, 'Blueste': 8, 'Sawyer': 7, 'BrkSide': 6,
           'Edwards': 5, 'OldTown': 4, 'BrDale': 3, 'IDOTRR': 2, 'MeadowV': 1}
    model["Neighborhood"].replace(dic,inplace = True)
def land_slope(model):
    # Dictionary values are created through the opposite of slope values. In other words, High value for sharp slope, Low value for flattened surface.
    
    #Gtl -> Gentle slope
    #Mod -> Moderate Slope
    #Sev -> Severe Slope
    
    dic = {"Gtl":3,"Mod":2,"Sev":1}
    model["LandSlope"].replace(dic,inplace = True)
train["Garage"] = train["GarageCars"] * train["GarageArea"]
corr_matrix = train.corr()
print(corr_matrix["SalePrice"].sort_values(ascending = False)[3:6])
# By creating new feature we get better correlation.
# Function implemented to complete null values by most repeated value for each one of them.
def complete_null(model):
    for i in model.isnull().sum()[model.isnull().sum() != 0].index:
        model[i].fillna(model[i].mode()[0],inplace = True)
from scipy.stats import norm
# histogram and normal probability plot. Thanks to Pedro Marcelino for this insightful trick. I learned that the notebook created by him.
sns.distplot(train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
train["SalePrice"] = np.log(train["SalePrice"])
def changes(model):
    model.drop(["GarageCars","Utilities","BsmtFinSF2","BsmtUnfSF"],axis=1,inplace = True) # No need them
    
    # Used the functions implemented above
    land_slope(model) 
    neig_func(model)
    
    # To get better correlation, I segmented the data with slightly better values.
    model["YearBuilt"] = pd.cut(model["YearBuilt"],bins=[1870,1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,1995,2000,2003,2005,2007,2011],labels =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).astype(int)
    
    # Categorical features to numerical features
    model["Foundation"].replace({"PConc":6,"Stone":5,"CBlock":4,"Wood":3,"BrkTil":2,"Slab":1},inplace = True)
    model.replace({"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1},inplace = True)
    model["CentralAir"].replace({"Y":1,"N":0},inplace = True)
    
    # Some ineffable feature engineering 
    model["FirstandSecondSum"] = model["2ndFlrSF"] + model["1stFlrSF"]
    model["Qual"] = model["KitchenQual"] + model["ExterQual"]
    model["NeigYear"] = model["Neighborhood"] + model["YearBuilt"]
    model["AllQual"] = model["OverallQual"] * model["Qual"]
    model["hadi"] = model["FirstandSecondSum"] + model["TotalBsmtSF"] # "hadi" means "come on" in Turkish. I created this feature hopefully when I was stucked in generating new feautes.
    
    # Log transformation same like we did for "SalePrice"
    model["GrLivArea"] = np.log(model["GrLivArea"])
    model["LotArea"] = np.log(model["LotArea"])
    model['hadi'] = np.log(model['hadi'])
    
    # Another anomaly detection
    model = model[model["LotArea"] < 100000].reset_index(drop=True)
    
    # Some brand new features after I reach top 40%.
    model = model.assign(HasFirePlace = (model["Fireplaces"] != 0).astype(int))
    model = model.assign(HasGarageArea = (model["GarageArea"] != 0).astype(int))
    model = model.assign(HasOpenPorchSF = (model["OpenPorchSF"] != 0).astype(int))
    model = model.assign(HasPoolArea = (model["PoolArea"] != 0).astype(int))
    model = model.assign(HasMasVnrArea = (model["MasVnrArea"] != 0).astype(int))  
    
    model.drop(["KitchenQual","OverallQual","Qual","Fireplaces","OpenPorchSF","FirstandSecondSum","TotalBsmtSF","ExterQual","GarageArea","2ndFlrSF","1stFlrSF","YearRemodAdd","PoolArea"],axis=1,inplace = True)

    return model
train = changes(train)
train["BsmtQual"].fillna(train.BsmtQual.median(),inplace = True)
train["MasVnrArea"].fillna(train.MasVnrArea.median(),inplace = True)
train["BsmtCond"].fillna(3,inplace=True)
complete_null(train)
train = pd.get_dummies(train)
print("Train Data has {} rows".format(train.shape[0]))
print("Train Data has {} columns".format(train.shape[1]))
print(f"Train Data memory usage is {train.memory_usage().sum() / 1024 ** 2:.3f} MB")
train.info()
corr_matrix = train.corr()
corr_matrix["SalePrice"].sort_values(ascending = False)[:20]
# Take a look on how it works with some powerful models
def predict_func(df):
    X = df.drop("SalePrice",axis=1)
    y = df.loc[:,"SalePrice"]
    
    ran_for_reg = RandomForestRegressor()
    scores_rand = cross_val_score(ran_for_reg, X, y, cv=15,scoring="neg_root_mean_squared_error")
    print("Random Forest Regressor: " + str(-scores_rand.mean()))
    
    grad_boost_reg = GradientBoostingRegressor()
    scores_grad = cross_val_score(grad_boost_reg, X, y, cv=15,scoring="neg_root_mean_squared_error")
    print("Gradient Boost Regression: " + str(-scores_grad.mean()))
    
    xg_boost_reg = xgb.XGBRegressor()
    scores_xg = cross_val_score(xg_boost_reg, X, y, cv=15,scoring="neg_root_mean_squared_error")
    print("XG Boost Regression: " + str(-scores_xg.mean()))
predict_func(train.drop("Id",axis=1)) # We do not need Id column deeply
X_train = train.drop(["SalePrice","Id"],axis=1)
y_train = train.loc[:,"SalePrice"]
def test_changes(model):
    model["GarageCars"].fillna(model.GarageCars.median(),inplace = True)
    model["GarageArea"].fillna(model.GarageArea.mean(),inplace = True)
    model["Garage"] = model["GarageCars"] * model["GarageArea"]
    model.drop(["GarageCars","LotFrontage","Alley","FireplaceQu","PoolQC","Fence","MiscFeature","Utilities","BsmtFinSF2","BsmtUnfSF"],axis=1,inplace = True)
    land_slope(model)
    neig_func(model)
    model["YearBuilt"] = pd.cut(model["YearBuilt"],bins=[1870,1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,1995,2000,2003,2005,2007,2011],labels =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).astype(int)
    model["Foundation"].replace({"PConc":6,"Stone":5,"CBlock":4,"Wood":3,"BrkTil":2,"Slab":1},inplace = True)
    model.replace({"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1},inplace = True)
    model["CentralAir"].replace({"Y":1,"N":0},inplace = True)
    model["FirstandSecondSum"] = model["2ndFlrSF"] + model["1stFlrSF"]
    model["KitchenQual"].fillna(model.KitchenQual.median(),inplace = True)
    model["ExterQual"].fillna(model.ExterQual.median(),inplace = True)
    model["Qual"] = model["KitchenQual"] + model["ExterQual"]
    model["NeigYear"] = model["Neighborhood"] + model["YearBuilt"]
    
    model["AllQual"] = model["OverallQual"] * model["Qual"]
    model["hadi"] = model["FirstandSecondSum"] + model["TotalBsmtSF"]
    model["GrLivArea"] = np.log(model["GrLivArea"])
    model["LotArea"] = np.log(model["LotArea"])
    model['hadi'] = np.log(model['hadi'])
    model = model.assign(HasFirePlace = (model["Fireplaces"] != 0).astype(int))
    model = model.assign(HasGarageArea = (model["GarageArea"] != 0).astype(int))
    model = model.assign(HasOpenPorchSF = (model["OpenPorchSF"] != 0).astype(int))
    model = model.assign(HasPoolArea = (model["PoolArea"] != 0).astype(int))
    model = model.assign(HasMasVnrArea = (model["MasVnrArea"] != 0).astype(int))
    model.drop(["KitchenQual","OverallQual","Qual","Fireplaces","OpenPorchSF","FirstandSecondSum","TotalBsmtSF","ExterQual","GarageArea","2ndFlrSF","1stFlrSF","YearRemodAdd","PoolArea"],axis=1,inplace = True)
    return model
test = test_changes(test)
test["BsmtQual"].fillna(test.BsmtQual.median(),inplace = True)
test["MasVnrArea"].fillna(test.MasVnrArea.median(),inplace = True)
test["hadi"].fillna(test.hadi.median(),inplace = True)
complete_null(test)
test = pd.get_dummies(test)
X_test = test.drop("Id",axis=1)
print("Test Data has {} rows".format(test.shape[0]))
print("Test Data has {} columns".format(test.shape[1]))
print(f"Test Data memory usage is {test.memory_usage().sum() / 1024 ** 2:.3f} MB")
for i in X_train.columns:   
    if not i in X_test:
        print(i)
not_in = ['Condition2_RRAe',
 'Condition2_RRAn',
 'Condition2_RRNn',
 'HouseStyle_2.5Fin',
 'RoofMatl_Membran',
 'RoofMatl_Metal',
 'RoofMatl_Roll',
 'Exterior1st_ImStucc',
 'Exterior1st_Stone',
 'Exterior2nd_Other',
 'Heating_Floor',
 'Heating_OthW',
 'Electrical_Mix']

X_train.drop(not_in,axis=1,inplace=True)
predict_func(train.drop("Id",axis=1))
# Parameters we are about to use for trying to get best result for xgboosting. I decide to use xgboosting because it works well better than any other regression models. I try all of the models and conclude with xgboosting.
space = {'n_estimators':hp.quniform('n_estimators', 1000, 4000, 100),
         'gamma':hp.uniform('gamma', 0.01, 0.05),
         'learning_rate':hp.uniform('learning_rate', 0.00001, 0.025),
         'max_depth':hp.quniform('max_depth', 3,7,1),
         'subsample':hp.uniform('subsample', 0.60, 0.95),
         'colsample_bytree':hp.uniform('colsample_bytree', 0.60, 0.98),
         'colsample_bylevel':hp.uniform('colsample_bylevel', 0.60, 0.98),
         'reg_lambda': hp.uniform('reg_lambda', 1, 20)
        }

def objective(params):
    params = {'n_estimators': int(params['n_estimators']),
             'gamma': params['gamma'],
             'learning_rate': params['learning_rate'],
             'max_depth': int(params['max_depth']),
             'subsample': params['subsample'],
             'colsample_bytree': params['colsample_bytree'],
             'colsample_bylevel': params['colsample_bylevel'],
             'reg_lambda': params['reg_lambda']}
    
    xb_a= xgb.XGBRegressor(**params)
    score = cross_val_score(xb_a, X_train, y_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1).mean()
    return -score
best = fmin(fn= objective, space= space, max_evals=20, rstate=np.random.RandomState(1), algo=tpe.suggest)
xb_b = xgb.XGBRegressor(random_state=0,
                        n_estimators=int(best['n_estimators']), 
                        colsample_bytree= best['colsample_bytree'],
                        gamma= best['gamma'],
                        learning_rate= best['learning_rate'],
                        max_depth= int(best['max_depth']),
                        subsample= best['subsample'],
                        colsample_bylevel= best['colsample_bylevel'],
                        reg_lambda= best['reg_lambda']
                       )

xb_b.fit(X_train, y_train)
mean_squared_error(y_train,xb_b.predict(X_train))
FirstTest = xb_b.predict(X_test)
FirstTest = FirstTest.reshape(-1,1)
FirstTest = np.exp(FirstTest) 
result = pd.DataFrame(data = test["Id"])
result["SalePrice"]= FirstTest
result.to_csv(r'xgboost_result.csv',index = False, header=True)