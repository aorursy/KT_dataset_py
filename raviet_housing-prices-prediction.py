# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train
test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.info()
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
Nl=round(train.isnull().mean()*100,2)
Nl[Nl>0]
Nl=round(test.isnull().mean()*100,2)
Nl[Nl>0]
datasets=[train,test]
col=["Alley","PoolQC","Fence","MiscFeature","Id"]
train=train.drop(columns=col,axis=1)
test=test.drop(columns=col,axis=1)
train[(train["MasVnrType"]!=train["MasVnrType"]) & (train["MasVnrArea"]==train["MasVnrArea"])]
train[(train["BsmtQual"]!=train["BsmtQual"]) & (train["BsmtCond"]==train["BsmtCond"])]
train[(train["BsmtQual"]!=train["BsmtQual"]) & (train["BsmtFinType1"]==train["BsmtFinType1"])]
train["MasVnrType"].mode()
train["BsmtQual"]=train["BsmtQual"].fillna(train["BsmtQual"].mode())
train["BsmtQual"].unique()
train["BsmtQual"].mode()
train["LotFrontage"]=train["LotFrontage"].fillna(round(train["LotFrontage"].mean(),0))
train["LotArea"]=train["LotArea"].fillna(round(train["LotArea"].mean(),0))
train["MasVnrType"]=train["MasVnrType"].fillna(train["MasVnrType"].mode()[0])
train["MasVnrArea"]=train["MasVnrArea"].fillna(train["MasVnrArea"].mean())
train["BsmtQual"]=train["BsmtQual"].fillna(train["BsmtQual"].mode()[0])
train["BsmtCond"]=train["BsmtCond"].fillna(train["BsmtCond"].mode()[0])
train["BsmtExposure"]=train["BsmtExposure"].fillna(train["BsmtExposure"].mode()[0])
train["BsmtFinType1"]=train["BsmtFinType1"].fillna(train["BsmtFinType1"].mode()[0])
train["BsmtFinType2"]=train["BsmtFinType2"].fillna("NA")
train["Electrical"]=train["Electrical"].fillna(train["Electrical"].mode()[0])
train["FireplaceQu"]=train["FireplaceQu"].fillna("Unknown")
train["GarageType"]=train["GarageType"].fillna(train["GarageType"].mode()[0])
train["GarageYrBlt"]=train["GarageYrBlt"].fillna(round(train["GarageYrBlt"].mean(),0))
train["GarageFinish"]=train["GarageFinish"].fillna(train["GarageFinish"].mode()[0])
train["GarageQual"]=train["GarageQual"].fillna(train["GarageQual"].mode()[0])
train["GarageCond"]=train["GarageCond"].fillna(train["GarageCond"].mode()[0])


train.info()
t = pd.DataFrame(data={"col": train.dtypes.index, "type": train.dtypes}).reset_index(drop=True)
col_names = t["col"][t.type != "object"]
col_names
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

fig = plt.figure(figsize=(12,12))

ax = fig.add_subplot(1,1,1)
cax = ax.matshow(train.corr(), interpolation = 'nearest')

fig.colorbar(cax)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

ax.set_xticklabels(col_names)
ax.set_yticklabels(col_names);
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if ((corr_matrix.iloc[i, j] >= threshold) or (corr_matrix.iloc[i, j] <= -threshold)) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns and colname!='SalePrice':
                    del dataset[colname] # deleting the column from the dataset
    print(dataset)
    plt.figure(figsize=(12,12))
    sns.heatmap(dataset.corr(), annot=False)
    plt.show()
        
correlation(train,0.7)
train.info()
train.describe()
sns.distplot(train['SalePrice']);


plt.xlim(-100, 500)
sns.boxplot(x=train.LotFrontage)
train=train[train.LotFrontage<200]
sns.boxplot(x=train.LotArea)
train=train[train.LotArea<100000]
sns.boxplot(x=train.YearBuilt)

sns.boxplot(x=train.MasVnrArea)
def MasVnrarea(x):
    if(x<100):
        return "low"
    elif(x>=100 and x <=500):
        return "mid"
    else:
        return "high"
train.MasVnrArea=train.MasVnrArea.apply(lambda x: MasVnrarea(x))
test.MasVnrArea=test.MasVnrArea.apply(lambda x: MasVnrarea(x))
train.MasVnrArea
sns.countplot(x="MasVnrArea",data=train)
sns.countplot(x="MasVnrArea",data=test)

sns.boxplot(x=train.BsmtFinSF1)
train=train[train.BsmtFinSF1<2500]
train["SalePrice"] = np.log1p(train["SalePrice"])
train.info()
sns.distplot(train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
sns.distplot(train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)
sns.distplot(np.log1p(train['GrLivArea']), fit=norm);
fig = plt.figure()
res = stats.probplot(np.log1p(train['GrLivArea']), plot=plt)
train['GrLivArea']=np.log1p(train['GrLivArea'])
sns.distplot(train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'], plot=plt)
train['TotalBsmtSF']=np.log1p(train['TotalBsmtSF'])
sns.distplot((train[train['TotalBsmtSF']>0].TotalBsmtSF), fit=norm);
fig = plt.figure()
res = stats.probplot((train[train['TotalBsmtSF']>0].TotalBsmtSF), plot=plt)
df_train=pd.get_dummies(train)
df_train
y_train = train.SalePrice.values
df_train
X=df_train.drop("SalePrice",axis=1)
X
y=train.SalePrice
y
for col in test.columns:
    if( col not in train.columns):
        test=test.drop(col,axis=1)
        
for col in test.columns:
    if(test[col].dtype=='O'):
        test[col]=test[col].fillna(test[col].mode()[0])
    else:
        test[col]=test[col].fillna(round(test[col].mean(),0))
test1=pd.get_dummies(test)
for col in X.columns:
    if(col not in test1.columns) :
        test1[col]=0
        print(col)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split,cross_val_predict
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from yellowbrick.model_selection import RFECV
n_folds = 5
def rmsle_cv(model):
    kf = StratifiedKFold(n_folds, shuffle=True, random_state=42).get_n_splits(df_train.values)
    rmse= np.sqrt(-cross_val_score(model, X.values, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
score=rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
xgb1 = XGBRegressor()
parameters = { #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03,.05], #so called `eta` value
              'max_depth': [5,10,50],
              'min_child_weight': [4,10],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500,1000,2000],
            'random_state':[5]
                }

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 5,
                        n_jobs = -1,                        
                        verbose=True)
xgb_grid.fit(X,y)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
test2=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
xgb2=XGBRegressor(n_estimators=1000,learning_rate=.03,colsample_bytree=.7,max_depth=5,min_child_weight=4,objective="reg:linear",subsample=.7)
xgb2.fit(X,y)
test3=test1.reindex(columns=X.columns.tolist())
final=xgb2.predict(test3)
final=np.exp(final)
output = pd.DataFrame({'Id': test2.Id, 'SalePrice': final})
output.to_csv('HousingPrices_XGB.csv', index=False)
rf = RandomForestRegressor()
parameters = {'bootstrap': [True],
                 'max_depth': [10, 50, 70],
                 'max_features': ['sqrt'],
                 'min_samples_leaf': [10, 20, 4],
                 'min_samples_split': [5, 10],
                 'n_estimators': [500,1000,2000],
                 'random_state':[5]
                }

rf_grid = GridSearchCV(rf,
                        parameters,
                        cv = 5,
                        n_jobs = -1,
                        scoring="neg_mean_squared_error",
                        verbose=True)
rf_grid.fit(X,y)

print(rf_grid.best_score_)
print(rf_grid.best_params_)




model_param={
    'RFR':{
        'model':RandomForestRegressor(),
        'param':{'bootstrap': [True, False],
                 'max_depth': [10, 50, 60, 70],
                 'max_features': ['auto', 'sqrt'],
                 'min_samples_leaf': [1, 2, 4],
                 'min_samples_split': [2, 5, 10],
                 'n_estimators': [500,1000,2000]
                }
    },
    'Logistic Regression':{
        'model':LogisticRegression(),
        'param':{
            'C':[5,10,100]
        }
    }
    
}
scores=[]

for model_name,mp in model_param.items():
    clf=GridSearchCV(mp['model'],mp['param'],cv=5, return_train_score=False,scoring="neg_mean_squared_error")
    clf.fit(X,y)
    scores.append({
        'model':model_name,
        'best_score':clf.best_score_,
        'best_params': clf.best_params_
    })

pd.dataframe(scores)
scores=[]

for model_name,mp in model_param.items():
    clf=RandomizedSearchCV(mp['model'],mp['param'],cv=5, return_train_score=False)
    clf.fit(X,y)
    scores.append({
        'model':model_name,
        'best_score':clf.best_score_,
        'best_params': clf.best_params_
    })

pd.dataframe(scores)
scores
output = pd.DataFrame({'Id': test2.Id, 'SalePrice': final})
output.to_csv('HousingPrices_submission.csv', index=False)
output
GBoost.fit(X,y)
pred=GBoost.predict(test1)
for col in X.columns:
    if col not in test1.columns:
        print(col)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

col_names=(X.columns).tolist()
col_names
test3=test2.reindex(columns=col_names)
X.columns
test1.columns
final1=xg_reg.predict(test3)
final1=np.exp(final1)
output = pd.DataFrame({'Id': test2.Id, 'SalePrice': final1})
output.to_csv('HousingPrices_submission2.csv', index=False)
output
xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, preds))
rmse
xg_reg.predict(test1)
