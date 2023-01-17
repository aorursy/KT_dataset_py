# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import skew

from sklearn.preprocessing import RobustScaler,MinMaxScaler

import matplotlib.gridspec as gridspec

from scipy import stats

import matplotlib.style as style

style.use('seaborn-colorblind')

train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sub=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')



dataset =  pd.concat(objs=[train,test], axis=0,sort=False).reset_index(drop=True)
dataset.head()
#exploring the data

dataset.columns
print(dataset.shape)
dataset.describe()

#So we get the range,mean,std of all the numericale columns.
dataset.tail()
#Analyse Sales price

dataset['SalePrice'].describe()
#fig=plt.figure(figsize=(18,12))

#grid = GridSpec(ncols=2, nrows=1, figure=fig)

f, axes = plt.subplots(1, 2, figsize=(25, 15), sharex=True)

sns.despine(left=True)

#fig,axs= plt.subplot(2,1,1)

sns.distplot(dataset['SalePrice'],ax=axes[0])

sns.boxplot(dataset['SalePrice'],orient='h',ax=axes[1])

plt.tight_layout()
print(dataset['SalePrice'].skew())

print(dataset['SalePrice'].kurt())
dataset["SalePrice"] = np.log1p(dataset["SalePrice"])

dataset["SalePrice"] 
print(dataset['SalePrice'].skew())

print(dataset['SalePrice'].kurt())
f, axes = plt.subplots(1, 2, figsize=(25, 15), sharex=True)

sns.despine(left=True)

sns.distplot(dataset['SalePrice'],ax=axes[0])

sns.boxplot(dataset['SalePrice'],orient='h',ax=axes[1])

plt.tight_layout()
#Realtionship b/w some columns and saleprice

cols= ["LotArea","Street","LandSlope","GrLivArea", "TotalBsmtSF",'LotFrontage']

n_rows= 2

n_col= 3

fig,axs= plt.subplots(2,3,figsize=(20,16))



for r in range(0,n_rows):

    for c in range(0,n_col):

        i= r*n_col + c

        ax=axs[r][c]

        sns.scatterplot(dataset[cols[i]], dataset['SalePrice'],ax=ax)
#CHaecking Null values

dataset.isnull().sum()
plt.figure(figsize=(16,10))

sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.tight_layout()
cat=dataset.select_dtypes("object")

for column in cat:

    dataset[column].fillna(dataset[column].mode()[0], inplace=True)

    #dataset[column].fillna("NA", inplace=True)





fl=dataset.select_dtypes(["float64","int64"]).drop("SalePrice",axis=1)

for column in fl:

    dataset[column].fillna(dataset[column].median(), inplace=True)

    #dataset[column].fillna(0, inplace=True)
#checking through heatmap

plt.figure(figsize=(16,10))

sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.tight_layout()
#Auto Detect Outliers



train_o=dataset[dataset["SalePrice"].notnull()]

outliers = [30, 88, 462, 523, 632, 1298, 1324]

from sklearn.neighbors import LocalOutlierFactor

def detect_outliers(x, y, top=5, plot=True):

    lof = LocalOutlierFactor(n_neighbors=40, contamination=0.1)

    x_ =np.array(x).reshape(-1,1)

    preds = lof.fit_predict(x_)

    lof_scr = lof.negative_outlier_factor_

    out_idx = pd.Series(lof_scr).sort_values()[:top].index

    if plot:

        f, ax = plt.subplots(figsize=(9, 6))

        plt.scatter(x=x, y=y, c=np.exp(lof_scr), cmap='RdBu')

    return out_idx



outs = detect_outliers(train_o['GrLivArea'], train_o['SalePrice'],top=5) #got 1298,523

outs

plt.show()
#Detect and Remove outliers



from collections import Counter

all_outliers=[]

numeric_features = train_o.dtypes[train_o.dtypes != 'object'].index

for feature in numeric_features:

    try:

        outs = detect_outliers(train_o[feature], train_o['SalePrice'],top=5, plot=False)

    except:

        continue

    all_outliers.extend(outs)



print(Counter(all_outliers).most_common())

for i in outliers:

    if i in all_outliers:

        print(i)

train_o = train_o.drop(train_o.index[outliers])

test_o=dataset[dataset["SalePrice"].isna()]

dataset =  pd.concat(objs=[train_o, test_o], axis=0,sort=False).reset_index(drop=True)
dataset.isnull().sum()
train_heat=dataset[dataset["SalePrice"].notnull()]

train_heat=train_heat.drop(["Id"],axis=1)

#style.use('ggplot')

#sns.set_style('whitegrid')

plt.subplots(figsize = (30,20))

## Plotting heatmap. 



# Generate a mask for the upper triangle (taken from seaborn example gallery)

mask = np.zeros_like(train_heat.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(train_heat.corr(), 

            cmap=sns.diverging_palette(2, 132, l=60, n=6),#colours 

            mask=mask,

            annot=True, 

            center = 0, 

           );

## Give title. 

plt.title("Heatmap of all the Features", fontsize = 30);
train_heat.corr().abs()
feature_corr = train_heat.corr().abs()

target_corr=dataset.corr()["SalePrice"].abs()

target_corr=pd.DataFrame(target_corr)

target_corr=target_corr.reset_index()

feature_corr_unstack= feature_corr.unstack()

df_fc=pd.DataFrame(feature_corr_unstack,columns=["corr"])

df_fc=df_fc[(df_fc["corr"]>=.80)&(df_fc["corr"]<1)].sort_values(by="corr",ascending=False)

df_dc=df_fc.reset_index()



target_corr=df_dc.merge(target_corr, left_on='level_1', right_on='index',

          suffixes=('_left', '_right'))
target_corr
#dataset=dataset.drop(["GarageArea","TotRmsAbvGrd","TotalBsmtSF","PoolArea","hasgarage","has2ndfloor","YearBuilt","Fireplaces"],axis=1)

dataset=dataset.drop(["GarageArea","TotRmsAbvGrd"],axis=1)
dataset=pd.get_dummies(dataset,columns=cat.columns)
train=dataset[dataset["SalePrice"].notnull()]

test=dataset[dataset["SalePrice"].isna()]
n_f= dataset.shape[1]

k = n_f # if you change it 10 model uses most 10 correlated features

corrmat=abs(dataset.corr())

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

train_x=train[cols].drop("SalePrice",axis=1)

train_y=train["SalePrice"]

X_test=test[cols].drop("SalePrice",axis=1)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.20, random_state=r_s)
# from sklearn.linear_model import LinearRegression

# from sklearn.metrics import mean_squared_error,mean_absolute_error

# from sklearn.metrics import accuracy_score









# reg = LinearRegression().fit(x_train, y_train)

# y_pred=reg.predict(x_test)







# acc_logreg = accuracy_score(y_pred, y_test)

# #print(y_pred)

# #reg.score(x_train, y_train)

# #print(mean_absolute_error(y_test,y_pred))

# #mean_squared_error(y_test,y_pred))

# #np.sqrt(mean_squared_error(y_test,y_pred)))
r_s=42  

from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor,ExtraTreesRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from xgboost import XGBRegressor

from sklearn.linear_model import Ridge,RidgeCV,BayesianRidge,LinearRegression,Lasso,LassoCV,ElasticNet,RANSACRegressor,HuberRegressor,PassiveAggressiveRegressor,ElasticNetCV

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import VotingRegressor

from sklearn.svm import SVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.cross_decomposition import CCA

from sklearn.neural_network import MLPRegressor







my_regressors=[ 

               ElasticNet(alpha=0.01),

               ElasticNetCV(),

               CatBoostRegressor(logging_level='Silent',random_state=r_s),

               GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber',random_state =r_s),

               LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       random_state=r_s

                                       ),

               RandomForestRegressor(random_state=r_s),

               AdaBoostRegressor(random_state=r_s),

               ExtraTreesRegressor(random_state=r_s),

               SVR(C= 20, epsilon= 0.008, gamma=0.0003),

               Ridge(alpha=13),

               RidgeCV(),

               BayesianRidge(),

               DecisionTreeRegressor(),

               LinearRegression(),

               KNeighborsRegressor(),

               Lasso(alpha=0.00047,random_state=r_s),

               LassoCV(),

               KernelRidge(),

               CCA(),

               MLPRegressor(random_state=r_s),

               HuberRegressor(),

               RANSACRegressor(random_state=r_s),

               PassiveAggressiveRegressor(random_state=r_s)

               #XGBRegressor(random_state=r_s)

              ]



regressors=[]



for my_regressor in my_regressors:

    regressors.append(my_regressor)





scores_val=[]

scores_train=[]

MAE=[]

MSE=[]

RMSE=[]





for regressor in regressors:

    scores_val.append(regressor.fit(X_train,y_train).score(X_val,y_val))

    scores_train.append(regressor.fit(X_train,y_train).score(X_train,y_train))

    y_pred=regressor.predict(X_val)

    MAE.append(mean_absolute_error(y_val,y_pred))

    MSE.append(mean_squared_error(y_val,y_pred))

    RMSE.append(np.sqrt(mean_squared_error(y_val,y_pred)))



    

results=zip(scores_val,scores_train,MAE,MSE,RMSE)

results=list(results)

results_score_val=[item[0] for item in results]

results_score_train=[item[1] for item in results]

results_MAE=[item[2] for item in results]

results_MSE=[item[3] for item in results]

results_RMSE=[item[4] for item in results]





df_results=pd.DataFrame({"Algorithms":my_regressors,"Training Score":results_score_train,"Validation Score":results_score_val,"MAE":results_MAE,"MSE":results_MSE,"RMSE":results_RMSE})

df_results
best_models=df_results.sort_values(by="RMSE")

best_model=best_models.iloc[0][0]

best_stack=best_models["Algorithms"].values

best_models
best_model.fit(X_train,y_train)

y_test=best_model.predict(X_test)

test_Id=test['Id']

my_submission = pd.DataFrame({'Id': test_Id, 'SalePrice': np.expm1(y_test)})

my_submission.to_csv('submission_bm.csv', index=False)

# print("Model Name: "+str(best_model))

# print(best_model.score(X_val,y_val))

# y_pred=best_model.predict(X_val)

# print("RMSE: "+str(np.sqrt(mean_squared_error(y_val,y_pred))))
my_submission.to_csv('submission1.csv', index=False)
sub=pd.read_csv('./submission_bm.csv')
sub