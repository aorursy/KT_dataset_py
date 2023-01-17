# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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



import warnings  

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_l=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_l=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

desc=pd.read_fwf("/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt")

dataset =  pd.concat(objs=[train_l, test_l], axis=0,sort=False).reset_index(drop=True)

sample=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

n_r=0.6                # Remove Null value ratio more than n_r. For example 0.6 means if column null ratio more than %60 then remove column

s_r=0.50               # If skewness more than %75 transform column to get normal distribution

c_r=1                  # Remove correlated columns

n_f= dataset.shape[1]  # n_f number of features. dataset.shape[1] means all columns. If you change it to 10, it will select 10 most correlated feature

r_s=42                  # random seed
def plotting_3_chart(df, feature): 

    ## Creating a customized chart. and giving in figsize and everything. 

    fig = plt.figure(constrained_layout=True, figsize=(12,8))

    ## crea,ting a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    #gs = fig3.add_gridspec(3, 3)



    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Histogram')

    ## plot the histogram. 

    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)



    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(df.loc[:,feature], plot = ax2)



    ## Customizing the Box Plot. 

    ax3 = fig.add_subplot(grid[:, 2])

    ## Set title. 

    ax3.set_title('Box Plot')

    ## Plotting the box plot. 

    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );

 



print('Skewness: '+ str(dataset['SalePrice'].skew())) 

print("Kurtosis: " + str(dataset['SalePrice'].kurt()))

plotting_3_chart(dataset, 'SalePrice')
#log transform the target:

dataset["SalePrice"] = np.log1p(dataset["SalePrice"])
print('Skewness: '+ str(dataset['SalePrice'].skew()))   

print("Kurtosis: " + str(dataset['SalePrice'].kurt()))

plotting_3_chart(dataset, 'SalePrice')


dataset_isna=dataset.isna()

dataset_isna_sum=dataset_isna.sum()

dataset_isna_ratio=dataset_isna_sum/len(dataset)

if "SalePrice" in dataset_isna_ratio:

    dataset_isna_ratio.drop("SalePrice",inplace=True)

remove_columns=dataset_isna_ratio[dataset_isna_ratio>n_r]

columns=pd.DataFrame(remove_columns)

print("This Columns will be remove because of null ratio higher than %"+str(n_r*100)+": ")

print(remove_columns)

dataset=dataset.drop(columns.index,axis=1)
cat=dataset.select_dtypes("object")

for column in cat:

    dataset[column].fillna(dataset[column].mode()[0], inplace=True)

    #dataset[column].fillna("NA", inplace=True)





fl=dataset.select_dtypes(["float64","int64"]).drop("SalePrice",axis=1)

for column in fl:

    dataset[column].fillna(dataset[column].median(), inplace=True)

    #dataset[column].fillna(0, inplace=True)



train_o=dataset[dataset["SalePrice"].notnull()]

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



outs = detect_outliers(train_o['GrLivArea'], train_o['SalePrice'],top=5)

outs

plt.show()
outs
from collections import Counter

outliers=outs

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
from scipy.special import boxcox1p

from scipy.stats import boxcox

lam = 0.15



#log transform skewed numeric features:

numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index



skewed_feats = dataset[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > s_r]

skewed_feats = skewed_feats.index



dataset[skewed_feats] = boxcox1p(dataset[skewed_feats],lam)

dataset.columns[dataset.isnull().any()]
train_heat=dataset[dataset["SalePrice"].notnull()]

train_heat=train_heat.drop(["Id"],axis=1)

style.use('ggplot')

sns.set_style('whitegrid')

plt.subplots(figsize = (30,20))

## Plotting heatmap. 



# Generate a mask for the upper triangle (taken from seaborn example gallery)

mask = np.zeros_like(train_heat.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True





sns.heatmap(train_heat.corr(), 

            cmap=sns.diverging_palette(255, 133, l=60, n=7), 

            mask = mask, 

            annot=True, 

            center = 0, 

           );

## Give title. 

plt.title("Heatmap of all the Features", fontsize = 30);
feature_corr = train_heat.corr().abs()

target_corr=dataset.corr()["SalePrice"].abs()

target_corr=pd.DataFrame(target_corr)

target_corr=target_corr.reset_index()

feature_corr_unstack= feature_corr.unstack()

df_fc=pd.DataFrame(feature_corr_unstack,columns=["corr"])

df_fc=df_fc[(df_fc["corr"]>=.80)&(df_fc["corr"]<1)].sort_values(by="corr",ascending=False)

df_dc=df_fc.reset_index()



#df_dc=pd.melt(df_dc, id_vars=['corr'], var_name='Name')

target_corr=df_dc.merge(target_corr, left_on='level_1', right_on='index',

          suffixes=('_left', '_right'))



cols=target_corr["level_0"].values



target_corr
dataset=dataset.drop(["GarageArea","TotRmsAbvGrd"],axis=1)
dataset=pd.get_dummies(dataset,columns=cat.columns)

all_features = dataset.keys()

# Removing features.

dataset = dataset.drop(dataset.loc[:,(dataset==0).sum()>=(dataset.shape[0]*0.9994)],axis=1)

dataset = dataset.drop(dataset.loc[:,(dataset==1).sum()>=(dataset.shape[0]*0.9994)],axis=1) 

# Getting and printing the remaining features.

remain_features = dataset.keys()

remov_features = [st for st in all_features if st not in remain_features]

print(len(remov_features), 'features were removed:', remov_features)
train=dataset[dataset["SalePrice"].notnull()]

test=dataset[dataset["SalePrice"].isna()]

k = n_f # if you change it 10 model uses most 10 correlated features

corrmat=abs(dataset.corr())

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

train_x=train[cols].drop("SalePrice",axis=1)

train_y=train["SalePrice"]

X_test=test[cols].drop("SalePrice",axis=1)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.20, random_state=r_s)


from sklearn.utils.testing import all_estimators

from sklearn import base



estimators = all_estimators()



for name, class_ in estimators:

    if issubclass(class_, base.RegressorMixin):

       print(name+"()")
np.random.seed(seed=r_s)



from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor

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

               ElasticNet(alpha=0.001,l1_ratio=0.70,max_iter=100,tol=0.01, random_state=r_s),

               ElasticNetCV(l1_ratio=0.9,max_iter=100,tol=0.01,random_state=r_s),

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

               Ridge(alpha=6),

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

               HistGradientBoostingRegressor(random_state=r_s),

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

print("Model Name: "+str(best_model))

print(best_model.score(X_val,y_val))

y_pred=best_model.predict(X_val)

print("RMSE: "+str(np.sqrt(mean_squared_error(y_val,y_pred))))
plt.figure(figsize=(10,7))

y_pred=best_model.predict(X_val)

sns.regplot(x=y_val,y=y_pred,truncate=False)

plt.show()
i_num=[]

j_num=[]

score=[]

RMSE=[]

for i in range(1,8):

    stack_models=i

    for j in range(1,4):

        base_model=j

        best_n_models=best_models.head(stack_models).index

        regressors_top_n=list( regressors[i] for i in best_n_models)



        from mlxtend.regressor import StackingCVRegressor

        stack = StackingCVRegressor(regressors=regressors_top_n,meta_regressor= best_stack[base_model], use_features_in_secondary=True)

        comb=stack.fit(X_train,y_train)

        y_pred=comb.predict(X_val)

        score.append(comb.score(X_val,y_val))

        RMSE.append(np.sqrt(mean_squared_error(y_val,y_pred)))

        i_num.append(i)

        j_num.append(j)

        

opt_resr=zip(score,RMSE,i_num,j_num)

opt_resr=set(opt_resr)

opt_resr_score=[item[0] for item in opt_resr]

opt_resr_RMSE=[item[1] for item in opt_resr]

opt_resr_i_num=[item[2] for item in opt_resr]

opt_resr_j_num=[item[3] for item in opt_resr]







df_opt_resr=pd.DataFrame({"Score":opt_resr_score,"RMSE":opt_resr_RMSE,"i_num":opt_resr_i_num,"j_num":opt_resr_j_num})

df_opt_resr=df_opt_resr.sort_values(by="RMSE")

opt_best_model_i_num=df_opt_resr.iloc[0][2].astype(int)

opt_best_model_j_num=df_opt_resr.iloc[0][3].astype(int)

best_n_models=best_models.head(opt_best_model_i_num).index

regressors_top_n=list( regressors[i] for i in best_n_models)

from mlxtend.regressor import StackingCVRegressor

stack = StackingCVRegressor(regressors=regressors_top_n,meta_regressor= regressors[opt_best_model_j_num], use_features_in_secondary=True)

comb=stack.fit(X_train,y_train)

print(comb.score(X_val,y_val))

y_pred=comb.predict(X_val)

print("RMSE: "+str(np.sqrt(mean_squared_error(y_val,y_pred))))
plt.figure(figsize=(10,7))

y_pred=comb.predict(X_val)

sns.regplot(x=y_val,y=y_pred,truncate=False)

plt.show()
y_test=comb.predict(X_test)

test_Id=test['Id']

my_submission = pd.DataFrame({'Id': test_Id, 'SalePrice': np.expm1(y_test)})

#my_submission = pd.DataFrame({'Id': test_Id, 'SalePrice': np.expm1(y_test)})

my_submission.to_csv('submission_stack.csv', index=False)
def blended_predictions(X):

    return ((0.40 * comb.predict(X)) + \

            (0.40 * best_models.iloc[0][0].predict(X)) + \

            (0.10 * best_models.iloc[1][0].predict(X)) + \

            (0.10 * best_models.iloc[2][0].predict(X)))



y_pred=blended_predictions(X_val)





#y_pred=comb.predict(X_val)

print("RMSE: "+str(np.sqrt(mean_squared_error(y_val,y_pred))))



y_test=blended_predictions(X_test)

test_Id=test['Id']

my_submission = pd.DataFrame({'Id': test_Id, 'SalePrice': np.expm1(y_test)})

my_submission.to_csv('submission_blend3.csv', index=False)



import shap  # package used to calculate Shap values



xgb=XGBRegressor(random_state=2)

xgb=xgb.fit(X_train,y_train)



# Create object that can calculate shap values

explainer = shap.TreeExplainer(xgb)



# calculate shap values. This is what we will plot.

# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.

shap_values = explainer.shap_values(X_val)



# Make plot. Index of [1] is explained in text below.

shap.summary_plot(shap_values, X_val)

shap.force_plot(explainer.expected_value, shap_values[0,:], X_val.iloc[0,:],matplotlib=True)
most_corr=dataset.corr()["SalePrice"].abs().sort_values(ascending=False).index[1]

shap.dependence_plot(most_corr, shap_values, X_val, interaction_index="SaleCondition_Partial")
