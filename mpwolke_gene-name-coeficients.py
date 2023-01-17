# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import seaborn



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Configure necessary imports

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score

from sklearn.linear_model import LogisticRegression

import matplotlib.gridspec as gridspec

from scipy.stats import skew

from sklearn.preprocessing import RobustScaler,MinMaxScaler

import matplotlib.gridspec as gridspec

from scipy import stats

import matplotlib.style as style

style.use('seaborn-colorblind')
df = pd.read_csv('/kaggle/input/ai4all-project/results/classifier/lasso_1se/lasso_nonzero_coefs.csv')

df.head()
n_r=0.6                # Remove Null value ratio more than n_r. For example 0.6 means if column null ratio more than %60 then remove column

s_r=0.50               # If skewness more than %75 transform column to get normal distribution

c_r=1                  # Remove correlated columns

n_f= df.shape[1]  # n_f number of features. dataset.shape[1] means all columns. If you change it to 10, it will select 10 most correlated feature

r_s=42                  # random seed
print(f"data shape: {df.shape}")
sns.heatmap(df.isnull(),cmap = 'magma',cbar = False)
# categorical features

categorical_feat = [feature for feature in df.columns if df[feature].dtypes=='O']

print('Total categorical features: ', len(categorical_feat))

print('\n',categorical_feat)
# categorical features with missing values

categorical_nan = [feature for feature in df.columns if df[feature].isna().sum()>0 and df[feature].dtypes=='O']

print(categorical_nan)
cat=df.select_dtypes("object")

for column in cat:

    df[column].fillna(df[column].mode()[0], inplace=True)

    #dataset[column].fillna("NA", inplace=True)





fl=df.select_dtypes(["float64","int64"]).drop("coef",axis=1)

for column in fl:

    df[column].fillna(df[column].median(), inplace=True)

    #dataset[column].fillna(0, inplace=True)
sns.heatmap(df.isnull(),cmap = 'magma',cbar = False)
from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()

df["variable"] = encoder.fit_transform(df["variable"].fillna('Nan'))

df["gene_name"] = encoder.fit_transform(df["gene_name"].fillna('Nan'))

df.head()
degree=round(df['coef'].mean(),2)

fig = go.Figure(go.Indicator(

    mode = "gauge+number",

    gauge = {

       'axis': {'range': [None, 100]}},

    value = degree,

    title = {'text': "Average coef %"},

    domain = {'x': [0, 1], 'y': [0, 1]}

))

fig.show()
sns.catplot('gene_name','coef',data = df)
t = df[['variable','gene_name']].groupby('gene_name').agg([np.sum])



t
t.plot()
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

 



print('Skewness: '+ str(df['coef'].skew())) 

print("Kurtosis: " + str(df['coef'].kurt()))

plotting_3_chart(df, 'coef')
#log transform the target:

df["coef"] = np.log1p(df["coef"])
print('Skewness: '+ str(df['coef'].skew()))   

print("Kurtosis: " + str(df['coef'].kurt()))

plotting_3_chart(df, 'coef')
train_o=df[df["coef"].notnull()]

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



outs = detect_outliers(train_o['gene_name'], train_o['coef'],top=5)

outs

plt.show()
outs
from collections import Counter

outliers=outs

all_outliers=[]

numeric_features = train_o.dtypes[train_o.dtypes != 'object'].index

for feature in numeric_features:

    try:

        outs = detect_outliers(train_o[feature], train_o['coef'],top=5, plot=False)

    except:

        continue

    all_outliers.extend(outs)



print(Counter(all_outliers).most_common())

for i in outliers:

    if i in all_outliers:

        print(i)

train_o = train_o.drop(train_o.index[outliers])

test_o=df[df["coef"].isna()]

df =  pd.concat(objs=[train_o, test_o], axis=0,sort=False).reset_index(drop=True)
from scipy.special import boxcox1p

from scipy.stats import boxcox

lam = 0.15



#log transform skewed numeric features:

numeric_feats = df.dtypes[df.dtypes != "object"].index



skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > s_r]

skewed_feats = skewed_feats.index



df[skewed_feats] = boxcox1p(df[skewed_feats],lam)
df.columns[df.isnull().any()]
train_heat=df[df["coef"].notnull()]

train_heat=train_heat.drop(["gene_name"],axis=1)

style.use('ggplot')

sns.set_style('whitegrid')

plt.subplots(figsize = (10,6))

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

target_corr=df.corr()["coef"].abs()

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
all_features = df.keys()

# Removing features.

df = df.drop(df.loc[:,(df==0).sum()>=(df.shape[0]*0.9994)],axis=1)

df = df.drop(df.loc[:,(df==1).sum()>=(df.shape[0]*0.9994)],axis=1) 

# Getting and printing the remaining features.

remain_features = df.keys()

remov_features = [st for st in all_features if st not in remain_features]

print(len(remov_features), 'features were removed:', remov_features)
train=df[df["coef"].notnull()]

test=df[df["coef"].isna()]
k = n_f # if you change it 10 model uses most 10 correlated features

corrmat=abs(df.corr())

cols = corrmat.nlargest(k, 'coef')['coef'].index

train_x=df[cols].drop("coef",axis=1)

train_y=df["coef"]

X_test=test[cols].drop("coef",axis=1)
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

test_variable=test['variable']

my_submission = pd.DataFrame({'variable': test_variable, 'coef': np.expm1(y_test)})

my_submission.to_csv('submission_bm.csv', index=False)

print("Model Name: "+str(best_model))

print(best_model.score(X_val,y_val))

y_pred=best_model.predict(X_val)

print("RMSE: "+str(np.sqrt(mean_squared_error(y_val,y_pred))))
plt.figure(figsize=(10,7))

y_pred=best_model.predict(X_val)

sns.regplot(x=y_val,y=y_pred,truncate=False)

plt.show()