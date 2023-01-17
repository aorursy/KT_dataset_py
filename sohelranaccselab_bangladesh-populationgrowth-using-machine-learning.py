# ****-------Notebook Summary----***



#Data Science, Machine Learning



#Data Visualization,EDA Analysis, Data Pre-processing,Data Cleaning,Data Split

#-------------------------------------------------------------------------------------------------

#Machine Learning Algorithm:



#Multiples Linear Regression models:XgboostRegressor,Catboost Regressor,Random forest Regressor,....





#Visualize output at graph
#Enivornment Setup
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
from matplotlib import pyplot as plt
#Data Read, Data Visualization,EDA Analysis,Data Pre-Processing,Data Splitting
#Data Read

file_path = '../input/bangladesh-population-growth-ratio'

df=pd.read_csv(f'{file_path}/data-resource_2016_10_24_bangladesh-population-growth-ratio.csv')
df.head()
df = df.loc[:,~df.columns.duplicated()]
import pandas_profiling
# preparing profile report



profile_report = pandas_profiling.ProfileReport(df,minimal=True)

profile_report
df.info()
df.describe()
df.shape
df.Population.value_counts()
df.apply(lambda x: sum(x.isnull()),axis=0)
df.groupby("Male").mean()
import seaborn as sns
sns.scatterplot(x = df.index,y = df['Population'])
sns.regplot(x = df.index,y = df['Population'])
sns.scatterplot(x = df.index,y = df['Population'],hue = df['Male'])
sns.scatterplot(x = df.index,y = df['Population'],hue = df['Female'])
sns.set()

fig = plt.figure(figsize = [15,20])

cols = ['Male', 'Female', 'Population']

cnt = 1

for col in cols :

    plt.subplot(2,3,cnt)

    sns.distplot(df[col],hist_kws=dict(edgecolor="k", linewidth=1,color='green'),color='red')

    cnt+=1

plt.show() 
drop_cols = ['Year'] 

df = df.drop(drop_cols, axis=1)
X = df.drop('Population', axis=1)

y = df['Population']
from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.model_selection import  train_test_split, cross_val_score



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train.shape
X_train
# Distplot

fig, ax2 = plt.subplots(2, 2, figsize=(16, 16))

sns.distplot(df['Male'],ax=ax2[0][0])

sns.distplot(df['Female'],ax=ax2[0][1])

y_train
# Feature Scaling

sc = RobustScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.metrics import mean_squared_error

from sklearn import metrics

from scipy.stats import pearsonr

warnings.filterwarnings("ignore")



target = "Population"

def model(algorithm,dtrainx,dtrainy,dtestx,dtesty,of_type,plot=False):

    

    print (algorithm)

    print ("***************************************************************************")

    algorithm.fit(dtrainx,dtrainy)

    

    #print(algorithm.get_params(deep=True))

    

    prediction = algorithm.predict(dtestx)

    

    print ("ROOT MEAN SQUARED ERROR :", np.sqrt(mean_squared_error(dtesty,prediction)) )

    print ("***************************************************************************")

    

    print ('Performance on training data :', algorithm.score(dtrainx,dtrainy)*100)

    print ('Performance on testing data :', algorithm.score(dtestx,dtesty)*100)



    print ("***************************************************************************")

    if plot==True:

        sns.jointplot(x=dtesty, y=prediction, stat_func=pearsonr,kind="reg", color="b") 

    

       

    prediction = pd.DataFrame(prediction)

    cross_val = cross_val_score(algorithm,dtrainx,dtrainy,cv=5)#,scoring="neg_mean_squared_error"

    cross_val = cross_val.ravel()

    print ("CROSS VALIDATION SCORE")

    print ("************************")

    print ("cv-mean :",cross_val.mean()*100)

    print ("cv-std  :",cross_val.std()*100)

    

    if plot==True:

        plt.figure(figsize=(20,22))

        plt.subplot(211)



        testy = dtesty.reset_index()["Population"]



        ax = testy.plot(label="originals",figsize=(20,9),linewidth=2)

        ax = prediction[0].plot(label = "predictions",figsize=(20,9),linewidth=2)

        plt.legend(loc="best")

        plt.title("ORIGINALS VS PREDICTIONS")

        plt.xlabel("index")

        plt.ylabel("values")

        ax.set_facecolor("k")

import xgboost as xgb

from xgboost.sklearn import XGBRegressor

xgr =XGBRegressor(random_state=42)

model(xgr,X_train,y_train,X_test,y_test,"feat",True)
xgr_1=XGBRegressor(random_state=42,learning_rate = 0.03,

                max_depth = 9, n_estimators = 1000,n_jobs=-1,reg_alpha=0.005,gamma=0.1,subsample=0.7,colsample_bytree=0.9, colsample_bylevel=0.9, colsample_bynode=0.9)

model(xgr_1,X_train,y_train,X_test,y_test,"feat",True)
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

param_grid={'n_estimators' : [1000,2000,3000,2500],

            'max_depth' : [1,2, 3,5,7,9,10,11,15],

            'learning_rate' :[ 0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.8, 1.0],

                                                     }

# Create a base model

xgbr = XGBRegressor(random_state = 42,reg_alpha=0.005,gamma=0.1,subsample=0.7,colsample_bytree=0.9, colsample_bylevel=0.9, colsample_bynode=0.9)



# Instantiate the grid search model

grid_search = GridSearchCV(estimator = xgbr, param_grid = param_grid, 

                          cv = 5, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

best_grid = grid_search.best_estimator_

model(best_grid,X_train,y_train,X_test,y_test,"feat",True)
from sklearn.ensemble import  RandomForestRegressor

rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=80,

           max_features='auto', max_leaf_nodes=None,

           min_impurity_decrease=0.0, min_impurity_split=None,

           min_samples_leaf=1, min_samples_split=2,

           min_weight_fraction_leaf=0.0, n_estimators=2000, n_jobs=-1,

           oob_score=False, random_state=None, verbose=0, warm_start=False)

model(rf,X_train,y_train,X_test,y_test,"feat")
from catboost import CatBoostRegressor
cb_model = CatBoostRegressor(iterations=2000,

                             learning_rate=0.03,

                             depth=9,

                             eval_metric='RMSE',

                             random_seed = 42,

                             bagging_temperature = 0.2,

                             od_type='Iter',

                             metric_period = 50,

                             od_wait=20)
model(cb_model,X_train,y_train,X_test,y_test,"feat",True)
#Multiple Machine Learning Algorithm for Resgression 
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.svm import SVR, LinearSVR

from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge

from sklearn.kernel_ridge import KernelRidge

from xgboost import XGBRegressor
model_dict = {

    'LinearRegession': LinearRegression(),

    'Ridge':Ridge(),

    'Lasso':Lasso(),

    'KernelRidge':KernelRidge(),

    'SGDRegressor':SGDRegressor(),

    'BayesianRidge':BayesianRidge(),

    'ElasticNet': ElasticNet(),

    'LinearSVR':LinearSVR(),

    'XGBRegressor':XGBRegressor(random_state=42, n_estimators=2000, max_depth=9),

    'RandomForestRegressor': RandomForestRegressor(random_state=0, n_estimators=2000, max_depth=9),

    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42, n_estimators=2000, max_depth=9, learning_rate=0.01)

}
data_list = list()

for name, model in model_dict.items():

    data_dict = dict()

    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)

    test_score = model.score(X_test, y_test)

    data_dict['model'] = name

    data_dict['train_score'] = train_score

    data_dict['test_score'] = test_score

    data_list.append(data_dict)

score_df = pd.DataFrame(data_list)

score_df['score_diff'] = score_df['train_score'] - score_df['test_score']

model_df = score_df.sort_values(['test_score'], ascending=[False])

model_df[model_df['test_score'] > 0.5]