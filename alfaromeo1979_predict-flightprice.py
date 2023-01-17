# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

pd.set_option('max_colwidth', 800)

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import seaborn as sns

import matplotlib.pylab as plt

from matplotlib import colors

%matplotlib inline



import itertools



#Preprocessing Library

from sklearn import preprocessing

import re

from scipy.stats import zscore



#Regression Library

from sklearn.linear_model import LinearRegression,ElasticNet,Lasso,Ridge,BayesianRidge

from lightgbm.sklearn import LGBMRegressor

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb

from sklearn import neighbors

from sklearn import svm



from sklearn import metrics

from sklearn.metrics import make_scorer



from sklearn.model_selection import cross_val_predict,cross_val_score,cross_validate,train_test_split



from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

#!pip install holidays

#import holidays

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
train = reduce_mem_usage(pd.read_excel("../input/Data_Train.xlsx"))

test = reduce_mem_usage(pd.read_excel("../input/Test_set.xlsx"))
train['Additional_Info'].unique()

train.dtypes
train.isnull().sum()

test.isnull().sum()
#Best Fit from other look-up

train['Route'].fillna('DEL → BLR → COK',inplace=True)

train['Total_Stops'].fillna('1 stop',inplace=True)
for combination in itertools.zip_longest(sorted(train['Airline'].unique()), sorted(test['Airline'].unique())):

    print(combination)

print ("-------------")    

for combination in itertools.zip_longest(sorted(train['Source'].unique()), sorted(test['Source'].unique())):

    print(combination)

print ("-------------")    

for combination in itertools.zip_longest(sorted(train['Destination'].unique()), sorted(test['Destination'].unique())):

    print(combination)

print ("-------------")    

for combination in itertools.zip_longest(sorted(train['Total_Stops'].unique()), sorted(test['Total_Stops'].unique())):

    print(combination)
plt.figure(figsize=(8,8))

plt.hist(train.Price,bins=100,color='b')

plt.title('Histogram of Flight Price')

plt.show()
train['Price'] = np.log(train['Price'])
plt.figure(figsize=(8,8))

plt.hist(train.Price,bins=100,color='b')

plt.title('Histogram of Flight Price')

plt.show()
def tominutes(vr):

    s=re.findall("\d+", str(vr))

    if len(s)>1:

        return (int(s[0]) *60 ) + int(s[1])

    else:

        return (int(s[0])*60)





le = preprocessing.LabelEncoder()

for dataset in [train,test]:

    dataset['airline_en']=le.fit_transform(dataset['Airline'])

    dataset['doj_year_en']=pd.to_datetime(dataset['Date_of_Journey']).dt.year

    dataset['doj_month_en']=pd.to_datetime(dataset['Date_of_Journey']).dt.month

    dataset['doj_day_en']=pd.to_datetime(dataset['Date_of_Journey']).dt.day

    dataset['doj_dayofyear_en']=pd.to_datetime(dataset['Date_of_Journey']).dt.dayofyear

    dataset['doj_dayofweek_en']=pd.to_datetime(dataset['Date_of_Journey']).dt.dayofweek

    dataset['doj_week_en']=pd.to_datetime(dataset['Date_of_Journey']).dt.week

    dataset['doj_weekofyear_en']=pd.to_datetime(dataset['Date_of_Journey']).dt.weekofyear

    dataset['doj_weekday_en']=pd.to_datetime(dataset['Date_of_Journey']).dt.weekday

    dataset['doj_quarter_en']=pd.to_datetime(dataset['Date_of_Journey']).dt.quarter

    dataset['ym']=dataset['doj_year_en'].astype(str).str.cat(dataset['doj_dayofyear_en'].astype(str),sep='.').astype(float)

    #dataset.loc[pd.to_datetime(dataset['Date_of_Journey']).isin(ind_hol), 'hol_en'] = 0

    

    dataset['source_en']=le.fit_transform(dataset['Source'])

    dataset.loc[(dataset.Destination=='New Delhi'),'Destination'] = 'Delhi'

    dataset['dest_en']=le.fit_transform(dataset['Destination'])

    dataset['route_merged']= dataset.Source.astype(str).str.cat(dataset.Destination.astype(str), sep='_')

    dataset['route_merged_en']=le.fit_transform(dataset['route_merged'])

    dataset['route_en']=le.fit_transform(dataset['Route'])

    dataset['dep_hr_en']=pd.DatetimeIndex(dataset['Dep_Time']).hour

    dataset['dep_min_en']=pd.DatetimeIndex(dataset['Dep_Time']).minute

    dataset['dep_time_en']=dataset['dep_hr_en'].astype(str).str.cat(dataset['dep_min_en'].astype(str),sep='.').astype(float)

    dataset['arr_hr_en']=pd.DatetimeIndex(dataset['Arrival_Time']).hour

    dataset['arr_min_en']=pd.DatetimeIndex(dataset['Arrival_Time']).minute

    dataset['arr_time_en']=dataset['arr_hr_en'].astype(str).str.cat(dataset['arr_min_en'].astype(str),sep='.').astype(float)

    dataset['duration_en']=dataset['Duration'].map(lambda x: tominutes(x))

    dataset['total_stops_en']=le.fit_transform(dataset['Total_Stops'])

    dataset['Additional_Info_en']=le.fit_transform(dataset['Additional_Info'])

    dataset['dep_slot_en']=le.fit_transform(pd.cut(pd.DatetimeIndex(dataset['Dep_Time']).hour,[-1, 12, 17, 24],labels=['Morning', 'Afternoon', 'Evening']))

    dataset['arr_slot_en']=le.fit_transform(pd.cut(pd.DatetimeIndex(dataset['Arrival_Time']).hour,[-1, 12, 17, 24],labels=['Morning', 'Afternoon', 'Evening']))

    #dataset['holiday_en'] = date(dataset['doj_year_en'],dataset['doj_month_en'],dataset['doj_day_en']).isin(ind_holidays)

    #cleanup_stops = {"Total_Stops": {"1 stop": 1, "2 stops": 2,"3 stops":3,"4 stops":4,"non-stop":0}}

    #dataset.replace(cleanup_stops, inplace=True)

    #dataset['total_stops_ens']=dataset['Total_Stops'].astype(int)
# monthly change of prices

ym_summary = train.groupby(['ym'])['Price'].agg(['mean','count'])



vmin = np.min(ym_summary['count'])

vmax = np.max(ym_summary['count'])

norm = colors.Normalize(vmin,vmax)



plt.figure(figsize=(15,7))

plt.scatter(x=np.arange(ym_summary.shape[0]), y =ym_summary['mean'],c= ym_summary['count'],

            s= ym_summary['count'],norm=norm ,alpha = 0.8, cmap='jet')



plt.plot(np.arange(ym_summary.shape[0]), ym_summary['mean'] ,'--')

plt.xticks(np.arange(ym_summary.shape[0]),ym_summary.index.values)

plt.yscale('log')

plt.xlabel('Year-Month')

plt.ylabel('Price (log scale)')

clb = plt.colorbar() 

clb.ax.set_title('number of sales')

plt.title('Averge Price by Month')

plt.show()
def scatter_p(fea,price):

    n_f = len(fea)

    n_row = n_f//3+1

    fig=plt.figure(figsize=(20,15))

    i = 1

    for f in fea:

        x=train[f]

        y=train[price]

        m, b = np.polyfit(x, y, 1)    

        

        ax=fig.add_subplot(n_row,3,i)

        plt.plot(x,y,'.',color='b')

        plt.plot(x, m*x + b, '-',color='r')

        plt.xlabel(f)

        plt.ylabel(price)

        i += 1
scatter_p(fea=['airline_en','source_en', 'dest_en','total_stops_en','route_en','Additional_Info_en'],price='Price')
scatter_p(fea=['dep_time_en','arr_time_en', 'duration_en','ym'],price='Price')
train['log_dep_time_en'] = np.log(train['dep_time_en'])

train['log_arr_time_en'] = np.log(train['arr_time_en'])

test['log_dep_time_en'] = np.log(test['dep_time_en'])

test['log_arr_time_en'] = np.log(test['arr_time_en'])
scatter_p(fea=['log_dep_time_en','log_arr_time_en', 'duration_en','ym'],price='Price')
train_dum=train[['airline_en','source_en', 'dest_en','total_stops_en','route_en','log_dep_time_en','log_arr_time_en', 'duration_en','ym','Additional_Info_en','Price']]

test_dum=test[['airline_en','source_en', 'dest_en','total_stops_en','route_en','log_dep_time_en','log_arr_time_en', 'duration_en','ym','Additional_Info_en']]

#train_dum = pd.get_dummies(train_dum,columns=['Airline','Source','Destination','Total_Stops'],prefix=['dum_','src_','dest_','st_'],drop_first=True)

#test_dum = pd.get_dummies(test_dum,columns=['Airline','Source','Destination','Total_Stops'],prefix=['dum_','src_','dest_','st_'],drop_first=True)

#train_dum.drop(['airline_en','source_en', 'dest_en','total_stops_en'],axis=1,inplace=True)

#test_dum.drop(['airline_en','source_en', 'dest_en','total_stops_en'],axis=1,inplace=True)
corr = train_dum.corr()

mask = np.zeros_like(corr,dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f,ax = plt.subplots(figsize=(8,8))

cmap=sns.diverging_palette(120,10,as_cmap=True) #

sns.heatmap(corr,mask=mask,cmap=cmap,center=0,square=False,linewidths=.5,cbar_kws={"shrink":.5})
X=train_dum.loc[:,train_dum.columns!='Price']

y=train_dum.loc[:,train_dum.columns=='Price']

#train_dum,test_dum = X.align(test_dum, join='outer', axis=1, fill_value=0)
#Regression Library

from sklearn.linear_model import LinearRegression,ElasticNet,Lasso,Ridge,BayesianRidge

from lightgbm.sklearn import LGBMRegressor

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor,BaggingRegressor,VotingClassifier

from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb

from sklearn import neighbors

from sklearn import svm



from sklearn.model_selection import KFold,train_test_split

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction



from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score
def RMSE(estimator,X_train, Y_train, cv,n_jobs=-1):

    cv_results = cross_val_score(estimator,X_train,Y_train,cv=cv,scoring="neg_mean_squared_error",n_jobs=n_jobs)

    return (np.sqrt(-cv_results)).mean()
def baseModels(train_X,train_y):

    model_EN=ElasticNet(random_state=0)

    model_SVR=svm.SVR(kernel='rbf',C=0.005)

    model_Lasso=Lasso(alpha=0.1,max_iter=1000)

    model_Ridge=Ridge(alpha=0.1)

    model_Linear=LinearRegression()

    model_LGBM=LGBMRegressor(boosting='gbdt', n_jobs=-1, random_state=2018)

    model_GBR=GradientBoostingRegressor(n_estimators=100,alpha=0.01)

    model_XGB = xgb.XGBRegressor(n_estimators=100, learning_rate=0.02, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=4)

    model_DTR = DecisionTreeRegressor(max_depth=4,min_samples_split=5,max_leaf_nodes=10)

    model_RFR=RandomForestRegressor(n_jobs=-1)

    model_KNN=neighbors.KNeighborsRegressor(3,weights='uniform')

    model_Bayesian=BayesianRidge()

    model_adaboost=AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None)

    model_bagreg=BaggingRegressor(base_estimator=None, n_estimators=100,  bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)

    kf = KFold(n_splits=5, random_state=None, shuffle=True)



    models={'ElasticNet':model_EN,'SVR':model_SVR,'Lasso':model_Lasso,'Ridge':model_Ridge,'LGBM':model_LGBM,

            'GBR':model_GBR,'XGB':model_XGB,'DTR':model_DTR,'RandomForest':model_RFR,'KNN':model_KNN,

            'Bayes':model_Bayesian,'Linear':model_Linear,'AdaBoost':model_adaboost,'Bagging':model_bagreg}

    rmse=[]

    for model in models.values():

        #scores = cross_val_score(model, train_X, train_y, cv=kf,scoring='neg_mean_squared_error')

        #rmse.append(abs(scores.mean()))

        rmse.append(RMSE(model,train_X,train_y,kf))                         

    dataz = pd.DataFrame(data={'RMSE':rmse},index=models.keys())

    return  dataz
baseModels(X,y)
model_rf=RandomForestRegressor(n_jobs=-1)

RMSE(model_rf,X,y,10)
model_LGBM=LGBMRegressor(boosting='gbdt',learning_rate=0.5, n_estimators=850,num_leaves=8,random_state=400,

                        colsample_bytree=0.65,subsample=0.7,reg_alpha=1,reg_lambda=1.4,n_jobs=-1)

RMSE(model_LGBM,X,y,10)
model_bagreg=BaggingRegressor(base_estimator=None, n_estimators=600,  bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=500, verbose=0)

RMSE(model_bagreg,X,y,10)
from mlxtend.regressor import StackingRegressor

model_SVR=svm.SVR(kernel='rbf',C=0.02)

stregr = StackingRegressor(regressors=[model_bagreg, model_LGBM, model_rf], 

                           meta_regressor=model_LGBM)

RMSE(stregr,X,y,5)
stregr.fit(X,y)

preds = stregr.predict(test_dum)

submission = pd.DataFrame()

submission['Price'] = np.exp(preds)

submission.head()

submission.to_excel('submission.xlsx')