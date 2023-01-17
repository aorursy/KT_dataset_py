# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.svm import LinearSVR,SVR
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor

import warnings

def load_data(road='../input/'):
    gc.collect()
    df_train = pd.read_csv('{}train.csv'.format(road))
    df_test = pd.read_csv('{}test.csv'.format(road))
    sub = pd.DataFrame(columns=['id','formation_energy_ev_natom','bandgap_energy_ev'])
    sub.id = df_test['id']
    df_train.drop('id',axis=1,inplace=True)
    df_test.drop('id',axis=1,inplace=True)
    gc.collect()
    
    return df_train,df_test,sub

df_train,df_test,sub = load_data()
df_train.head()
df_train.describe()
def feature_engine(df):
    df['al'] = df['number_of_total_atoms'] * df['percent_atom_al']
    df['ga'] = df['number_of_total_atoms'] * df['percent_atom_ga']    
    df['in'] = df['number_of_total_atoms'] * df['percent_atom_in']    
    df['all'] = df['al'] + df['ga'] + df['in']
    df['spacegroup'] = df['spacegroup'].astype('object')
    
    try:
        df.drop(['formation_energy_ev_natom','bandgap_energy_ev'],axis=1,inplace=True)
    except:
        pass
    return pd.get_dummies(df)

Y1 = df_train['formation_energy_ev_natom']
Y2 = df_train['bandgap_energy_ev']
# Y1 = np.log1p(Y1)
# Y2 = np.log1p(Y2)
df_train = feature_engine(df_train)
df_test = feature_engine(df_test)
df_train.head()
def plot_norm(feature):
    sns.distplot(df_train[feature],fit=norm)
    plt.show()
    plt.scatter(df_train[feature],Y1,label='formation_energy_ev_natom',alpha=0.1,c='r')
    plt.legend()
    plt.title(feature)
    plt.show()
    plt.scatter(df_train[feature],Y2,label='bandgap_energy_ev',alpha=0.1,c='g')
    plt.legend()
    plt.title(feature)
    plt.show()
    
for col in ['number_of_total_atoms','percent_atom_al','percent_atom_ga','percent_atom_in','lattice_vector_1_ang','lattice_vector_1_ang','lattice_vector_3_ang']:
#     plot_norm(col)
    pass
def plot_plot(feature):
#     fig,ax = plt.subplots(1,2,figsize=(10,10))
    plt.subplot(121)
    plt.scatter(df_train[feature],Y1,label='formation_energy_ev_natom')
    plt.legend()
    plt.subplot(122)
    plt.scatter(df_train[feature],Y2,label='bandgap_energy_ev')
    plt.legend()
#     plt.
#     plt.show()
    

plot_plot('percent_atom_al')
x_train,x_test,y_train,y_test = train_test_split(df_train,Y2,test_size=0.2,random_state=7)
rs = StandardScaler()
x_train = rs.fit_transform(x_train)
x_test = rs.transform(x_test)

def validate_data(model,x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test):
    model.fit(x_train,y_train)
    ss = StandardScaler()
#     x_train = ss.fit_transform(x_train)
#     x_test = ss.transform(x_test)
    y_pred_ = model.predict(x_train)
    y_pred = model.predict(x_test)
    print('train:\n {}'.format(np.sqrt(MSE(np.log1p(y_train),np.log1p(y_pred_)))))
#     print(y_pred)
    print('test :\n {}'.format(np.sqrt(MSE(np.log1p(y_test),np.log1p(y_pred)))))

    
def make_sub(model1,model2):
    model1.fit(df_train,Y1)
    model2.fit(df_train,Y2)
    y_pred_ = model1.predict(df_test)
    y_pred = model2.predict(df_test)
    sub['formation_energy_ev_natom'] = y_pred_
    sub['bandgap_energy_ev'] = y_pred
    sub.to_csv('sub.csv',index=False)
    print('submit finished')


lr =LinearRegression()
validate_data(lr)


# # lsvr = LinearSVR()
# '''
# train:
#  0.39189838428500523
# test :
#  0.3961722538975962
# '''

# params = {'C':range(70,83),'gamma':[0.00045,0.0005,0.00055,0.0006]}

# svr = SVR(kernel='rbf')
# grid = GridSearchCV(svr,params,cv=10,scoring='neg_mean_squared_error',n_jobs=4)
# print('griding ...')
# grid.fit(df_train,Y1)
# print('grided.')
# print(np.sqrt(-grid.best_score_))
# print(grid.best_params_)
# '''
# FOR Y1:
# ========== 
# {'C': 80.0, 'gamma': 0.00043333333333333337} #Y1最优参数
# train:
#  0.07913795570714843l
# test :
#  0.09461798476834833
 
 
#  {'C': 98, 'gamma': 0.0004}Y2最佳参数
# train:
#  0.10085788749915815
# test :
#  0.1266636905806479
 
 
#  {'C': 80, 'gamma': 0.0005}
# train:
#  0.0977365610708793
# test :
#  0.1220943632350841
# '''

# svr.set_params(**{'C': 80, 'gamma': 0.0005})
# print('validate_data ...')
# validate_data(svr)
# cv_result = pd.DataFrame(grid.cv_results_)
# cv_result.sort_values(by='mean_train_score',ascending=False)

# make_sub(SVR(kernel='rbf',**{'C': 80.0, 'gamma': 0.00043333333333333337}),SVR(kernel='rbf',** {'C': 71, 'gamma': 0.0005}))
# gbr_params = {
#     'n_estimators':[10,50,100,200,300,400]
# }

# gbr = GradientBoostingRegressor()

# gbr_grid = GridSearchCV(gbr,gbr_params,cv=10,scoring='neg_mean_squared_error')
# gbr_grid.fit(df_train,Y1)
# print(gbr_grid.best_params_)
# gbr.set_params(**gbr_grid)
# validate_data(gbr)
sns.distplot(Y2,fit=norm)
plt.show()
start = time.time()

#随即森林的话加上这几个参数可以限制过拟合
params = {'max_features':[0.5,0.8],'min_samples_split':[3,6],'min_samples_leaf':[8,10,13]}
rfr = RandomForestRegressor(bootstrap=False,n_estimators=300,random_state=7)
grid = GridSearchCV(rfr,params,cv=10,scoring='neg_mean_squared_error')
# print('griding ...')
# grid.fit(df_train,Y1)
# print('grided.')
# print(grid.best_params_)
# print(grid.best_score_)
# rfr.set_params(**grid.best_params_)
# validate_data(rfr)

'''
{'n_estimators': 300}
-0.0019397235246521008
train:
 0.04755024260678095
test :
 0.10433543760745331
 
{'max_features': 3, 'min_samples_split': 10, 'n_estimators': 350}
train:
 0.023594949361497492
test :
 0.035265905887075705 
 
{'max_features': 7, 'min_samples_leaf': 10, 'min_samples_split': 3}
train:
 0.02699623360807393
test :
 0.035858930226791506
 
 n_estimators=300
 train:
 0.026120196179068262
test :
 0.03492471879080006
 
 n_estimators=100
 train:
 0.02618497421220294
test :
 0.034835206929937995
'''

print('spend time :{:.2f}s'.format(time.time() - start))
# rfr.set_params(**{'max_features': 7, 'min_samples_leaf': 10, 'min_samples_split': 3,'n_estimators':300})
# validate_data(rfr)
from lightgbm import LGBMRegressor
import lightgbm as lgb


def model_fit_lgb(model,model_params,x_train,y_train,early_stop_rounds=5):
    model_train = lgb.Dataset(x_train,y_train)
    print('cving...')
    cv_result = lgb.cv(model_params,
                       model_train,
                       early_stopping_rounds=early_stop_rounds,
                       nfold=50,
                       stratified=False,#回归问题加stratified=False！
                       shuffle=True,
                       num_boost_round=5000,
                       seed=0,
                       metrics='rmse')
    
    print('cv finished.')
    n_estimators = len(cv_result['rmse-mean'])#这里注意values的长度！
    print(n_estimators)
    model.set_params(n_estimators=n_estimators)   
    
'''
learning_rate=0.1,max_depth=6,subsample=0.8,subsample_freq=1,colsample_bytree=0.8，n_estimators=100
train:
 0.06019847356156149
test :
 0.10068118284715544
 
'''
lgb_params ={
    'learning_rate':0.1,'bagging_fraction':0.8,'feature_fraction':0.8,
    'num_leave':50, #加入num_leave防止过拟合
    'metrics':'rmse','bagging_freq':10
}

lgb1 = LGBMRegressor(**lgb_params)

model_fit_lgb(lgb1,lgb_params,x_train,y_train)
# model_fit(lgb_params,lgb1,x_train,y_train)
validate_data(lgb1)
'''
 train:
 0.06290452900706987
test :
 0.10081715525549317
'''
params =  {'max_depth':range(3,11),'num_leaves':range(55,65)}
grid = GridSearchCV(lgb1,params,cv=10,scoring='neg_mean_squared_error')
# grid.fit(df_train,Y2)
# print(grid.best_score_)
# print(grid.best_params_)

lgb2 = LGBMRegressor(n_estimators=61,learning_rate=0.1,bagging_fraction=0.8,feature_fraction=0.8,
                    max_depth=7,
                    num_leave=60, #加入num_leave防止过拟合
                    metrics='rmse',bagging_freq=10)
validate_data(lgb2)
'''
{'max_depth': 7, 'num_leaves': 60}
train:
 0.06499574811552297
test :
 0.09924309162427868
'''
params = {'min_child_samples':range(10,50,10),'min_child_weight':np.linspace(0.00005,0.0005,10)}
grid = GridSearchCV(lgb2,params,cv=10,scoring='neg_mean_squared_error')
# grid.fit(df_train,Y2)
# print(grid.best_score_)
# print(grid.best_params_)
# print(grid.best_params_)
lgb3 = LGBMRegressor(n_estimators=61,learning_rate=0.1,bagging_fraction=0.8,feature_fraction=0.8,
                    max_depth=7,
                    num_leave=60, #加入num_leave防止过拟合
                    metrics='rmse',bagging_freq=10,
                    min_child_samples=20,
                    min_child_weight=0.5,
                    )

'''
{'min_child_samples': 20, 'min_child_weight': 0.0001}
train:
 0.06499574811552297
test :
 0.09924309162427868
'''
validate_data(lgb3)
params = {'bagging_fraation':np.linspace(0.4,0.5,10),'feature_fraction':np.linspace(0.4,0.5,10)}
grid = GridSearchCV(lgb3,params,cv=10,scoring='neg_mean_squared_error')
# grid.fit(df_train,Y2)

# print(grid.best_score_)
# print(grid.best_params_)
lgb4 = LGBMRegressor(n_estimators=61,learning_rate=0.1,bagging_fraction=0.4,feature_fraction=0.4555555,
                    max_depth=7,
                    num_leave=60, #加入num_leave防止过拟合
                    metrics='rmse',bagging_freq=10,
                    min_child_samples=20,
                    min_child_weight=0.5,
                    )
# print(grid.best_params_)
# validate_data(lgb4)

'''
{'bagging_fraation': 0.4, 'feature_fraction': 0.4555555}
train:
 0.07466134994804485
test :
 0.09831102767307756
'''

params = {'reg_alpha':np.linspace(0.04,0.05,50),'reg_gamma':[0]}

grid =  GridSearchCV(lgb4,params,cv=10,scoring='neg_mean_squared_error')
# grid.fit(df_train,Y2)
# print(grid.best_score_)
# print(grid.best_params_)
lgb5 = LGBMRegressor(n_estimators=61,learning_rate=0.1,bagging_fraction=0.4,feature_fraction=0.4555555,
                    max_depth=7,
                    num_leave=60, #加入num_leave防止过拟合
                    metrics='rmse',bagging_freq=10,
                    min_child_samples=20,
                    min_child_weight=0.5,
                    reg_alpha= 0.04204081632653062,
                    reg_gamma=0,
                    )
# print(grid.best_score_)
# print(grid.best_params_)
# validate_data(lgb5)

'''
{'reg_alpha': 0.04140816326530612, 'reg_gamma': 0}
train:
 0.0744088692165038
test :
 0.09802256997806044
 
 {'reg_alpha': 0.04204081632653062, 'reg_gamma': 0}
train:
 0.07440965133014965
test :
 0.09802215359344574
 
'''
params = {
    'bagging_freq':range(10,70,5)
}
grid = GridSearchCV(lgb4,params,cv=10,scoring='neg_mean_squared_error')
# grid.fit(df_train,Y2)
# print(grid.best_score_)
# print(grid.best_params_)
# print(grid.best_params_)
lgb6 = LGBMRegressor(n_estimators=61,learning_rate=0.1,bagging_fraction=0.4,feature_fraction=0.4555555,
                    max_depth=7,
                    num_leave=60, #加入num_leave防止过拟合
                    metrics='rmse',bagging_freq=10,
                    min_child_samples=20,
                    min_child_weight=0.5,
                    reg_alpha= 0.04204081632653062,
                    reg_gamma=0,
                    )
validate_data(lgb6)
params = dict(n_estimators=61,learning_rate=0.1,bagging_fraction=0.4,feature_fraction=0.4555555,
                    max_depth=7,
                    num_leave=60, #加入num_leave防止过拟合
                    metrics='rmse',bagging_freq=10,
                    min_child_samples=20,
                    min_child_weight=0.5,
                    reg_alpha= 0.04204081632653062,
                    reg_gamma=0,)
model_fit_lgb(lgb5,params,x_train,y_train)
make_sub(SVR(kernel='rbf',**{'C': 80.0, 'gamma': 0.00043333333333333337}),lgb5)

# from xgboost import XGBRegressor
# import xgboost as xgb


# def model_fit(model,x_train=x_train,y_train=y_train,early_stop_round=50):
#     model_params = model.get_xgb_params()
    
#     model_train = xgb.DMatrix(x_train,y_train)
    
#     cv_result = xgb.cv(model_params,model_train,num_boost_round=5000,early_stopping_rounds=early_stop_round,metrics='rmse')
    
#     n_estimators = len(cv_result)
#     model.set_params(n_estimators=n_estimators)
#     print('n_estimators:{}'.format(n_estimators))
    


# xgb1 = XGBRegressor(learning_rate=0.1,max_depth=5,subsample=0.8,seed=7)
# model_fit(xgb1)
# # xgb1.fit(x_train,y_train)
# validate_data(xgb1)
# params = {'max_depth':[3,4,5],'min_child_weight':[5,6,7]}
# grid = GridSearchCV(xgb1,params,cv=10,scoring='neg_mean_squared_error')
# grid.fit(df_train,Y1)
# print(grid.best_params_)
# print(grid.best_score_)
# xgb2 = XGBRegressor(learning_rate=0.1,subsample=0.8,seed=7,max_depth= 4, min_child_weight= 7)
# validate_data(xgb2)
# params = {'gamma':[0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,]}
# grid = GridSearchCV(xgb2,params,cv=10,scoring='neg_mean_squared_error')
# grid.fit(df_train,Y1)
# print(grid.best_params_)
# print(grid.best_score_)
# xgb3 = XGBRegressor(learning_rate=0.1,subsample=0.8,seed=7,max_depth= 4, min_child_weight= 7,gamma=0.05)
# validate_data(xgb3)
# params = {
#     'subsample':np.linspace(0,1,6),
#     'colsample_bytree':np.linspace(0.1,1,6),
# }
# grid = GridSearchCV(xgb3,params,cv=10,scoring='neg_mean_squared_error')
# grid.fit(df_train,Y1)
# print(grid.best_score_)
# print(grid.best_params_)
# xgb4 = XGBRegressor(learning_rate=0.1,seed=7,max_depth= 4, min_child_weight= 7,gamma=0.05,subsample=0.8,colsample_bytree=0.82)
# validate_data(xgb4)
# params = {
#     'reg_alpha':[0.0009,0.00095,0.001,0.0014]
# }
# grid = GridSearchCV(xgb4,params,cv=10,scoring='neg_mean_squared_error')
# grid.fit(df_train,Y1)
# print(grid.best_score_)
# print(grid.best_params_)
# xgb5 = XGBRegressor(learning_rate=0.1,seed=7,max_depth= 4, min_child_weight= 7,gamma=0.05,subsample=0.8,colsample_bytree=0.82,reg_alpha=0.0009)
# validate_data(xgb5)
# params = {
#     'reg_lambda':[0.0014,0.005,0.006,0.01,0.02,0.06]
# }
# grid = GridSearchCV(xgb4,params,cv=10,scoring='neg_mean_squared_error')
# grid.fit(df_train,Y1)
# print(grid.best_score_)
# print(grid.best_params_)
# xgb6 = XGBRegressor(learning_rate=0.1,seed=7,max_depth= 4, min_child_weight= 7,gamma=0.05,subsample=0.8,colsample_bytree=0.82,reg_alpha=0.0009,reg_lambda=0.001)
# validate_data(xgb6)
# make_sub(xgb6,lr)
# params = {'learning_rate':[0.25,0.1]}
# xgb7 = XGBRegressor(learning_rate=0.01,seed=7,max_depth=4,min_child_weight=7,gamma=0.05,subsample=0.8,colsample_bytree=0.82,reg_alpha=0.0009,reg_lambda=0.001)
# grid = GridSearchCV(xgb7,params,cv=10,scoring='neg_mean_squared_error')
# grid.fit(df_train,Y1)
# print(grid.best_params_)
# print(grid.best_score_)
# cv_result = cross_val_score(xgb7,df_train,Y1,cv=10,scoring='neg_mean_squared_error')
# cv_result
# validate_data(xgb7)