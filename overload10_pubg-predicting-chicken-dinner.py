import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
#pd.options.mode.use_inf_as_na = True
import os
os.listdir('../input')
pubg_train = pd.read_csv('../input/train_V2.csv')
pubg_test = pd.read_csv('../input/test_V2.csv') 
#Working with only 1000000 rows.

#pubg_train = pd.read_csv('../input/train_V2.csv',nrows=100000)
#pubg_test = pd.read_csv('../input/test_V2.csv',nrows=100000) 
print(pubg_train.head())
pubg_train.isnull().sum()
#pubg_train.loc[pubg_train['winPlacePerc'] == np.nan]
pubg_train.dropna(how='any',inplace=True)

pubg_train['matchType'].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()  
pubg_train['matchType'] = le.fit_transform(pubg_train['matchType'])
pubg_test['matchType'] = le.fit_transform(pubg_test['matchType'])
pubg_train['Id']= pd.factorize(pubg_train.Id)[0]
pubg_train['groupId']= pd.factorize(pubg_train.groupId)[0]
pubg_train['matchId']= pd.factorize(pubg_train.matchId)[0]

submission = pubg_test.loc[:,['Id']]
pubg_test['Id']= pd.factorize(pubg_test.Id)[0]
pubg_test['groupId']= pd.factorize(pubg_test.groupId)[0]
pubg_test['matchId']= pd.factorize(pubg_test.matchId)[0]

pubg_train.head()
#pubg_train.groupby('winPlacePerc').size()
#pubg_train.skew()

pubg_train.info(memory_usage='deep')
#Function to optimize the memory by downgrading the datatype to optimal length
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


dtype_list=[]
for col in pubg_train.columns:
    dtype_list.append(pubg_train[col].dtypes)
dtype_list=list(set(dtype_list))
print("Total Datatypes present: %s "%dtype_list)

# Analysing for Train dataset
for dtype in dtype_list:
    
    if 'int' in str(dtype):
        print("Analyse %s"%str(dtype))
        df_int=pubg_train.select_dtypes(include=[str(dtype)])
        converted_int = df_int.apply(pd.to_numeric,downcast='unsigned')
    
        print(mem_usage(df_int))
        print(mem_usage(converted_int))
        
    elif 'float' in str(dtype):
        print("Analyse %s"%str(dtype))
        df_float = pubg_train.select_dtypes(include=[str(dtype)])
        converted_float = df_float.apply(pd.to_numeric,downcast='float')
        
        print(mem_usage(df_float))
        print(mem_usage(converted_float))

print("Memory Usage of Original dataset: %s"%mem_usage(pubg_train))
pubg_train[converted_int.columns] = converted_int
pubg_train[converted_float.columns] = converted_float
print("Memory Usage of Optimized dataset: %s"%mem_usage(pubg_train))
pubg_train.info(memory_usage='deep')
# Analysing for Test dataset
for dtype in dtype_list:
    
    if 'int' in str(dtype):
        print("Analyse %s"%str(dtype))
        df_int=pubg_test.select_dtypes(include=[str(dtype)])
        converted_int = df_int.apply(pd.to_numeric,downcast='unsigned')
    
        print(mem_usage(df_int))
        print(mem_usage(converted_int))
        
    elif 'float' in str(dtype):
        print("Analyse %s"%str(dtype))
        df_float = pubg_test.select_dtypes(include=[str(dtype)])
        converted_float = df_float.apply(pd.to_numeric,downcast='float')
        
        print(mem_usage(df_float))
        print(mem_usage(converted_float))
print("Memory Usage of Original dataset: %s"%mem_usage(pubg_test))
pubg_test[converted_int.columns] = converted_int
pubg_test[converted_float.columns] = converted_float

print("Memory Usage of Optimized dataset: %s"%mem_usage(pubg_test))

pubg_train['playerJoined'] = pubg_train.groupby('matchId')['matchId'].transform('count')
pubg_test['playerJoined']=pubg_test.groupby('matchId')['matchId'].transform('count')

plt.figure(figsize=(15,10))
sns.countplot(x='playerJoined', data=pubg_train[pubg_train['playerJoined']>49])
plt.title("Players Joined",fontsize=15)
plt.show()

pubg_train['headShotsPerKill'] = pubg_train['headshotKills']/pubg_train['kills']
pubg_test['headShotsPerKill'] = pubg_test['headshotKills']/pubg_test['kills']

pubg_train['headShotsPerKill'].fillna(0,inplace=True)
pubg_test['headShotsPerKill'].fillna(0,inplace=True)

pubg_train['headShotsPerKill'].replace(np.inf,0,inplace=True)
pubg_test['headShotsPerKill'].replace(np.inf,0,inplace=True)
pubg_train['distTravelled'] = pubg_train['rideDistance']+pubg_train['swimDistance']+pubg_train['walkDistance']
pubg_test['distTravelled'] = pubg_test['rideDistance']+pubg_test['swimDistance']+pubg_test['walkDistance']

#pubg_train.head()
pubg_train['itemsUsed'] = pubg_train['heals']+pubg_train['boosts']
pubg_test['itemsUsed'] = pubg_test['heals']+pubg_test['boosts']
#pubg_train['roadKillsperRide'] =  pubg_train['roadKills']/pubg_train['rideDistance']
#pubg_test['roadKillsperRide'] =  pubg_test['roadKills']/pubg_test['rideDistance']


#pubg_train['roadKillsperRide'].replace([np.inf,-np.inf],np.nan)
#pubg_train['roadKillsperRide'].fillna(0,inplace=True)
#pubg_test['roadKillsperRide'].fillna(0,inplace=True)
#pubg_train['roadKillsperRide'].replace(np.inf,0,inplace=True)
#pubg_test['roadKillsperRide'].replace(np.inf,0,inplace=True)
#pubg_train.head()
#pubg_train['roadKillsperRide_log'] = np.log(1+pubg_train.roadKillsperRide)
#pubg_test['roadKillsperRide_log'] = np.log(1+pubg_test.roadKillsperRide)
#pubg_train.drop(columns='roadKillsperRide_log',inplace=True)
pubg_train.skew()
#pubg_train['headShotsPerKill'].unique()
#type(0.73684211)
#type(pubg_train['headShotsPerKill'][12])
#pubg_train['headShotsPerKill'][12]
pubg_train['teamwork']=pubg_train['revives']+pubg_train['assists']
pubg_test['teamwork']=pubg_test['revives']+pubg_test['assists']
#pubg_train.head()
pubg_train['killpoints_norm'] = (pubg_train['killPoints']*pubg_train['playerJoined'])/100
pubg_train['damageDealt_norm']= (pubg_train['damageDealt']*pubg_train['playerJoined'])/100
pubg_train['kills_norm'] = (pubg_train['kills']*pubg_train['playerJoined'])/100
pubg_train['rankPoints_norm']= (pubg_train['rankPoints']*pubg_train['playerJoined'])/100
pubg_train['roadKills_norm']= (pubg_train['roadKills']*pubg_train['playerJoined'])/100
pubg_train['teamKills_norm']= (pubg_train['teamKills']*pubg_train['playerJoined'])/100
pubg_train['winPoints_norm']= (pubg_train['winPoints']*pubg_train['playerJoined'])/100
pubg_train['headShotsPerKill_norm']= (pubg_train['headShotsPerKill']*pubg_train['playerJoined'])/100
#pubg_train['roadKillsperRide_norm']= (pubg_train['roadKillsperRide']*pubg_train['playerJoined'])/100
pubg_train['teamwork_norm']= (pubg_train['teamwork']*pubg_train['playerJoined'])/100
pubg_train['roadKills_log'] = np.log(1+pubg_train.roadKills)
pubg_test['roadKills_log'] = np.log(1+pubg_test.roadKills)
pubg_test['killpoints_norm'] = (pubg_test['killPoints']*pubg_test['playerJoined'])/100
pubg_test['damageDealt_norm']= (pubg_test['damageDealt']*pubg_test['playerJoined'])/100
pubg_test['kills_norm'] = (pubg_test['kills']*pubg_test['playerJoined'])/100
pubg_test['rankPoints_norm']= (pubg_test['rankPoints']*pubg_test['playerJoined'])/100
pubg_test['roadKills_norm']= (pubg_test['roadKills']*pubg_test['playerJoined'])/100
pubg_test['teamKills_norm']= (pubg_test['teamKills']*pubg_test['playerJoined'])/100
pubg_test['winPoints_norm']= (pubg_test['winPoints']*pubg_test['playerJoined'])/100
pubg_test['headShotsPerKill_norm']= (pubg_test['headShotsPerKill']*pubg_test['playerJoined'])/100
#pubg_test['roadKillsperRide_norm']= (pubg_test['roadKillsperRide']*pubg_test['playerJoined'])/100
pubg_test['teamwork_norm']= (pubg_test['teamwork']*pubg_test['playerJoined'])/100



pubg_train['vehicleDestroys_log'] = np.log(1+pubg_train.vehicleDestroys)
pubg_test['vehicleDestroys_log'] = np.log(1+pubg_test.vehicleDestroys)
pubg_train['swimDistance_log']=np.log(1+pubg_train.swimDistance)
pubg_test['swimDistance_log']=np.log(1+pubg_test.swimDistance)
pubg_train['teamKills_log']=np.log(1+pubg_train.teamKills)
pubg_test['teamKills_log']=np.log(1+pubg_test.teamKills)



#pubg_train.head()

#pubg_train.drop(columns=['killPoints','damageDealt','kills','rankPoints','roadKills','teamKills','winPoints','headShotsPerKill','roadKillsperRide','teamwork'],inplace=True);
#pubg_test.drop(columns=['killPoints','damageDealt','kills','rankPoints','roadKills','teamKills','winPoints','headShotsPerKill','roadKillsperRide','teamwork'],inplace=True);

#pubg_train.skew()

pubg_train.drop(columns=['Id','groupId','matchId',],inplace=True)
pubg_test.drop(columns=['Id','groupId','matchId',],inplace=True)
pubg_train.head()
pubg_test.head()

from sklearn.model_selection import train_test_split

X=pubg_train.drop(['winPlacePerc'],axis=1)
y=pubg_train['winPlacePerc']


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22,shuffle=True,stratify=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22,shuffle=True)
print('X_train.shape %s, X_test.shape %s\ny_train.shape %s, y_test.shape %s'%(X_train.shape,X_test.shape,y_train.shape,y_test.shape))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=False)
# Fit only to the training data
scaler.fit(X_train)
Scaled_X_train = scaler.transform(X_train)
Scaled_X_test = scaler.transform(X_test)
'''scaler = StandardScaler()
Scaled_X_train = scaler.fit_transform(X_train)
Scaled_X_test = scaler.fit_transform(X_test)'''
#X_train['roadKillsperRide'].max()
#y_train.max()
#np.isnan(np.min(X_train))
#np.max(X_train['roadKillsperRide'])
#type(X_train)
#pubg_train.info()
from sklearn import model_selection
#performance metrics

#Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
#Classifiers
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
sample_X_train = Scaled_X_train[:100000,]
sample_y_train = y_train[:100000]
#y_test.shape
sample_X_train.shape
def randomized_search():
    pipeline1 = Pipeline((
    ('clf', RandomForestRegressor(max_depth=500,max_features=None)),    
    ))

    pipeline2 = Pipeline((
    ('clf', LogisticRegression(class_weight='balanced',penalty='l2',solver='saga',max_iter=100)),
    ))

    pipeline3 = Pipeline((
    ('clf', MLPRegressor()),
    ))

    pipeline4 = Pipeline((
    ('clf', DecisionTreeRegressor(splitter='random')),
    ))

    parameters1 = {
    'clf__n_estimators': [ 100],    
    #'clf__max_features': ['auto', 'sqrt', 'log2',None],
    #'clf__max_depth': [500,600,700]   
    }

    parameters2 = { 
    'clf__C':[0.1,0.001,1,10]
    }

    parameters3 = {        
    'clf__alpha':[0.001,0.00001,0.0001],
    'clf__max_iter':[250,400,500],
    'clf__batch_size':[100,500,1000]
    }
    
    parameters4 = {        
    'clf__max_depth': [10,100,500,1000],
    'clf__max_features': ['auto', 'sqrt', 'log2', None],
    'clf__min_weight_fraction_leaf':[0,0.25,0.5]
    }

    pars = [(parameters1,"RandomForestRegressor"), (parameters2,"LogisticRegression"), (parameters3,"MLPRegressor"), (parameters4,"DecisionTreeRegressor")]
    pips = [pipeline1, pipeline2, pipeline3, pipeline4]

    '''
    for i in range(len(pars)):    
        model,name = pars[i]
        print('-'*50)
        print( "Starting Randomized Search for %s" %name)        
        #rs = GridSearchCV(pips[1], pars[1], verbose=5, refit=False, n_jobs=3,cv=5)
        rs = RandomizedSearchCV(pips[i], model, verbose=5, refit=False, n_jobs=3,cv=5,random_state=22,n_iter=20)
        rs = rs.fit(sample_X_train, sample_y_train)
        print("Finished Randomized Search for %s"%name)
        print('Best Score %.5f'%rs.best_score_)
        print('Best Param %s'%rs.best_params_)
        print('-'*50)
    '''

    model,name = pars[0]
    print('-'*50)
    print( "Starting Randomized Search for %s" %name)        
        #rs = GridSearchCV(pips[1], pars[1], verbose=5, refit=False, n_jobs=3,cv=5)
    rs = RandomizedSearchCV(pips[0], model, verbose=5, refit=False, n_jobs=3,cv=5,random_state=22,n_iter=20)
    rs = rs.fit(sample_X_train, sample_y_train)
    print("Finished Randomized Search for %s"%name)
    print('Best Score %.5f'%rs.best_score_)
    print('Best Param %s'%rs.best_params_)
    print('-'*50)
        
#randomized_search()

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

#Create Linear Regression Model object
'''
reg = linear_model.LinearRegression()

#Train Model using train dataset
reg.fit(Scaled_X_train,y_train)

#Make Prediction using cross validation set
y_pred_reg = reg.predict(Scaled_X_test)

'''
# r2_score score: 1 is perfect prediction
'''
r2Score=r2_score(y_test,y_pred_reg)
print("Variance score (r2_score): %f"%r2Score)
print('Model accuracy:%.2f '%(r2Score*100))
 
print("Root mean squared error of test:%f"%sqrt(mean_squared_error(y_test,y_pred_reg)))'''



from sklearn.neural_network import MLPClassifier, MLPRegressor
'''
mlp = MLPRegressor(solver='adam',hidden_layer_sizes=(10,10,10),alpha=0.1, random_state=1)

mlp.fit(Scaled_X_train,y_train)

y_pred_mlp = mlp.predict(Scaled_X_test)
'''


# r2_score score: 1 is perfect prediction
'''
mlp_score=mlp.score(Scaled_X_test,y_test)
print("Variance score (r2_score): %f"%mlp_score)
print('Model accuracy:%.2f '%(mlp_score*100))

print("Root mean squared error of test:%f"%sqrt(mean_squared_error(y_test,y_pred_mlp)))
'''

#Collect Garbage 'see doc for more info'
import gc
gc.collect()
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
'''
rfr = RandomForestRegressor(max_depth=500,criterion='mse',max_features=None,n_estimators=75,bootstrap=True,min_samples_leaf=16,n_jobs=5,min_samples_split=4)
rfr.fit(Scaled_X_train,y_train)
y_pred_rfr = rfr.predict(Scaled_X_test)


# r2_score score: 1 is perfect prediction
rfr_score=r2_score(y_pred=y_pred_rfr,y_true=y_test)
print("Variance score (r2_score): %f"%rfr_score)
print('Model accuracy:%.2f '%(rfr_score*100))

print("Root mean squared error of test:%f"%sqrt(mean_squared_error(y_test,y_pred_rfr)))

'''
#kfold = model_selection.KFold(n_splits=10,random_state=22,shuffle=True,)
#cv_result = model_selection.cross_val_score(rfr,X=X_train,y=y_train, cv=kfold,scoring='r2',verbose=True)
import lightgbm as lgb
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',    
    'max_depth':-1,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,    
    'num_leaves':1000,
    'max_bin':1000,
    'verbose': 0
}
# create dataset for lightgbm
lgb_train = lgb.Dataset(Scaled_X_train, y_train)
lgb_eval = lgb.Dataset(Scaled_X_test, y_test, reference=lgb_train)
print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_eval                
                )
print('Starting predicting...')
# predict
y_pred = gbm.predict(Scaled_X_test, num_iteration=gbm.best_iteration)
print('Done !')
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
gbm_score=r2_score(y_pred=y_pred,y_true=y_test)
print("Variance score (r2_score): %f"%gbm_score)
print('Model accuracy:%.2f '%(gbm_score*100))

print("Root mean squared error of test:%f"%sqrt(mean_squared_error(y_test,y_pred)))

'''
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

dtr=DecisionTreeRegressor(max_depth=4)
dtr_boost=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300,random_state=12)


dtr.fit(X_train,y_train)

y_pred=dtr.predict(X_test)


r2_score score: 1 is perfect prediction
dtr_score=dtr.score(X_test,y_test)
print("Variance score (r2_score): %f"%dtr_score)
print('Model accuracy:%.2f '%(dtr_score*100))


takes a hell lot of time.....Do not use this!
dtr_boost.fit(X_train,y_train)

y_pred=dtr_boost.predict(X_test)

r2_score score: 1 is perfect prediction
dtrBoost_score=dtr_boost.score(X_test,y_test)
print("Variance score (r2_score): %f"%dtrBoost_score)
print('Model accuracy:%.2f '%(dtrBoost_score*100))
''' 
print('')
#Predicting the test dataset values
#Scaling the test dataset
scaled_test = scaler.transform(pubg_test)
final_pred2 = gbm.predict(scaled_test, num_iteration=gbm.best_iteration)
#final_pred2 = gbm.predict(scaled_test)
submission['winPlacePerc'] = final_pred2
submission.head()
submission.to_csv('final_submission.csv',index=False)