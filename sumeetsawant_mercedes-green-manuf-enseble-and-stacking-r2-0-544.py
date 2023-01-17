# Import the required library 



import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns 

import os 

import zipfile

import glob 



import tensorflow as tf 

import tensorflow_addons as tfa

from tensorflow.keras import layers

from tensorflow.keras import optimizers 

from tensorflow.keras.models import Model,load_model

from keras.callbacks import Callback

import ml_metrics



from sklearn import preprocessing 



%matplotlib inline 

plt.style.use('fivethirtyeight')



import warnings

warnings.filterwarnings('ignore')


path ='/kaggle/input/mercedes-benz-greener-manufacturing'



working_path='/kaggle/working'





if (os.getcwd()!=path):

    os.chdir(path)

    

#Uzip the data 



for file in glob.glob('*.zip'):

    with zipfile.ZipFile(os.path.join(path,file), 'r') as zip_ref:

        zip_ref.extractall(working_path)





os.chdir(working_path)



# Import the dataset 





df_train =pd.read_csv('./train.csv')



df_test=pd.read_csv('./test.csv')



df_submission=pd.read_csv('./sample_submission.csv')



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 150)

df_train.head()
df_train.shape, df_test.shape
df_dtypes=pd.DataFrame({'col':df_train.columns,'dtypes':df_train.dtypes}).reset_index(drop=True)

np.transpose(df_dtypes[:400])
# lets see the distribution of y variable 

plt.figure(figsize=(15,8))

plt.subplot(1, 2, 1)

plt.scatter(range(df_train.shape[0]), np.sort(df_train.y.values));

plt.xlabel('index', fontsize=12);

plt.ylabel('y', fontsize=12);



plt.subplot(1,2,2)

plt.hist(df_train.y,bins=15)

plt.xlim(40,180)

plt.xlabel('y', fontsize=12);

plt.ylabel('counts', fontsize=12);
df_train=df_train[df_train.y<=180]
df_train.isnull().sum().sum()
df_test.head()
df_test.isnull().sum().sum()
df_train['test']=0

df_test['test']=1



data=pd.concat([df_train,df_test],axis=0)



data.shape
# Lets seperate the int features 



interger_columns=[]



for col in data.columns:

    if col not in ['X0','X1','X2','X3','X4','X5','X6','X8']:

        interger_columns.append(col)

        

data=data[interger_columns]



data.head()
data.dtypes
col=[c for c in data.columns if c not in ['y','test','ID']]



# Feature engineering 



data['sum']=data[col].sum(axis=1)

data['mean']=data[col].mean(axis=1)

data['median']=data[col].median(axis=1)

data['skew']=data[col].skew(axis=1)

data['kurt']=data[col].kurtosis(axis=1)

data['mode']=data[col].mode(axis=1)



data.head()
# Lets split the data back  into test and train set 



train =data[data['test']!=1]

test =data[data['test']==1]





test.shape, train.shape
from sklearn import preprocessing



X=train.drop(columns=['ID','y'],axis=1)

y=train['y']





from sklearn import model_selection

X_train,X_val,y_train,y_val=model_selection.train_test_split(X,y,shuffle=True,test_size=0.1,random_state=101)
# Scale the Continous Variable 



normalize_col=['sum','skew','kurt']



for col in normalize_col:

    scaler=preprocessing.MinMaxScaler()

    scaler.fit(X_train[col].values.reshape(-1,1))

    X_train.loc[:,col]=scaler.transform(X_train[col].values.reshape(-1,1))

    X_val.loc[:,col]=scaler.transform(X_val[col].values.reshape(-1,1))

    test.loc[:,col]=scaler.transform(test[col].values.reshape(-1,1))

    

# Scaling Response variable 



y_scaler=preprocessing.MinMaxScaler()

y_scaler.fit(y_train.values.reshape(-1,1))

y_train=y_scaler.transform(y_train.values.reshape(-1,1))

y_val=y_scaler.transform(y_val.values.reshape(-1,1))


from sklearn import linear_model

reg = linear_model.RidgeCV(alphas=(0.001,0.01,0.1,0.3,0.003,1,5))



reg.fit(X_train,y_train)

pred_reg=reg.predict(X_val)





from sklearn import metrics 





#Actually not the right way to score but just wanted to use max data in RidgeCV model for traning . Completely leaving out val set from training gives R2 score ~0.59

print('The R2 score for Ridge Regression is {}'.format(metrics.r2_score(y_val,pred_reg)))
lasso=linear_model.Lasso(alpha=0.001,random_state=101)



lasso.fit(X_train,y_train)



pred_lasso=lasso.predict(X_val)



print('The R2 score for Lasso Regression is {}'.format(metrics.r2_score(y_val,pred_lasso)))
br=linear_model.BayesianRidge()



br.fit(X_train,y_train)



pred_bayesian=br.predict(X_val)



print('The R2 score for Bayesian Regression is {}'.format(metrics.r2_score(y_val,pred_bayesian)))
#from sklearn.model_selection import RandomizedSearchCV

#from sklearn import ensemble 



#params = {



#'n_estimators': [50,100,150,200,250],



## Number of features to consider at every split

#'max_features' : ['auto', 'sqrt'],



## Maximum number of levels in tree

#'max_depth' :[5,10,15,20,25],



## Minimum number of samples required to split a node

#'min_samples_split' : [2, 5, 10],



## Minimum number of samples required at each leaf node

#'min_samples_leaf': [1, 2, 4],



# Method of selecting samples for training each tree

#'bootstrap' : [True, False],

    

#'criterion':['mse', 'mae']

    

#}



#from sklearn.metrics import r2_score, make_scorer

#r2_scorer = make_scorer(r2_score)





#rf = ensemble.RandomForestRegressor()



#folds = 5

#param_comb = 20



#kfold = model_selection.KFold(n_splits=folds, shuffle = True, random_state = 101)



#random_search = RandomizedSearchCV(rf, param_distributions=params, n_iter=param_comb, scoring=r2_scorer, n_jobs=1, cv=kfold.split(X,y), verbose=5, random_state=101,refit=True )



#random_search.fit(X, y)



#print("The best score is {}".format(random_search.best_score_ ))



#print('/n')



#print ('The best paramerts are {}'.format(random_search.best_estimator_))
# From running the random search CV above  we get the following values 



from sklearn import ensemble



rf = ensemble.RandomForestRegressor(max_depth=10, max_features='sqrt', min_samples_leaf=4,n_estimators=50,random_state=101)

rf.fit(X_train,y_train)



rf_predict=rf.predict(X_val)

metrics.r2_score(y_val,rf_predict)


# Result Prediction with Lasso and Ridge 





test.drop('y',axis=1,inplace=True)



test_Ridge=reg.predict(test.drop('ID',axis=1))

test_Lasso=lasso.predict(test.drop('ID',axis=1))

test_Bayesian=br.predict(test.drop('ID',axis=1))

test_random_forest=rf.predict(test.drop('ID',axis=1))



test['Ridge_Prediction']=test_Ridge

test['Lasso_Prediction']=test_Lasso

test['Bayesian_Prediction']=test_Bayesian

test['Random_Prediction']=test_random_forest



test.loc[:,'y']=test[['Ridge_Prediction', 'Lasso_Prediction','Bayesian_Prediction','Random_Prediction']].mean(axis=1)
#Stacking all Validation dataset prediction into a dataframe 

df_stacking=pd.DataFrame(np.column_stack([y_val,pred_lasso,pred_reg,pred_bayesian,rf_predict]),

                         columns=['y','lasso','Ridge','Bayesian','rf'])



#All X_Test set predictions using all the above 4 models this will be later multiplied with the weights of the stacking model to get the final model 

df_stacking_test=pd.DataFrame(np.column_stack([test.ID,test_Lasso,test_Ridge,test_Bayesian,test_random_forest]),

                              columns=['ID','lasso','Ridge','Bayesian','rf'])



for col in df_stacking.columns:

    df_stacking.loc[:,col]=y_scaler.inverse_transform(df_stacking[col].values.reshape(-1,1))



for col in ['lasso','rf','Ridge','Bayesian']:

    df_stacking_test.loc[:,col]=y_scaler.inverse_transform(df_stacking_test[col].values.reshape(-1,1))
lr_stack=linear_model.LinearRegression()

lr_stack.fit(df_stacking[['lasso','rf','Ridge','Bayesian']].values,df_stacking['y'].values)



df_stacking_test.loc[:,'y']=lr_stack.predict(df_stacking_test[['lasso','rf','Ridge','Bayesian']])

df_stacking_test[['ID','y']].to_csv('/kaggle/working/Stacking_Integer.csv',index=False)
predictions=test.copy()
df_train['test']=0

df_test['test']=1



data=pd.concat([df_train,df_test],axis=0)





col=[c for c in data.columns if c not in ['y','test','ID','X0','X1','X2','X3','X4','X5','X6','X8']]

data['sum']=data[col].sum(axis=1)

data['mean']=data[col].mean(axis=1)

data['median']=data[col].median(axis=1)

data['skew']=data[col].skew(axis=1)

data['kurt']=data[col].kurtosis(axis=1)

data['mode']=data[col].mode(axis=1)





for col in ['X0','X1','X2','X3','X4','X5','X6','X8']:

    lbl_XG=preprocessing.LabelEncoder()

    data.loc[:,col]=lbl_XG.fit_transform(data[col].values.reshape(-1,1))

    

train=data[data.test!=1]

test=data[data.test==1]



train.drop(columns='test',axis=1,inplace=True)

test.drop(columns='test',axis=1,inplace=True)



X=train.drop(columns=['y','ID'],axis=1)

y=train.y



X_test=test.drop('y',axis=1)


from sklearn import model_selection



X_train,X_val,y_train,y_val=model_selection.train_test_split(X,y,shuffle=True,random_state=101,test_size=0.1)



normalize_col=['sum','skew','kurt']



from sklearn import preprocessing





for col in normalize_col:

    scaler=preprocessing.MinMaxScaler()

    scaler.fit(X_train[col].values.reshape(-1,1))

    X_train.loc[:,col]=scaler.transform(X_train[col].values.reshape(-1,1))

    X_val.loc[:,col]=scaler.transform(X_val[col].values.reshape(-1,1))

    X_test.loc[:,col]=scaler.transform(X_test[col].values.reshape(-1,1))

    

    

# Scaling Response variable 

y_scaler=preprocessing.MinMaxScaler()

y_scaler.fit(y_train.values.reshape(-1,1))

y_train=y_scaler.transform(y_train.values.reshape(-1,1))

y_val=y_scaler.transform(y_val.values.reshape(-1,1))
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV
#y_db=np.concatenate((y_train,y_val))

#X_db=pd.concat([X_train,X_val])







#params = {

#        'learning_rate':[0.01,0.1,1],

#        'n_estimators':[50,100,150,200,250],

#        'min_child_weight': [1, 5, 10],

#        'gamma': [0.5, 1, 1.5, 2, 5],

#        'subsample': [0.6, 0.8, 1.0],

#        'colsample_bytree': [0.6, 0.8, 1.0],

#        'max_depth': [5,10,15],

#        'reg_lambda':[0.5,1]

#        }



#xgb = xgb.XGBRegressor(objective ='reg:squarederror',\

#                    silent=False, nthread=1)



#folds = 5

#param_comb = 20



#from sklearn.metrics import r2_score, make_scorer

#r2_scorer = make_scorer(r2_score)





#kfold = model_selection.KFold(n_splits=folds, shuffle = True, random_state = 101)



#random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring=r2_scorer, n_jobs=1, cv=kfold.split(X_db,y_db), verbose=5, random_state=101,refit=True )



#random_search.fit(X_db, y_db)





#print("The best score is {}".format(random_search.best_score_ ))



#print('/n')



#print ('The best paramerts are {}'.format(random_search.best_estimator_))
#From a previous Random Search CV run 



xgb=xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=50,learning_rate=1,min_child_weight=5,gamma=2,

                     colsample_by_tree=0.6,max_depth=15,reg_lambda=0.75,subsample=1)



xgb.fit(X_train,y_train)





X_test['XGB_predict']=xgb.predict(X_test.drop("ID",axis=1))



predictions['XGB_predictions']=X_test['XGB_predict']



X_test['XGB_predict']=y_scaler.inverse_transform(X_test['XGB_predict'].values.reshape(-1,1))



X_test_final=X_test[['ID','XGB_predict']]





# Private LB score for just XGboost 0.53881

X_test_final.to_csv('/kaggle/working/XG_boost_solution.csv',index=False)

from sklearn import ensemble



rfc = ensemble.RandomForestRegressor(max_depth=10, max_features='sqrt', min_samples_leaf=4,n_estimators=50)

rfc.fit(X_train,y_train)



rfc_predict=rfc.predict(X_val)



metrics.r2_score(y_val,rfc_predict)
predictions['Random_forest_entire']=rfc.predict(X_test.drop(columns=['ID','XGB_predict'],axis=1))
# Private LB score =0.54222



predictions['y1']=predictions[['Ridge_Prediction','Lasso_Prediction','Random_Prediction','XGB_predictions','Bayesian_Prediction']].mean(axis=1)

predictions['y1']=y_scaler.inverse_transform(predictions['y1'].values.reshape(-1,1))
# Private LB score =0.54324



predictions['y2']=((0.40*predictions['XGB_predictions']+0.25*predictions['Lasso_Prediction']+0.25*predictions['Ridge_Prediction']+0.10*predictions['Random_Prediction']))

predictions['y2']=y_scaler.inverse_transform(predictions['y2'].values.reshape(-1,1))

# Private LB score = 0.54453 Best Score of the notebook 

predictions['y3']=(0.4*y_scaler.inverse_transform(predictions['XGB_predictions'].values.reshape(-1,1))+ 0.6*df_stacking_test['y'].values.reshape(-1,1))
# Private LB score =0.54357



predictions['y4']=(0.25*y_scaler.inverse_transform(predictions['XGB_predictions'].values.reshape(-1,1))+ 0.5*df_stacking_test['y'].values.reshape(-1,1)+\

                    0.25*y_scaler.inverse_transform(predictions['Random_forest_entire'].values.reshape(-1,1)))
# Private LB score =0.54091



predictions['y5']=(0.15*predictions['XGB_predictions']+0.30*predictions['Lasso_Prediction']+0.30*predictions['Ridge_Prediction']+0.10*predictions['Random_Prediction']\

                    +0.15*predictions['Random_forest_entire'])



predictions['y5']=y_scaler.inverse_transform(predictions['y5'].values.reshape(-1,1))

#predictions[['ID','y3']].to_csv('/kaggle/working/Stacking_XGboost_Random_forest.csv',index=False)