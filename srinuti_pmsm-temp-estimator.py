#import files that are required for reading the data. 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
#plt.figure(figsize=(16,5))

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
import os
#print(os.listdir('../input'))
# create datafile

df= pd.read_csv('../input/electric-motor-temperature/pmsm_temperature_data.csv')
df.head()

df.info()
# all the columns shall be converted into float32 to reduce the file size. This increase the performance speed. Moreover the 
# accuracy beyond 7 digits is not critical. Atleast for prototype testing!

columns = list(df.columns[:-1])

for n in columns:
    df[n]= df[n].astype(np.float32)
    

# df['ambient']=df['ambient'].astype(np.float32)
df.info()
# first lets test on one profile id. Lets pick profile id == 4

def profile_id_df(dataframe, prof_id):
    '''
    Input:
    dataframe = Pandas dataframe 
    profile id = # profile id number out of df['profile_id'].unique()
    
    Output:
    filtered dataframe for a given profile id
    '''
       
    
    return dataframe.loc[dataframe['profile_id'] == prof_id]
#lets pick profile id 4 and carry out analysis on this

df_4 = profile_id_df(df,4)
df_4.shape
sns.jointplot(x='motor_speed', y='torque', data= df_4)

# looking at the plot there are lot junk data points. (Torque is directly proportional to motor speed) At zero motor speed
# torque cant increase to 2. Hence the data requires lot of cleaning! Also at various speed levels torque cant be zero. 
# So no clear explaintion provided along with the dataset. Hence we leave this as is and continue our journey of analysis. 
sns.jointplot(x='motor_speed', y='pm', data= df_4)
# dataframe split into x and y data
X = df_4.drop(['pm','profile_id'], axis = 1)
y = df_4['pm'] # rotor temperature 'pm'
# import sklearn 
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# Feature selection from selectKbest "Mutual info regression" is applicable for continous data type. Most of the other functions
# are for classification problems. 

method = SelectKBest(score_func= mutual_info_regression, k = 'all')

method.fit_transform(X_train, y_train)
# amazing results with straight correlation fit, i am unable to get the values greater than 0.5 correlation. 
# in this case the correlation you see is greater than 0.4 for some features. 

correlation_matrix = X_train.corr(method= 'pearson').abs()
#print(correlation_matrix)
upper_corr_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
#print(upper_corr_matrix)
plt.figure(figsize=(16,5))
sns.heatmap(data = upper_corr_matrix , cmap= 'YlGnBu', annot= True)
# filter the columns which have greater than 0.5 correlation !

to_filter = [column for column in upper_corr_matrix.columns if any (upper_corr_matrix[column] > 0.70)]
to_filter
# new reduced x input. 
X_new = df_4[to_filter]
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score, max_error,median_absolute_error, mean_squared_log_error

# function to evaulate performance of the regressor. 

def evaulation(model, y_pred, y_true):
    
    '''
    Input:- model = string (Name of the regressor)
    y_pred= model prediction
    y_true = actual labels. 
    
    Output:
    Dataframe with evaulation matrix. 
    
    '''
    
    # create data output frame for the evaluation. 
    data = [explained_variance_score(y_true,y_pred), 
            max_error(y_true,y_pred),
            mean_squared_error(y_true,y_pred),
            mean_absolute_error(y_true,y_pred),
            r2_score(y_true,y_pred, multioutput='uniform_average'),
            median_absolute_error(y_true,y_pred)           
            ]
    row_index = ['Exp_Var_Score', 'Max_Error','MSE','MAE','R2_Score', 'Median_Abs_Error']
    
    df = pd.DataFrame(data, columns= [model], index= row_index)
    
    return df
# Step1 Train test split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.3, random_state = 0)

# Step2 Initiate linear regressor
lr = LinearRegression()

# step3 fit the data
lr.fit(X_train, y_train)

# predict the test data
y_pred_lr = lr.predict(X_test)


#evaulation of the lr   

print('Intercept:', lr.intercept_)
print('Coefficients:', lr.coef_)


# Linear regressor evaulation parameters
df_linear = evaulation('linear', y_pred_lr, y_test)
df_linear
# Ridge regressor

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
ridge = Ridge()

params = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5]}

ridge_reg = GridSearchCV(ridge, params, scoring = 'neg_mean_squared_error', cv =5)

ridge_reg.fit(X_train, y_train)
ridge_alpha = ridge_reg.best_params_

print(ridge_alpha['alpha'])
print(ridge_reg.best_score_)


#Output:- 
#0.01 --> Alpha
#-0.06259961947798728 --> Best_score
    
# Displays various tests scores for each alpha value. Refer Rank_test_score to find out alpha = 0.01 is the best answer. 
ridge_reg.cv_results_
# now we got the optimum alpha value. Next step is to perform Ridge regression. 
ridge_reg_model= Ridge(alpha= 0.01)

ridge_reg_model.fit(X_train, y_train)

y_pred_ridge = ridge_reg.predict(X_test)


#evaulation of the Ridge  
print('Intercept:', ridge_reg_model.intercept_)
print('Coefficients:', ridge_reg_model.coef_)


# Linear regressor evaulation parameters
df_ridge = evaulation('ridge', y_pred_ridge, y_test)
df_ridge
from sklearn.metrics import r2_score
# default parameters before running gridsearch. 
svr = SVR(C=1, epsilon=0.2, kernel='rbf', gamma= 'scale', tol = 1e-6)



pipe = Pipeline( steps = [('Standardscaler', StandardScaler()), 
                          ('SVR', svr)])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)



#evaulation of the Ridge  
svr_reg = np.mean((y_pred - y_test)**2)
svr_reg_max = np.max((y_pred - y_test)**2)
svr_reg_min = np.min((y_pred - y_test)**2)

print('MSE:', svr_reg)
print('MSE Max:', svr_reg_max)
print('MSE Min:', svr_reg_min)
print('R2_score:', r2_score(y_test, y_pred))

# intercept and coefficients are available only for kernel = 'linear'

from sklearn.metrics import r2_score


# after performing gridsearch, following parameters yielded optiminum results. 

svr = SVR(C=80, epsilon=0.005, kernel='rbf', gamma=3, tol = .001, verbose = 0)



pipe = Pipeline( steps = [('Standardscaler', StandardScaler()), 
                          ('SVR', svr)])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
# intercept and coefficients are available only for kernel = 'linear'

df_svr = evaulation('SVR', y_pred, y_test)
df_svr
# the cell is commented, if you need to optimize further uncomment and modify the C, epsilon and gamma parameters. 
# the operation will take several hours to run. 

# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVR
# from sklearn.metrics import r2_score



# gsc = GridSearchCV(
#         estimator=SVR(kernel='rbf'),
#         param_grid={
#             'C': [80, 100, 120],
#             'epsilon': [0.001, 0.005],
#             'gamma': [3, 4, 5]
#         },
#         cv=20, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)


# grid_result = gsc.fit(X_test, y_test)

# best_parms = grid_result.best_params_


# best_svr = SVR(kernel='rbf', C=best_parms["C"], epsilon=best_parms["epsilon"], gamma=best_parms["gamma"],
#                coef0=0.1, shrinking=True, tol=0.001, cache_size=200, 
#                from sklearn.metrics import r2_score


# print(best_svr)

# output:
#     SVR(C=80, coef0=0.1, epsilon=0.005, gamma=3)
# from sklearn.metrics import r2_score

# kernel = ['rbf', 'poly', 'sigmoid', 'precomputed']

# svr = SVR(C=1, epsilon=0.2, kernel='poly', degree = 5)

# pipe = Pipeline( steps = [('Standardscaler', StandardScaler()), 
#                           ('SVR', svr)])

# pipe.fit(X_train, y_train)

# y_pred = pipe.predict(X_test)



# #evaulation of the Ridge  
# svr_reg = np.mean((y_pred - y_test)**2)
# svr_reg_max = np.max((y_pred - y_test)**2)
# svr_reg_min = np.min((y_pred - y_test)**2)

# print('MSE:', svr_reg)
# print('MSE Max:', svr_reg_max)
# print('MSE Min:', svr_reg_min)
# print('R2_score:', r2_score(y_test, y_pred))

# # intercept and coefficients are available only for kernel = 'linear'


# output:- 
# MSE: 0.03343396604718888
# MSE Max: 44.990648440689434
# MSE Min: 6.1491123972214586e-09
# R2_score: 0.9667921373579083

# Lasso Reg

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso


# Grid search analysis was done, 0.01 was the best alpha
#params = {'alpha': [.01]}

lasso = Lasso(alpha = .01)

lasso.fit(X_test, y_test)
y_pred_lasso = lasso.predict(X_test)
#pipe = Pipeline( steps = [('Standardscaler', StandardScaler()),('Lasso', lasso)])

#pipe.fit(X_train, y_train)

#y_pred_lasso = pipe.predict(X_test)

print('Coefficients',lasso.coef_)
print('Intercepts', lasso.intercept_)
#print('feature_name', X_train.columns)
df_lasso = evaulation('lasso', y_pred_lasso, y_test)
df_lasso
from sklearn.linear_model import ElasticNetCV, ElasticNet
elastic = ElasticNet(alpha = 0.01, l1_ratio = 0.5) # parameters were selected based on grid search 


elastic_score = elastic.fit(X_train, y_train)

y_pred_elastic = elastic_score.predict(X_test)


# evaulation
print('Intercept', elastic.intercept_)
print('Coefficients', elastic.coef_)

df_Elastic = evaulation('ElasticNet', y_pred_elastic, y_test)
df_Elastic
df_summary= pd.concat([df_Elastic, df_lasso, df_svr, df_ridge, df_linear], axis=1, sort=False)
df_summary
import tensorflow as tf
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense
# split the dataset into train, test, and validation sets

X_train_full, X_test, y_train_full, y_test = train_test_split(X_new, y, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)


# scale the datasets
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

def build_model():
    model = keras.models.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=X_train.shape[1:]),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation = 'linear')
        ])
    optimizer = tf.keras.optimizers.Adam(lr = 1e-5)
    
    model.compile(loss= 'mean_squared_error', optimizer= optimizer, metrics = ['mse'])
    
    return model

model = build_model()
model.summary()
EPOCHS = 100
history = model.fit(X_train, y_train, batch_size= 50, epochs= EPOCHS, verbose=0)
history_vald= model.fit(X_valid, y_valid, batch_size=32, epochs=EPOCHS, verbose = 0)
plt.plot(pd.DataFrame(history.history))
plt.plot(pd.DataFrame(history_vald.history))
plt.grid(True)
plt.gca().set_ylim(0,0.5)
plt.show()
y_pred_ANN = model.predict(X_test, batch_size=32, verbose= 0)
#y_pred_ANN
#y_test.values
mse_ANN = np.mean((y_pred_ANN - y_test.values)**2)
print(mse_ANN)
((y_pred_ANN - y_test.values)**2).max()
(y_pred_ANN - y_test.values).min()
y_pred_ANN
#plt.plot(y_test.values)
#plt.plot(y_pred_ANN)
plt.grid(True)
plt.plot(y_test.values, y_pred_ANN)
