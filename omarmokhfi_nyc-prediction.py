import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn import metrics

from sklearn.model_selection import train_test_split, GridSearchCV

from haversine import haversine

import statsmodels.formula.api as sm

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

import warnings; warnings.simplefilter('ignore')
#import the data from a csv file.

data = pd.read_csv("../input/nyc-taxi-trip-duration/train.zip")
data.head()
#Convert timestamp to datetime format to fetch the other details as listed below

data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

#data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])
#Calculate and assign new columns to the dataframe such as weekday,

#month and pickup_hour which will help us to gain more insights from the data.

#data['weekday'] = data.pickup_datetime.dt.weekday_name

data['month'] = data.pickup_datetime.dt.month

data['weekday_num'] = data.pickup_datetime.dt.weekday

data['pickup_hour'] = data.pickup_datetime.dt.hour
#calc_distance is a function to calculate distance between pickup and dropoff coordinates using Haversine formula.

def calc_distance(df):

    pickup = (df['pickup_latitude'], df['pickup_longitude'])

    drop = (df['dropoff_latitude'], df['dropoff_longitude'])

    return haversine(pickup, drop)
#Calculate distance and assign new column to the dataframe.

if 'distance' not in data.columns:

    data['distance'] = data.apply(lambda x: calc_distance(x), axis = 1)
#Calculate Speed in km/h for further insights

if 'speed' not in data.columns:

    data['speed'] = (data.distance/(data.trip_duration/3600))
#Dummify all the categorical features like "store_and_fwd_flag, vendor_id, month, weekday_num, pickup_hour, passenger_count" except the label i.e. "trip_duration"



dummy = pd.get_dummies(data.month, prefix='month')

dummy.drop(dummy.columns[0], axis=1, inplace=True) #avoid dummy trap

data = pd.concat([data,dummy], axis = 1)



dummy = pd.get_dummies(data.weekday_num, prefix='weekday_num')

dummy.drop(dummy.columns[0], axis=1, inplace=True) #avoid dummy trap

data = pd.concat([data,dummy], axis = 1)



dummy = pd.get_dummies(data.pickup_hour, prefix='pickup_hour')

dummy.drop(dummy.columns[0], axis=1, inplace=True) #avoid dummy trap

data = pd.concat([data,dummy], axis = 1)



if 'passenger_count' in data.columns:

    dummy = pd.get_dummies(data.passenger_count, prefix='passenger_count')

    dummy.drop(dummy.columns[0], axis=1, inplace=True) #avoid dummy trap

    data = pd.concat([data,dummy], axis = 1)
data.head()
data['passenger_count'] = data.passenger_count.map(lambda x: 1 if x == 0 else x)
data = data[data.passenger_count <= 6]
data = data[data.trip_duration <= 86400]
data = data[data.speed <= 104]
data = data[~((data.distance == 0) & (data.trip_duration >= 60))]
duo = data.loc[(data['distance'] <= 1) & (data['trip_duration'] >= 3600),['distance','trip_duration']].reset_index(drop=True)
data = data[~((data['distance'] <= 1) & (data['trip_duration'] >= 3600))]
data = data[data.pickup_longitude != data.pickup_longitude.min()]
data = data[data.pickup_longitude != data.pickup_longitude.min()]

#map_marker(data)
#First chech the index of the features and label

del data['id']

del data['dropoff_datetime']

del data['passenger_count']

del data['store_and_fwd_flag']

list(zip( range(0,len(data.columns)),data.columns))
Y = data.iloc[:,6].values

del data['trip_duration']

del data['pickup_datetime']

del data['vendor_id']

list(zip( range(0,len(data.columns)),data.columns))
X = data.iloc[:,range(0,46)].values
X1 = np.append(arr = np.ones((X.shape[0],1)).astype(int), values = X, axis = 1)
X1.shape
#Select all the features in X array

X_opt = X1[:,range(0,47)]

#regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()



#Fetch p values for each feature

#p_Vals = regressor_OLS.pvalues



#define significance level for accepting the feature.

#sig_Level = 0.05



#Loop to iterate over features and remove the feature with p value less than the sig_level

#while max(p_Vals) > sig_Level:

    #print("Probability values of each feature \n")

    #print(p_Vals)

    #X_opt = np.delete(X_opt, np.argmax(p_Vals), axis = 1)

    #print("\n")

    #print("Feature at index {} is removed \n".format(str(np.argmax(p_Vals))))

    #print(str(X_opt.shape[1]-1) + " dimensions remaining now... \n")

    #regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

    #p_Vals = regressor_OLS.pvalues

    #print("=================================================================\n")

    

#Print final summary

#print("Final stat summary with optimal {} features".format(str(X_opt.shape[1]-1)))

#regressor_OLS.summary()
#Split raw data

X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=4, test_size=0.2)



#Split data from the feature selection group

X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_opt,Y, random_state=4, test_size=0.2)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X,Y, random_state=4, test_size=0.2)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_pca = scaler.fit_transform(X_train_pca)

X_test_pca = scaler.transform(X_test_pca)
from sklearn.decomposition import PCA

pca = PCA().fit(X_train_pca)

#.plot(np.cumsum(pca.explained_variance_ratio_))

#plt.xlabel("number of components")

#plt.ylabel("Cumulative explained variance")

#plt.show()
arr = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

list(zip(range(1,len(arr)), arr))
pca_10 = PCA(n_components=43)

X_train_pca = pca_10.fit_transform(X_train_pca)

X_test_pca = pca_10.transform(X_test_pca)
#Linear regressor for the raw data

#regressor = LinearRegression() 

#regressor.fit(X_train,y_train) 



#Linear regressor for the Feature selection group

#regressor1 = LinearRegression() 

#regressor1.fit(X_train_fs,y_train_fs) 



#Linear regressor for the Feature extraction group

regressor2 = LinearRegression() 

regressor2.fit(X_train_pca,y_train_pca) 
#Predict from the test features of raw data

#y_pred = regressor.predict(X_test) 



#Predict from the test features of Feature Selection group

#y_pred = regressor1.predict(X_test_fs) 



#Predict from the test features of Feature Extraction group

y_pred_pca = regressor2.predict(X_test_pca) 
#Evaluate the regressor on the raw data

#print('RMSE score for the Multiple LR raw is : {}'.format(np.sqrt(metrics.mean_squared_error(y_test,y_pred))))

#print('Variance score for the Multiple LR raw is : %.2f' % regressor.score(X_test, y_test))

#print("\n")



#Evaluate the regressor on the Feature selection group

#print('RMSE score for the Multiple LR FS is : {}'.format(np.sqrt(metrics.mean_squared_error(y_test_fs,y_pred))))

#print('Variance score for the Multiple LR FS is : %.2f' % regressor1.score(X_test_fs, y_test_fs))

#print("\n")



#Evaluate the regressor on the Feature extraction group

print('RMSE score for the Multiple LR PCA is : {}'.format(np.sqrt(metrics.mean_squared_error(y_test_pca,y_pred_pca))))

print('Variance score for the Multiple LR PCA is : %.2f' % regressor2.score(X_test_pca, y_test_pca))
import pickle

import joblib

filename = 'mlr.pkl'

joblib.dump(regressor2, filename)
X_train.shape
#Find linear correlation of each feature with the target variable

from scipy.stats import pearsonr

df1 = pd.DataFrame(np.concatenate((X_train,y_train.reshape(len(y_train),1)),axis=1))

df1.columns = df1.columns.astype(str)



features = df1.iloc[:,:35].columns.tolist()

target = df1.iloc[:,35].name



correlations = {}

for f in features:

    data_temp = df1[[f,target]]

    x1 = data_temp[f].values

    x2 = data_temp[target].values

    key = f + ' vs ' + target

    correlations[key] = pearsonr(x1,x2)[0]

    

data_correlations = pd.DataFrame(correlations, index=['Value']).T

data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]
#instantiate the object for the Random Forest Regressor with default params from raw data

#regressor_rfraw = RandomForestRegressor(n_jobs=-1)



#instantiate the object for the Random Forest Regressor with default params for Feature Selection Group

#regressor_rf = RandomForestRegressor(n_jobs=-1)



# #instantiate the object for the Random Forest Regressor with tuned hyper parameters for Feature Selection Group

# regressor_rf1 = RandomForestRegressor(n_estimators = 26,

#                                      max_depth = 22,

#                                      min_samples_split = 9,

#                                      n_jobs=-1)



#instantiate the object for the Random Forest Regressor for Feature Extraction Group

regressor_rf2 = RandomForestRegressor(n_jobs=-1)





#Train the object with default params for raw data

#regressor_rfraw.fit(X_train,y_train)



#Train the object with default params for Feature Selection Group

#regressor_rf.fit(X_train_fs,y_train_fs)



# #Train the object with tuned params for Feature Selection Group

# regressor_rf1.fit(X_train_fs,y_train_fs)



# #Train the object with default params for Feature Extraction Group

regressor_rf2.fit(X_train_pca,y_train_pca)



print("\n")
#Predict the output with object of default params for Feature Selection Group

#y_pred_rfraw = regressor_rfraw.predict(X_test)



#Predict the output with object of default params for Feature Selection Group

#y_pred_rf = regressor_rf.predict(X_test_fs)



# #Predict the output with object of hyper tuned params for Feature Selection Group

# y_pred_rf1 = regressor_rf1.predict(X_test_fs)



#Predict the output with object of PCA params for Feature Extraction Group

y_pred_rfpca = regressor_rf2.predict(X_test_pca)



print("\n")
#Evaluate the model with default params for raw data

#print('RMSE score for the RF regressor raw is : {}'.format(np.sqrt(metrics.mean_squared_error(y_test,y_pred_rfraw))))

#print('RMSLE score for the RF regressor raw is : {}'.format(np.sqrt(metrics.mean_squared_log_error(y_test,y_pred_rfraw))))

#print('Variance score for the RF regressor raw is : %.2f' % regressor_rfraw.score(X_test, y_test))



#print("\n")



#Evaluate the model with default params for Feature Selection Group

#print('RMSE score for the RF regressor is : {}'.format(np.sqrt(metrics.mean_squared_error(y_test_fs,y_pred_rf))))

#print('RMSLE score for the RF regressor is : {}'.format(np.sqrt(metrics.mean_squared_log_error(y_test_fs,y_pred_rf))))

#print('Variance score for the RF regressor is : %.2f' % regressor_rf.score(X_test_fs, y_test_fs))



# print("\n")



# #Evaluate the model with tuned params for Feature Selection Group

# print('RMSE score for the RF regressor1 is : {}'.format(np.sqrt(metrics.mean_squared_error(y_test_fs,y_pred_rf1))))

# print('RMSLE score for the RF regressor1 is : {}'.format(np.sqrt(metrics.mean_squared_log_error(y_test_fs,y_pred_rf1))))

# print('Variance score for the RF regressor1 is : %.2f' % regressor_rf1.score(X_test_fs, y_test_fs))



#print("\n")



#Evaluate the model with PCA params  for Feature Extraction Group

print('RMSE score for the RF regressor2 is : {}'.format(np.sqrt(metrics.mean_squared_error(y_test_pca, y_pred_rfpca))))

print('Variance score for the RF regressor2 is : %.2f' % regressor_rf2.score(X_test_pca, y_test_pca))
filename = 'rfr.pkl'

joblib.dump(regressor_rf2, filename)
#instantiate the object for the XGBoost Regressor with default params for raw data

#regressor_xgbraw = XGBRegressor(n_jobs=-1)



#instantiate the object for the XGBoost Regressor with default params for Feature Selection Group

#regressor_xgb = XGBRegressor(n_jobs=-1)



#instantiate the object for the XGBoost Regressor with tuned hyper parameters for Feature Selection Group

regressor_xgb1 = XGBRegressor(n_estimators=300,

                            learning_rate=0.09,

                            gamma=0,

                            subsample=0.75,

                            colsample_bytree=1,

                            max_depth=7,

                            min_child_weight=4,

                            silent=1,

                            n_jobs=-1)



#instantiate the object for the XGBoost Regressor for Feature Extraction Group

#regressor_xgb2 = XGBRegressor(n_jobs=-1)





#Train the object with default params for raw data

#regressor_xgbraw.fit(X_train,y_train)



#Train the object with default params for Feature Selection Group

#regressor_xgb.fit(X_train_fs,y_train_fs)



#Train the object with tuned params for Feature Selection Group

regressor_xgb1.fit(X_train_pca,y_train_pca)



#Train the object with default params for Feature Extraction Group

#regressor_xgb2.fit(X_train_pca,y_train_pca)



print("\n")
#Predict the output with object of default params for raw data

#y_pred_xgbraw = regressor_xgbraw.predict(X_test)



#Predict the output with object of default params for Feature Selection Group

#y_pred_xgb = regressor_xgb.predict(X_test_fs)



#Predict the output with object of hyper tuned params for Feature Selection Group

y_pred_xgb1 = regressor_xgb1.predict(X_test_pca)



#Predict the output with object of PCA params for Feature Extraction Group

#y_pred_xgb_pca = regressor_xgb2.predict(X_test_pca)



print("\n")
#Evaluate the model with default params for raw data

#print('RMSE score for the XGBoost regressor raw is : {}'.format(np.sqrt(metrics.mean_squared_error(y_test,y_pred_xgbraw))))

# print('RMSLE score for the XGBoost regressor is : {}'.format(np.sqrt(metrics.mean_squared_log_error(y_test,y_pred_xgb))))

#print('Variance score for the XGBoost regressor raw is : %.2f' % regressor_xgbraw.score(X_test, y_test))



print("\n")



#Evaluate the model with default params for Feature Selection Group

#print('RMSE score for the XGBoost regressor is : {}'.format(np.sqrt(metrics.mean_squared_error(y_test_fs,y_pred_xgb))))

# print('RMSLE score for the XGBoost regressor is : {}'.format(np.sqrt(metrics.mean_squared_log_error(y_test,y_pred_xgb))))

#print('Variance score for the XGBoost regressor is : %.2f' % regressor_xgb.score(X_test_fs, y_test_fs))



print("\n")



#Evaluate the model with Tuned params for Feature Selection Group

#print('RMSE score for the XGBoost regressor1 is : {}'.format(np.sqrt(metrics.mean_squared_error(y_test_fs,y_pred_xgb1))))

# print('RMSLE score for the XGBoost regressor1 is : {}'.format(np.sqrt(metrics.mean_squared_log_error(y_test_fs,y_pred_xgb1))))

#print('Variance score for the XGBoost regressor1 is : %.2f' % regressor_xgb1.score(X_test_fs,y_test_fs))



print("\n")



#Evaluate the model with PCA params  for Feature Extraction Group

print('RMSE score for the XGBoost regressor2 is : {}'.format(np.sqrt(metrics.mean_squared_error(y_test_pca, y_pred_xgb1))))

print('Variance score for the XGBoost regressor2 is : %.2f' % regressor_xgb1.score(X_test_pca, y_test_pca))
filename = 'xgbr.pkl'

joblib.dump(regressor_xgb1, filename)
#Comparing test results for the XGBoost and RF regressor

print("Total sum of difference between the actual and the predicted values for the RF regressor is : %d"%np.abs(np.sum(np.subtract(y_test,y_pred_rfpca))))

print("Total sum of difference between the actual and the predicted values for the tuned XGB regressor is : %d"%np.abs(np.sum(np.subtract(y_test,y_pred_xgb1))))
#Define a function to plot learning curve.

def learning_curves(estimator, title, features, target, train_sizes, cv, n_jobs=-1):

    plt.figure(figsize = (14,5))

    train_sizes, train_scores, validation_scores = learning_curve(estimator, features, target, train_sizes = train_sizes, cv = cv, scoring = 'neg_mean_squared_error',  n_jobs=n_jobs)

    train_scores_mean = -train_scores.mean(axis = 1)

    validation_scores_mean = -validation_scores.mean(axis = 1)

    

    plt.grid()

    

    plt.plot(train_sizes, train_scores_mean,'o-', color="r", label = 'Training error')

    plt.plot(train_sizes, validation_scores_mean,'o-', color="g", label = 'Validation error')



    plt.ylabel('MSE', fontsize = 14)

    plt.xlabel('Training set size', fontsize = 14)

    

    title = 'Learning curves for a ' + title + ' model'

    plt.title(title, fontsize = 18, loc='left')

    

    plt.legend(loc="best")

    

    return plt



# score curves, each time with 20% data randomly selected as a validation set.

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=4)



# Plot learning curve for the RF Regressor

title = "Random Forest Regressor"



# Call learning curve with all dataset i.e. traininig and test combined because CV will take of data split.

learning_curves(regressor_xgb1, title, X_opt,Y, train_sizes=np.linspace(.1, 1.0, 5), cv=cv, n_jobs=-1)



#Plot learning curve for the XGBoost Regressor

#title = "XGBoost Regressor"



# Call learning curve on less number of estimators than the tuned estimator because it took too much time for the compilation.

#learning_curves(XGBRegressor(n_estimators=111,

                            #learning_rate=0.08,

                            #gamma=0,

                            #subsample=0.75,

                           # colsample_bytree=1,

                            #max_depth=7,

                            #min_child_weight=4,

                            #silent=1), title, X_opt,Y, train_sizes=np.linspace(.1, 1.0, 5), cv=cv, n_jobs=-1)



plt.show()