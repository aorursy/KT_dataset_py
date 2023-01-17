#making the imports / ignore warnings

import pandas as pd

import numpy as np

import gc

import matplotlib.pyplot as plt

%matplotlib inline

from IPython.display import Image

import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
# although not related I will show some images to get an idea about where the most earthquakes occur

global_earth_quakes = Image('../input/earth-quake-images/global_earth_quakes.jpg', width = 1000)

global_earth_quakes
#image showing nuclear plant locations and earth quake hot zones

nuclear_plants_locations = Image('../input/earth-quake-images/earth_quakes_nuclear_p_locations.jpg')

nuclear_plants_locations
#reading the training file (warning: huge size) specify data types to save memory

#I will be using garbage collection frequently to clear the memory



data_type = {'acoustic_data': np.int16, 'time_to_failure': np.float32}

train = pd.read_csv('../input/LANL-Earthquake-Prediction/train.csv', dtype=data_type)

train.head()
#garbage collection

gc.collect()
# plot to see the relation between given variable and target variable



train_ad_sample_df = train['acoustic_data'].values[::1000]

train_ttf_sample_df = train['time_to_failure'].values[::1000]



#function for plotting based on both features

def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1% sampled data"):

    fig, ax1 = plt.subplots(figsize=(12, 8))

    plt.title(title)

    plt.plot(train_ad_sample_df, color='r')

    ax1.set_ylabel('acoustic data', color='r')

    plt.legend(['acoustic data'], loc=(0.01, 0.95))

    ax2 = ax1.twinx()

    plt.plot(train_ttf_sample_df, color='g')

    ax2.set_ylabel('time to failure', color='g')

    plt.legend(['time to failure'], loc=(0.01, 0.9))

    plt.grid(True)



plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)

#delete the old frame

del train_ad_sample_df

del train_ttf_sample_df
#plot to show zoomed in view



train_ad_sample_df = train['acoustic_data'].values[:6291455]

train_ttf_sample_df = train['time_to_failure'].values[:6291455]

plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1% of data")

del train_ad_sample_df

del train_ttf_sample_df
#garbage collection

gc.collect()
#lets create a function to generate some statistical features based on the training data

# this is necessay as only one variable [acoustic data] is given to us in training set



def generate_features(X):

    strain = []

    strain.append(X.mean())

    strain.append(X.std())

    strain.append(X.min())

    strain.append(X.max())

    strain.append(X.kurtosis())

    strain.append(X.skew())

    strain.append(np.quantile(X,0.01))

    strain.append(np.quantile(X,0.05))

    strain.append(np.quantile(X,0.95))

    strain.append(np.quantile(X,0.99))

    strain.append(np.abs(X).max())

    strain.append(np.abs(X).mean())

    strain.append(np.abs(X).std())

    return pd.Series(strain)
# check the head

train.head()
# lets apply feature generation function

# also we will read the training file in chunks. chunk size specifies the number of rows which pandas will 

# read in one chunk



c_s = 10 ** 6

train = pd.read_csv('../input/LANL-Earthquake-Prediction/train.csv', iterator=True, chunksize= c_s, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})



X_train = pd.DataFrame()

y_train = pd.Series()

for df in train:

    ch = generate_features(df['acoustic_data'])

    X_train = X_train.append(ch, ignore_index=True)

    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))
#describe the data



X_train.describe()
#garbage collection

gc.collect()
# just a base line for cat boost

# get the best score without hyper parameter tuning



from catboost import CatBoostRegressor, Pool

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV





train_pool = Pool(X_train, y_train)

m = CatBoostRegressor(iterations=10000, loss_function='MAE', boosting_type='Ordered')

m.fit(X_train, y_train, silent=True)

m.best_score_
# now lets try SVM with rbf kernel + grid search for hyper paramter tuning



from sklearn.svm import NuSVR, SVR

from sklearn.model_selection import KFold



scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)



folds = KFold(n_splits= 5, shuffle= True, random_state= 101)



parameters = [{'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],

               'C': [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2]}]

               



reg1 = GridSearchCV(SVR(kernel='rbf', tol=0.01), parameters, cv= folds, scoring='neg_mean_absolute_error')

reg1.fit(X_train_scaled, y_train.values.flatten())

y_pred1 = reg1.predict(X_train_scaled)



print("Best CV score: {:.4f}".format(reg1.best_score_))

print(reg1.best_params_)
#garbage collection

gc.collect()
#making the imports

#TQDM is a progress bar library with good support for nested loops and Jupyter/IPython notebooks.



from sklearn.preprocessing import StandardScaler

from keras.models import Sequential

from keras.layers import Dense

from tqdm import tqdm
#reading the training file with data types int16 and float32



data_type = {'acoustic_data': np.int16, 'time_to_failure': np.float32}

train_data = pd.read_csv('../input/LANL-Earthquake-Prediction/train.csv', dtype=data_type)

train_data.head()
#garbage collection

gc.collect()
# making the training file ready to be fed into a NN



rows = 150000

segments = int(np.floor(train_data.shape[0] / rows))



X_train = pd.DataFrame(index = range(segments),dtype = np.float32,columns = ['mean','std','99quat','50quat','25quat','1quat'])

y_train = pd.DataFrame(index = range(segments),dtype = np.float32,columns = ['time_to_failure'])
# generating the features like mean/std/quantiles



for segment in tqdm(range(segments)):

    x = train_data.iloc[segment*rows:segment*rows+rows]

    y = x['time_to_failure'].values[-1]

    x = x['acoustic_data'].values

    X_train.loc[segment,'mean'] = np.mean(x)

    X_train.loc[segment,'std']  = np.std(x)

    X_train.loc[segment,'99quat'] = np.quantile(x,0.99)

    X_train.loc[segment,'50quat'] = np.quantile(x,0.5)

    X_train.loc[segment,'25quat'] = np.quantile(x,0.25)

    X_train.loc[segment,'1quat'] =  np.quantile(x,0.01)

    y_train.loc[segment,'time_to_failure'] = y
#using standard scaler to scale the data



scaler = StandardScaler()

X_scaler = scaler.fit_transform(X_train)
#garbage collection

gc.collect()
#compiling the sequential model. Simple model with input shape 6 and activation function rectified linear

# as it is a regression task so use Mean Absolute Error as measuring matrix

# will use default optimizer adam



model = Sequential()

model.add(Dense(32,input_shape = (6,),activation = 'relu'))

model.add(Dense(32,activation = 'relu'))

model.add(Dense(32,activation = 'relu'))

model.add(Dense(1))

model.compile(loss = 'mae',optimizer = 'adam')
#train the model (30 epochs)

#feed in the scaled training data



model.fit(X_scaler,y_train.values.flatten(),epochs = 30)
#reading the submission file from input directory



sub_data = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv',index_col = 'seg_id')
#building the test data frame using same columns as X_train



X_test = pd.DataFrame(columns = X_train.columns,dtype = np.float32,index = sub_data.index)
#feature generation for test data



for seq in tqdm(X_test.index):

    test_data = pd.read_csv('../input/LANL-Earthquake-Prediction/test/'+seq+'.csv')

    x = test_data['acoustic_data'].values

    X_test.loc[seq,'mean'] = np.mean(x)

    X_test.loc[seq,'std']  = np.std(x)

    X_test.loc[seq,'99quat'] = np.quantile(x,0.99)

    X_test.loc[seq,'50quat'] = np.quantile(x,0.5)

    X_test.loc[seq,'25quat'] = np.quantile(x,0.25)

    X_test.loc[seq,'1quat'] =  np.quantile(x,0.01)
#garbage collect

gc.collect()
#scale the test data using pre-defined scaler



X_test_scaler = scaler.transform(X_test)
#making the predictions

pred = model.predict(X_test_scaler)
sub_data.head()
#import xgboost (we will use regressor)



import xgboost as xgb
#use the same already scaled X and y from NN part



xgb_model = xgb.XGBRegressor()



xgb_model.fit(X_scaler,y_train.values)
#predictions without hyper paramter tuning



pred = xgb_model.predict(X_test_scaler)
# hyperparameter tuning with XGBoost (will take some time to run)



# creating a KFold object with 3 splits



folds = KFold(n_splits= 3, shuffle= True, random_state= 101)



# specify range of hyperparameters

param_grid = {'learning_rate': [0.01, 0.1, 0.2, 0.3], 

             'subsample': [0.3, 0.6, 0.9, 1],

              'n_estimators' : [5, 10, 15, 20],

              'max_depth' :[2,4,6,8]          

             }          





# specify model

xgb_model = xgb.XGBRegressor()



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = xgb_model, 

                        param_grid = param_grid, 

                        scoring='neg_mean_absolute_error', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True, 

                        n_jobs= -1)      
#train the model

model_cv.fit(X_scaler,y_train.values)
# printing the optimal accuracy score and hyperparameters

print('We can get neg_mean_absolute_error:',model_cv.best_score_,'using',model_cv.best_params_)
# define model with best paramters and train plus make predictions



xgb_model = xgb.XGBRegressor(learning_rate= 0.2, max_depth= 4, n_estimators= 10, subsample= 0.9)



xgb_model.fit(X_scaler,y_train.values)



pred = xgb_model.predict(X_test_scaler)



# read the submission file and populate it with predictions



sample_submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')

sample_submission['time_to_failure'] = pred



print(sample_submission.shape)



print('\n')



sample_submission.head()
#write to csv file



sample_submission.to_csv('Final_EQ_sub.csv', index=False)