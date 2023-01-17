# Import some libraries
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # graphing
import os
from datetime import datetime, timedelta # Used to subtract days from a date
import seaborn as sb

print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

# Import environment
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
# Import training dataset
(market_train_df, _) = env.get_training_data()
# Heat map with market_train_df
C_mat = market_train_df.corr()
fig = plt.figure(figsize=(15,15))
sb.heatmap(C_mat,vmax=0.5,square=True,annot=True)
plt.show()
def remove_outlier(df,column_list,lower_percentile,upper_percentile):
    for i in range(len(column_list)):
        #upper_bound = np.percentile(df[column_list[i]],upper_percentile)
        #lower_bound = np.percentile(df[column_list[i]],lower_percentile)
        df = (df[(df[column_list[i]]<np.percentile(df[column_list[i]],upper_percentile)) & (df[column_list[i]]>np.percentile(df[column_list[i]],lower_percentile))])
    return df
#outlier_removal_list = ['returnsClosePrevRaw1','returnsOpenPrevRaw1','returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevRaw10','returnsOpenPrevRaw10','returnsClosePrevMktres10','returnsOpenPrevMktres10','returnsOpenNextMktres10']
outlier_removal_list = [ 'returnsClosePrevRaw1',
                         'returnsOpenPrevRaw1',
                         'returnsClosePrevRaw10',
                         'returnsOpenPrevRaw10',
                         'returnsOpenNextMktres10']

market_data_no_outlier = remove_outlier(market_train_df,outlier_removal_list,2,98)
print("Number of data decreased from ",len(market_train_df['returnsOpenNextMktres10'])," to ",len(market_data_no_outlier['returnsOpenNextMktres10']))

C_mat = market_data_no_outlier.corr()
fig = plt.figure(figsize=(15,15))
sb.heatmap(C_mat,vmax=0.5,square=True,annot=True)
plt.show()
# proces data
def process_merged_data(df):
    # Drop rows with NaN values
    df = df.dropna()
    # Let's choose our features#
    features = ['time','returnsClosePrevRaw1','returnsOpenPrevRaw1','returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevRaw10','returnsOpenPrevRaw10','returnsClosePrevMktres10','returnsOpenPrevMktres10']
    x = df[features]
    y = df[['time','returnsOpenNextMktres10']]
    return x,y

market_data_no_outlier,market_data_no_outlier_target = process_merged_data(market_data_no_outlier)
from sklearn.preprocessing import StandardScaler
def scale_data(df,features):
    scaler = StandardScaler()
    df[features]=scaler.fit_transform(df[features])
    return df
features = ['returnsClosePrevRaw1',
         'returnsOpenPrevRaw1',
         'returnsClosePrevMktres1',
         'returnsOpenPrevMktres1',
         'returnsClosePrevRaw10',
         'returnsOpenPrevRaw10',
         'returnsClosePrevMktres10',
         'returnsOpenPrevMktres10']    
market_data_no_outlier_scaled = scale_data(market_data_no_outlier,features)
# Heat map with merged_data
features = ['returnsClosePrevRaw1',
         'returnsOpenPrevRaw1',
         'returnsClosePrevMktres1',
         'returnsOpenPrevMktres1',
         'returnsClosePrevRaw10',
         'returnsOpenPrevRaw10',
         'returnsClosePrevMktres10',
         'returnsOpenPrevMktres10']
temp_show = market_data_no_outlier_scaled[features]
temp_show['target']=market_data_no_outlier_target['returnsOpenNextMktres10']
C_mat = temp_show.corr()
fig = plt.figure(figsize=(15,15))
sb.heatmap(C_mat,vmax=0.5,square=True,annot=True)
plt.show()
del temp_show
# Splits data for training. Takes out 30 days worth of data between training and validation set to prevent data leakage
def split_train_test(x,y,test_size):    
    # Splits data as specified test_size and creates a gap of 30 days between train and test. This helps data leakage so that the model doesn't know the future when training
    X_train = x[x['time']<(x['time'][int(len(x)*(1-test_size))]-timedelta(days=30))]
    y_train = y[y['time']<(y['time'][int(len(x)*(1-test_size))]-timedelta(days=30))]
    X_test = x[x['time']>x['time'][int(len(x)*(1-test_size))]]
    y_test = y[y['time']>y['time'][int(len(y)*(1-test_size))]]   
    # Final Features to be used
    #features = ['returnsClosePrevRaw1','returnsOpenPrevRaw1','returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevRaw10','returnsOpenPrevRaw10','returnsClosePrevMktres10','returnsOpenPrevMktres10'] 
    features = ['returnsClosePrevRaw10','returnsOpenPrevRaw10','returnsClosePrevMktres10']
    
    X_train1 = X_train[features].copy()
    y_train1 = y_train['returnsOpenNextMktres10'].copy()
    train_time = y_train['time']
    
    X_test1 = X_test[features].copy()
    y_test1 = y_test['returnsOpenNextMktres10'].copy()
    test_time = y_test['time']
    return X_train1,X_test1,y_train1,y_test1,train_time,test_time

X_train,X_test,y_train,y_test,train_time,test_time = split_train_test(market_data_no_outlier_scaled,market_data_no_outlier_target,0.1)
print("Test data percentage : {} %".format(len(X_test)/(len(X_train)+len(X_test))*100))

# Model
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,Input
from keras.optimizers import Adam

# Initialize Model
model = Sequential()
# Input layer & hidden layer
model.add(Dense(32, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(32,activation='relu'))
# Output layer
model.add(Dense(1))
# Compile the architecture and view summary
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.summary()

from keras.callbacks import ModelCheckpoint,EarlyStopping

# checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
# checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_acc', verbose = 1, save_best_only = True, mode ='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto',restore_best_weights=True)
callbacks_list = [early_stopping]
#callbacks_list = [checkpoint,early_stopping]
model.fit(x=X_train.values,y=y_train.values, epochs=20,shuffle=True,validation_data=(X_test.values, y_test.values),callbacks=callbacks_list)# validation_split=0.2)#) #, callbacks=callbacks_list)
data = {'y_real':y_test[:20],'y_pred':(model.predict(X_test.values[:20])).reshape(1,-1)[0]}
pd.DataFrame(data)
def make_my_prediction(x):
    my_pred = (model.predict(x)).reshape(1,-1)[0]
    my_pred[my_pred>0]=1
    my_pred[my_pred<0]=-1
    return my_pred
# sigma_score function is considered as a custom evaluation metric for xgboost
# example of how custom evaluation function is incorporated into xgboost's training can be found here : https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py
def sigma_score(preds,dval,df):
    
    # get y_target values
    labels = dval
    # call time parameter to be used for grouping, so that we can add x_t values for each day
    df_time = df
    
    #calculate x_t and score as specified by the competition
    x_t = pd.Series(preds*labels)
    x_t_sum = x_t.groupby(df_time).sum()    
    score = (x_t_sum.mean())/(x_t_sum.std())
    return 'sigma_score', round(score,5)

my_pred_test = make_my_prediction(X_test.values)
print("test : ",sigma_score(my_pred_test,y_test,test_time))

my_pred_train = make_my_prediction(X_train.values)
print("train : ",sigma_score(my_pred_train,y_train,train_time))
for (market_obs_df, _, predictions_template_df) in env.get_prediction_days():  
    features = ['returnsClosePrevRaw10','returnsOpenPrevRaw10','returnsClosePrevMktres10']
    market_obs_df_scaled = scale_data(market_obs_df,features)    
    x_submission = market_obs_df_scaled[features].copy()
    # fill in NaN values with mean of rest of the values
    for i in range(len(features)):
         x_submission[features[i]]= x_submission[features[i]].fillna(x_submission[features[i]].mean())
    predictions_template_df['confidenceValue'] = make_my_prediction(x_submission)
    env.predict(predictions_template_df)
    del x_submission
print('Done!')
# Write submission file    
env.write_submission_file()