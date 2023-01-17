print("hello moto")
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import math 

from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

print(tf.__version__)
def data_to_np(input_file):
    
    #we need to define the nrows because the output files generated with Data_Refine include cuts that modify the shape
    #of the output files (maybe we can fix this later)
    
    df = pd.read_csv(input_file, sep=',', nrows = 900000)
    
    header_names = ['pickup_longitude','pickup_latitude','dropoff_longitude',
                    'dropoff_latitude','passenger_count','distance']
    
    #getting only the columns that we are going to use for the Keras NN
    
    df_train = df[header_names]
    np_df_train = df_train.values
    
    df_label = df['fare_amount']
    np_df_label = df_label.values
    
    return np_df_train, np_df_label    
def global_mean_per_column(mynp_train_list):
    
    sum_mean = 0
    
    for con in range(len(mynp_train_list)):
        sum_mean = sum_mean + np.mean(mynp_train_list[con], axis=0) 
    
    #mean_0 = np.mean(mynp_train_0, axis=0) 
    #mean_1 = np.mean(mynp_train_1, axis=0)
    
    mean = sum_mean/len(mynp_train_list)
        
    return mean
def global_std_per_column(mynp_train_list, global_mean):
    
    sum_mean_x2 = 0
    
    for con in range(len(mynp_train_list)):
        sum_mean_x2 += np.mean((mynp_train_list[con] - global_mean)**2, axis=0)
    
    #mean_x2_0 = np.mean((mynp_train_0 - global_mean)**2, axis=0)
    #mean_x2_1 = np.mean((mynp_train_1 - global_mean)**2, axis=0)
    
    std = np.sqrt(sum_mean_x2/len(mynp_train_list))
    
    return std

def norm_mynp_train(mynp_train, mean, std):
    
    mynp_train_norm = (mynp_train - mean)/std
       
    return mynp_train_norm    
mynp_train_0, mynp_label_0 = data_to_np('../input/my-taxi-fare-data/train_r0.csv')
mynp_train_1, mynp_label_1 = data_to_np('../input/my-taxi-fare-data/train_r1.csv')
mynp_train_2, mynp_label_2 = data_to_np('../input/my-taxi-fare-data/train_r2.csv')
order = np.argsort(np.random.random(mynp_label_0.shape))

mynp_train_0 = mynp_train_0[order]
mynp_label_0 = mynp_label_0[order]

order = np.argsort(np.random.random(mynp_label_1.shape))

mynp_train_1 = mynp_train_1[order]
mynp_label_1 = mynp_label_1[order]

order = np.argsort(np.random.random(mynp_label_2.shape))

mynp_train_2 = mynp_train_2[order]
mynp_label_2 = mynp_label_2[order]
mynp_train_list = []
mynp_train_list.append(mynp_train_0)
mynp_train_list.append(mynp_train_1)
mynp_train_list.append(mynp_train_2)

mynp_label_list = []
mynp_label_list.append(mynp_label_0)
mynp_label_list.append(mynp_label_1)
mynp_label_list.append(mynp_label_2)

global_mean = global_mean_per_column(mynp_train_list)
global_mean
global_std = global_std_per_column(mynp_train_list,global_mean)
global_std

#notice the small value of the std deviation of lat and lon variables
#def build_model(shape_1_of_np_array):
def build_model(shape_of_np_array):
    model = keras.Sequential([
            keras.layers.Dense(64, activation=tf.nn.relu, 
                               input_shape=(shape_of_np_array,)),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1)])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    #Definition of the function to minimize (loss function)
    #model.compile(loss='mse',optimizer=optimizer,metrics=["accuracy"])
    model.compile(loss='mse',optimizer=optimizer,metrics=["mae"])
    return model
# Display training progress by printing a single dot for each completed epoch.
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 5 == 0: print('epoch',epoch)
    print('.'),


df_test = pd.read_csv('../input/my-taxi-fare-data/test_r0.csv', sep=',')
df_test = df_test[['key','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','distance']]
df_test_fn = df_test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','distance']]

mynp_test = df_test_fn.values
mean_test = np.mean(mynp_test,axis=0)
std_test = mynp_test.std(axis=0)
mynp_test = (mynp_test - mean_test)/std_test

mynp_test
#normaalization
mynp_train_norm_0 = norm_mynp_train(mynp_train_list[0], global_mean, global_std)
mynp_train_norm_1 = norm_mynp_train(mynp_train_list[1], global_mean, global_std)
mynp_train_norm_2 = norm_mynp_train(mynp_train_list[2], global_mean, global_std)

#concatanation
mynp_train_concat = np.concatenate((mynp_train_norm_0,mynp_train_norm_1,mynp_train_norm_2),axis=0)
mynp_label_concat = np.concatenate((mynp_label_0,mynp_label_1,mynp_label_2),axis=0)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
mynp_train_norm_0.shape
#reading a sub-array from the big one for the grid study

grid_train = mynp_train_concat[:100]
grid_label = mynp_label_concat[:100]

print(grid_train.shape, grid_label.shape)
model_for_grid = KerasClassifier(build_fn=build_model, shape_of_np_array = mynp_train_0.shape[1], validation_split=0.2, verbose = 0) 
epochs = [3,5]
batches = [32,64]
param_grid = dict(epochs=epochs,batch_size=batches) 
grid = GridSearchCV(estimator=model_for_grid, param_grid=param_grid, n_jobs=1, scoring="neg_mean_absolute_error") 
grid_result = grid.fit(grid_train, grid_label,callbacks=[early_stop, PrintDot()])
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
model_after_gridSearch = build_model(mynp_train_0.shape[1]) 
EPOCHS = grid_result.best_params_['epochs']
BATCH_SIZE = grid_result.best_params_['batch_size']

print(EPOCHS,BATCH_SIZE)
model_after_gridSearch.fit(mynp_train_concat, mynp_label_concat, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])
#model_after_gridSearch.fit(mynp_train_concat, mynp_label_concat, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])
#test_predictions = test_predictions_total/len(mynp_train_super_array)
#test_predictions = test_predictions_total
test_predictions = model_after_gridSearch.predict(mynp_test).flatten()
test_predictions
test_key_array = df_test['key'].values
test_key_array
df_output = pd.DataFrame({'key': test_key_array,'fare_amount': test_predictions})
df_output
df_output.to_csv('submission_file.csv', index = False)
