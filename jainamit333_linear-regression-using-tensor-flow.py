import os
import math
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.data import Dataset
from sklearn import metrics
from matplotlib import cm
from amit_lib import convert_yes_no_to_int, shuffle, split_training_and_test, scaling_of_data, bucketization_by_size_of_bucket
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)
data = pd.read_csv('data/housing_lr/Housing.csv',index_col=0,header=0)
data.head(5)
data = convert_yes_no_to_int(data)
data.describe()
data.plot(x='bedrooms', y='price', style='o') 
print(" price increases till 4 bedroom then descreases")
print(" most data is for 3 and 4 bedroom")
data.plot(x='bathrms', y='price', style='o') 
data.plot(x='stories', y='price', style='o') 
data.plot(x='lotsize', y='price', style='o') 
print ('lot size seems to be good feature')
data.plot(x='driveway', y='price', style='o') 
print(" seems to be feature vector")
print(data.plot(x='recroom', y='price', style='o'))
print(data.plot(x='fullbase', y='price', style='o'))
print(data.plot(x='gashw', y='price', style='o'))
print(data.plot(x='airco', y='price', style='o'))
print(data.plot(x='garagepl', y='price', style='o'))
print(data.plot(x='prefarea', y='price', style='o'))
data = shuffle(data)
feature_set = ['lotsize','bedrooms','stories','driveway','garagepl']
target_set = ['price']
data = bucketization_by_size_of_bucket(data, 'lotsize', 2000)
data = scaling_of_data(data, 'price', 10000, True)
data.describe()
#data = scaling_of_data_t(data, 'price', 10000, True)

print(data.plot(x='lotsize', y='price', style='o'))
print(data.plot(x='bedrooms', y='price', style='o'))
print(data.plot(x='stories', y='price', style='o'))
print(data.plot(x='driveway', y='price', style='o'))
print(data.plot(x='garagepl', y='price', style='o'))

training_data, test_data = split_training_and_test(data_frame=data, training_percentage=90)
print(training_data.shape)
print(test_data.shape)
def tf_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])
# features names : feature_set, target_set
# target name: 
training_feature_df = training_data[feature_set]
training_target_df = training_data[target_set]
test_feature_df = test_data[feature_set]
test_target_df = test_data[target_set]
tf_feature_columns=tf_feature_columns(training_feature_df)
print(training_feature_df.shape)
print(training_target_df.shape)
print(test_feature_df.shape)
print(test_target_df.shape)
print(tf_feature_columns)
#features 
#targets
def input_function(features, targets, batch_size, epochs=None):
    features = {key:np.array(val) for key,val in dict(features).items()}
    ds = Dataset.from_tensor_slices((features,targets))
    ds = ds.batch(batch_size).repeat(epochs)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features,labels
# feature_columns: tf_features_ have to check what kind of data it is
# training_features: dataframe: with all input features
# training_target: dataframe: with target feature
def train_model(learning_rate, steps, batch_size, feature_columns, training_features, training_target, validation_features, validation_target,target_column_name):
    periods = 10
    steps_per_period = steps / periods
    
    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    # tell tf to create linear rregression by telling feature column names and optimizer
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns,optimizer=my_optimizer)
    
    training_input_fn = lambda: input_function(training_features, training_target[target_column_name],batch_size)
    # during prediction number of epoch is 1 and batch size is also 1
    predict_training_input_fn = lambda: input_function(training_features, training_target[target_column_name], 1,epochs=1)
    predict_validation_input_fn = lambda: input_function(validation_features, validation_target[target_column_name],1, epochs=1)
    
    print("Training Mode")
    training_rmse = []
    validation_rmse = []
    
    for period in range (0, periods):
        linear_regressor.train(input_fn=training_input_fn,steps=steps_per_period)
        
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
        training_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(training_predictions, training_target))
        validation_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(validation_predictions, validation_target))
        
        print('period {} training {} validation error {}'.format(period, training_root_mean_squared_error, validation_root_mean_squared_error))
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
        
        
    print("Model training finished.")
    
    # logic to plot graph can be common
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    
    return linear_regressor
    
    
_ = train_model(
    learning_rate=0.01,
    steps=900,
    batch_size=20,
    feature_columns=tf_feature_columns,
    training_features=training_feature_df,
    training_target=training_target_df,
    validation_features=test_feature_df,
    validation_target=test_target_df,
    target_column_name ='price')
_ = train_model(
    learning_rate=0.01,
    steps=2000,
    batch_size=100,
    feature_columns=tf_feature_columns,
    training_features=training_feature_df,
    training_target=training_target_df,
    validation_features=test_feature_df,
    validation_target=test_target_df,
    target_column_name ='price')