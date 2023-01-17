
from __future__ import print_function
from IPython import display
import math
import pandas as pd
import numpy as np
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import metrics
tf.logging.set_verbosity(tf.logging.ERROR)
#df = pd.read_csv('ml-100k/u.item', sep='|',  encoding='latin-1')
df = pd.read_csv ("../input/Real_estate_valuation_data_set.csv",sep=';',  error_bad_lines=False, encoding='latin-1')
df = df.reindex (np.random.permutation(df.index))
df.head()
df.tail()
df.info()
df['Y house price of unit area'] = df['Y house price of unit area'].str.replace(",", ".").astype (float)
#df['X3 distance to the nearest MRT station'] = df['X3 distance to the nearest MRT station'].str.replace(",", ".").astype (float)
df['X2 house age']= df['X2 house age'].str.replace(",", "." ).astype (float)

def preprocess_features (df):
    selected_features = df [['X4 number of convenience stores',
                             'X2 house age']]
#     #df['X1 transaction date'].str.replace(",", ".").astype (float)
#     df['X2 house age']= df['X2 house age'].str.replace(",", "." ).astype (float)
#     df['X3 distance to the nearest MRT station'] = df['X3 distance to the nearest MRT station'].str.replace(",", ".").astype (float)
#     #df['X4 number of convenience stores'].str.replace(",", ".").astype (float)
#     df['X5 latitude'] = df['X5 latitude'].str.replace(",", ".").astype (float)
#     df['X6 longitude'] = df['X6 longitude'].str.replace(",", ".").astype (float)
    
    processed_features = selected_features.copy()
    processed_features.rename (columns = {"X4 number of convenience stores" : "X4_number_of_convenience_stores", 
                                          "X2 house age" : "X2_house_age"}, inplace = True)
    return processed_features

def preprocess_targets (df):
    
    output_targets = df[['Y house price of unit area']]
    output_targets.rename (columns = {"Y house price of unit area" : "Y_house_price_of_unit_area"}, inplace = True)
    return output_targets

      
preprocess_targets (df.head(313))
# split_transaction_date = pd.DataFrame()
# split_transaction_date = df['X1 transaction date'].str.split(',', expand = True)
# #split_transaction_date.rename ({"0": "year", "1": "month"})
# split_transaction_date.reset_index (inplace = True)
# split_transaction_date.columns = ['null','year', 'month']
# #split_transaction_date.head()
# df_update = pd.concat ([df.drop ('X1 transaction date', axis = 1) , split_transaction_date.drop ('null', axis = 1)], axis = 1, sort = False )
# df_update.tail()
#choose the first 313 examples for training
training_examples = preprocess_features (df.head(313))
training_targets = preprocess_targets (df.head(313))

#choose the last 100 examples for testing

testing_examples = preprocess_features (df.tail(100))
testing_targets = preprocess_targets (df.tail(100))

#checking whether we have done the right thing
print("Training examples summary:")
display.display(training_examples.describe())
print("Training targets summary:")
display.display(training_targets.describe())
# construct feature columns

def construct_feature_columns (input_features):
#     print (set([tf.feature_column.numeric_column(my_feature)
#               for my_feature in input_features])
    #The name of inout numerica features to use
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])
#training_examples
construct_feature_columns (training_examples)
#training_targets['Y house price of unit area']
def my_input_fn (features, targets, batch_size = 1, shuffle = True, num_epochs = None ):
    # input features
    features = {key : np.array (value) for key,value in dict (features).items()}
    
    # contsruct a dataset
    
    ds = Dataset.from_tensor_slices ((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # shuffle
    if shuffle:
        ds = ds.shuffle(300)
        
    #return the next batch of data
    
    features, labels = ds.make_one_shot_iterator().get_next()
    return features,labels


def get_quantile_based_boundaries (feature_values, num_buckets):
    boundaries = np.arrange (1.0, num_buckets)/ num_buckets
    quantiles = feature_values.quantile (boundaries)
    return [quantiles[q] for q in quantiles.keys()]
#Divide households into 7 buckets


def model_train(learning_rate, steps, batch_size, training_examples, training_targets):
    periods = 10
    steps_per_period = steps/periods
    print (steps_per_period)
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm (my_optimizer,5.0)
    
    linear_regressor = tf.estimator.LinearRegressor(
                        feature_columns = construct_feature_columns (training_examples), 
                        optimizer=my_optimizer 
    )
    
    # create input functions
    training_input_fn = lambda: my_input_fn (training_examples, training_targets ['Y_house_price_of_unit_area'] , batch_size = batch_size)
    predict_training_input = lambda: my_input_fn (training_examples, training_targets ['Y_house_price_of_unit_area'] ,
                                                 num_epochs = 1, shuffle = False)
    #Train the model, but do so inside a loop so that we can periodically access
    
    print ("Training model")
    print ("RMSE on training data")
    
    training_rmse = []
    
    for period in range (periods):
        print (period)
        #Train the model starting from prior state
        linear_regressor.train (
            input_fn = training_input_fn,
            steps = steps_per_period )
        
        training_predictions = linear_regressor.predict (input_fn = predict_training_input)
        training_predictions = np.array ([item['predictions'][0] for item in training_predictions])
        
        # compute traiing and validation loss
        
        training_root_mean_squared_error = math.sqrt (metrics.mean_squared_error (training_predictions, training_targets))
        #print the current loss
        print ("period %0.2f and error %0.3f" % (period,training_root_mean_squared_error))
        
        training_rmse.append (training_root_mean_squared_error)
    print ("Model training finished")
         # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
        
    plt.legend()
    plt.show()
    return linear_regressor
        
    
model_train (learning_rate = 0.01,
            steps = 100,
            batch_size = 1,
            training_examples = training_examples,
            training_targets = training_targets)
