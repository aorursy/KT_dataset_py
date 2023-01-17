import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import seaborn as sns

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_df = pd.read_csv("/kaggle/input/calhousetrain/california_housing_train.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index)) 
test_df = pd.read_csv("/kaggle/input/calhousetest/california_housing_test.csv")

train_df_norm = (train_df - train_df.mean())/train_df.std()
test_df_norm = (test_df - test_df.mean())/test_df.std()

train_df_norm.head()
feature_columns = []
resolution_in_Zs = 0.3

#bucketize
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(np.arange(int(min(train_df_norm['latitude'])), 
                                     int(max(train_df_norm['latitude'])), 
                                     resolution_in_Zs))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column, latitude_boundaries)

#bucketize
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arange(int(min(train_df_norm['longitude'])), 
                                      int(max(train_df_norm['longitude'])), 
                                      resolution_in_Zs))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column, 
                                                longitude_boundaries)
#crossed
latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)

#append 
feature_columns.append(crossed_feature) 

median_income = tf.feature_column.numeric_column("median_income")
feature_columns.append(median_income)

population = tf.feature_column.numeric_column("population")
feature_columns.append(population)

my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
print("preprocessed")
def plot_the_loss_curve(epochs, mse):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.plot(epochs, mse, label="Loss")
    plt.legend()
    plt.ylim([mse.min()*0.95, mse.max() * 1.03])
    plt.show()  
def create_model(learning_rate,feature_layer):
    model = tf.keras.models.Sequential()
    
    model.add(feature_layer)
    model.add(layers.Dense(units=20,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.04),name='h1'))
    model.add(layers.Dense(units=12,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.04),name='h1'))
    model.add(layers.Dense(units=1,name='o'))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                 loss="mean_squared_error",
                 metrics=[tf.keras.metrics.MeanSquaredError()])
    return model
def train_model(model,dataset,epochs,batch_size,label_name):
    features = {name:np.array(value) for name,value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features,y=label,batch_size=batch_size,epochs=epochs,
                        shuffle=True)
    
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["mean_squared_error"]
    
    return epochs,rmse
learning_rate = 0.01
epochs = 15
batch_size = 1000
label_name = "median_house_value"

my_model = create_model(learning_rate, my_feature_layer)

epochs, mse = train_model(my_model, train_df_norm, epochs, batch_size, label_name)
plot_the_loss_curve(epochs, mse)

test_features = {name:np.array(value) for name, value in test_df_norm.items()}
test_label = np.array(test_features.pop(label_name)) 
print("\n Evaluate the linear regression model against the test set:")
my_model.evaluate(x = test_features, y = test_label, batch_size=batch_size)


