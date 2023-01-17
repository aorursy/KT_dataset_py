import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
data = pd.read_csv('../input/Concrete_Data_Yeh.csv')
data = shuffle(data)
data.describe()
print("Null values check ", data.columns[data.isna().any()].tolist())
training_features = ['cement','slag', 'flyash', 'water', 'superplasticizer', 'coarseaggregate',  'fineaggregate', 'age' ]
label_feature = ['csMPa']
selected_feature_data = data
m = data.shape[0]
n = len(training_features)
percentage_of_training = 85
number_example_in_training = int((percentage_of_training * m)/100)
number_example_in_test = int(m - number_example_in_training)

print('number_example_in_training', number_example_in_training)
print('number_example_in_test', number_example_in_test)

training_data_features = selected_feature_data.head(number_example_in_training)[training_features]
training_data_labels = selected_feature_data.head(number_example_in_training)[label_feature]

test_data_features = selected_feature_data.head(number_example_in_test)[training_features]
test_data_labels = selected_feature_data.head(number_example_in_test)[label_feature]


print('shape of traing data features', training_data_features.shape)
print('shape of traing data labels', training_data_labels.shape)
print('shape of test data features', test_data_features.shape)
print('shape of test data features', test_data_labels.shape)

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 1000 == 0: print('epoch ', epoch,)

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 20])
def build_model(learning_rate):

  model = keras.models.Sequential([
          keras.layers.Dense(17, activation=tf.nn.relu,kernel_regularizer= keras.regularizers.l2(0.01),
                       input_shape=(training_data_features.shape[1],)),
    keras.layers.Dense(15, activation=tf.nn.relu,kernel_regularizer= keras.regularizers.l2(0.01),
                       input_shape=(training_data_features.shape[1],)),
    keras.layers.Dense(7, activation=tf.nn.relu, kernel_regularizer= keras.regularizers.l2(0.01)),  
    keras.layers.Dense(3, activation=tf.nn.relu,kernel_regularizer= keras.regularizers.l2(0.01)),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model
learning_rate = 0.0003
epochs = 10000


model = build_model(learning_rate=learning_rate)
history = model.fit(training_data_features, training_data_labels, epochs=epochs,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])        
print('Learning Rate ', learning_rate)
print(model.evaluate(test_data_features, test_data_labels, verbose=0))
print(plot_history(history))
    
predicted_value = model.predict(test_data_features)
MSE = mean_squared_error(np.asmatrix(test_data_labels), predicted_value)
print("Mean Square Error ", MSE)
for i in range(0,5):
    print(predicted_value[i], "--", np.asmatrix(test_data_labels)[i])
