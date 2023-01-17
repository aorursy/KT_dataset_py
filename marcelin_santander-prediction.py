import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import math
from keras.utils import to_categorical
import keras.backend as K

# ========================  import data  =======================================================================================

# train data
train_dataframe = pd.read_csv("../input/train.csv", sep=",")

# test data
test_dataframe = pd.read_csv("../input/test.csv", sep=",")

#train_dataframe = train_dataframe.reindex(np.random.permutation(train_dataframe.index))

test_ID = test_dataframe["ID"]
labels = train_dataframe["target"]
#print(test_ID)
# ============================================ DEFINE FUNCTIONS ==================================================================================
def rmse_k(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
def rmsle_k(y, y0): 
    return K.sqrt(K.mean(K.square(tf.log1p(y) - tf.log1p(y0)))) 
def log_normalize(series):
  import math
  return series.apply(lambda x:math.log(x+1.0))

def normalize(dat):
  train_numerical = dat.select_dtypes(exclude=['object'])
  columns = train_numerical.columns
  for c in columns:
    dat[c] = log_normalize(dat[c])
  return dat

def preprocessingData(train):
  train_numerical = train.select_dtypes(exclude=['object'])
  train_numerical.fillna(0,inplace = True)
  train_categoric = train.select_dtypes(include=['object'])
  train_categoric.fillna('NONE',inplace = True)
  train = train_numerical.merge(train_categoric, left_index = True, right_index = True)
  train2 = train.drop('ID',axis = 1)
  return train2

def TrainAndTest(train_a,test_a):
  train_drop = train_a.drop('target', axis=1)
  train_objs_num = len(train_drop)
  dataset = pd.concat(objs=[train_drop, test_a], axis=0)
  dataset_preprocessed = pd.get_dummies(dataset)
  train_preprocessed = dataset_preprocessed[:train_objs_num]
  test_preprocessed = dataset_preprocessed[train_objs_num:]
  return train_preprocessed,test_preprocessed

def preprocess_features(train_dataframe):
  selected_features = train_dataframe
  return selected_features.drop('target', axis=1)

def preprocess_targets(train_dataframe):
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  #output_targets["target"] = (train_dataframe["target"] / 100000.0)
  output_targets["target"] = (train_dataframe["target"])
  return output_targets

# ==============================================================================================================================

def train_model(epochs,batch_size,my_optimizer, training_examples,training_targets,validation_examples,validation_targets,test_examples):
    model = keras.Sequential([
    keras.layers.Dense(15000, activation=tf.nn.relu, 
                       input_dim=training_examples.shape[1]),
    #keras.layers.Dense(training_examples.shape[1], activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10000, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.relu)])
    
    model.compile(loss='msle',optimizer=my_optimizer, metrics=[rmsle_k,rmse_k])
    # ======  PRINT SUMARIZE OF ARCHITECTURE =========================================
    model.summary()
    # Store training stats
    history = model.fit(training_examples, training_targets, epochs=epochs,batch_size=batch_size,validation_data = (validation_examples, validation_targets))
    def plot_history(history):
      plt.figure()
      plt.xlabel('Epoch')
      plt.ylabel('Root Mean Squared Error')
      plt.plot(history.history['rmse_k'], label='Training RMSE', color="blue")
      plt.plot(history.history['val_rmse_k'],label = 'Validation RMSE', color="red")
      plt.legend()
      plt.show()

      plt.figure()
      plt.xlabel('Epoch')
      plt.ylabel('Root Mean Squared Logarithm Error')
      plt.plot(history.history['rmsle_k'], label='Training RMSLE', color="blue")
      plt.plot(history.history['val_rmsle_k'],label = 'Validation RMSLE', color="red")
      plt.legend()
      plt.show()
    """ ========================  evaluation of model ========================"""
    plot_history(history)
    [loss,rmsle,rmse] = model.evaluate(validation_examples, validation_targets)
    #print("Final RMSE of validation: {}".format(math.sqrt(mse)))
    print("Final RMSE of validation: {}".format(rmse))
    print("Final RMSLE of validation: {}".format(rmsle))
    #print("Final MSE of validation: {}".format(mse))
    """ ========================  Prediction of model ========================"""
    test_predictions = model.predict(test_examples).flatten()
    return test_predictions

# ===================================================================================================
train_dataframe = train_dataframe.reindex(np.random.permutation(train_dataframe.index))
labels2 = train_dataframe.loc[:,["target"]]

#convert NaNs to an empty string using
train_1 = preprocessingData(train_dataframe)
test_1  = preprocessingData(test_dataframe)
train,test = TrainAndTest(train_1,test_1)


training_examples, validation_examples, training_targets, validation_targets  = train_test_split(train, labels2,test_size=0.2)
# Examples for training.
#training_examples = preprocess_features(training_examples)
training_targets = preprocess_targets(training_targets)
# Examples for validation.
#validation_examples = preprocess_features(validation_examples)
validation_targets = preprocess_targets(validation_targets)


# -------------------------------------------- TEST ----------------------------------------------------------------
#training_examples = normalize(training_examples)
#validation_examples = normalize(validation_examples)
test_examples = test
#test_examples = normalize(test_examples)

# ------------------------------------------------------------------------------------------------------------------
y_pred = train_model(
    epochs=20,
	batch_size=300,
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001),
    #my_optimizer = tf.train.RMSPropOptimizer(0.001),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets,
    test_examples=test_examples
)

print("============================= final prediction ====================================")
print(y_pred)
  

# submission
submission = pd.DataFrame({
    "ID": test_ID,
    "target": y_pred
})
submission.to_csv('submission.csv', index=False) 
