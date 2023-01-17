import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
sns.set(rc={'figure.figsize':(12,10)})
### Fetching data and getting info of data
data = pd.read_csv('https://raw.githubusercontent.com/shrikantnarayankar15/Insaid_term_deep_learning_Advertisement/master/advertisement_success.csv')
data.head()
data.info()
### Finding the unique values in the dataset
data.nunique()
### First Finding the null values
data.isnull().sum()
sns.countplot(x='industry', hue='netgain', data=data)
sns.countplot(x='genre', hue='netgain', data=data)
sns.barplot(x='netgain', y='average_runtime(minutes_per_week)', data=data)
sns.countplot(x='airtime', hue='netgain', data=data)
data['airlocation'].value_counts().nlargest(10).plot(kind='bar')
data.airlocation.value_counts(normalize=True).mul(100)
sns.countplot(x='expensive', hue='netgain', data=data)
data.groupby('netgain')['expensive'].value_counts(normalize=True).mul(100)
sns.countplot(x='money_back_guarantee', hue='netgain', data=data)
corr = data.corr()
sns.heatmap(corr, annot=True)
data.groupby('netgain')['ratings'].mean()
sns.violinplot(x='netgain', y='ratings',data=data, cut=5, width=0.1)
sns.boxplot(y='average_runtime(minutes_per_week)', x='targeted_sex', hue='netgain', data=data)
sns.boxplot(y='average_runtime(minutes_per_week)', x='industry', data=data)
sns.boxplot(y='average_runtime(minutes_per_week)', x='airtime', data=data)
sns.boxplot(y='average_runtime(minutes_per_week)', x='airtime', hue='netgain', data=data)
sns.boxplot(y='average_runtime(minutes_per_week)', x='expensive', data=data)
data.airlocation.value_counts(normalize=True).mul(100).plot(kind='bar')
data.airlocation.value_counts(normalize=True).mul(100)
# 90% of ads are from US we will treat other ads into others
valid_airlocations = ['United-States', 'Mexico']
data.airlocation = data.airlocation.apply(lambda x: x if x in valid_airlocations else 'Others')
data.head()
data['netgain'].value_counts()
category_cols = ['realtionship_status', 'industry', 'genre', 'targeted_sex', 'airtime', 'airlocation', 'expensive', 'money_back_guarantee']
data = pd.get_dummies(data, columns=category_cols)
# Encoding netgain to 0/1
data['netgain'] = data['netgain'].apply(lambda x: 1 if x else 0) 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = data.drop(['id', 'netgain'], axis=1)
y = data['netgain']

# Before passing data to neural network it must be standardized
scaling = StandardScaler()
scaling.fit(X)
X = scaling.transform(X)

#Startify since we have imbalanced dataset 
X_train, X_valid, y_train, y_valid = train_test_split(X, np.array(y), random_state=42, stratify=np.array(y))

num_input_nodes = X_train.shape[1]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout,BatchNormalization
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from functools import partial
from sklearn.metrics import accuracy_score
class NN_model():
  def __init__(self, X, y):
    self.X_train = X
    self.y_train = y
    self.model = Sequential()
    self.model_info = None
  
  def create_nn(self, num_input_nodes, target_output_number, hidden_layers=[100], activation='relu', dropout=False, batch_nom=False):

    #first layer
    self.model.add(Dense(hidden_layers[0], input_dim = num_input_nodes, activation=activation))

    for hidden_layer_size in hidden_layers[1:]:
      self.model.add(Dense(hidden_layer_size, activation=activation))
      if dropout:
        self.model.add(Dropout(0.2))
      if batch_nom:
        self.model.add(BatchNormalization())

    #decide the activation function on number of targets 
    if target_output_number > 1:
      self.model.add(Dense(target_output_number, activation='softmax'))
    else:
      self.model.add(Dense(target_output_number, activation='sigmoid'))
  
  def train(self, optimzer = SGD, learning_rate=0.01, loss='mse', metrics=['accuracy'], val_split=0.2, epochs=10, class_weight=None):
    optimizer = partial(optimzer, learning_rate=learning_rate)()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print(self.model.summary())
    
    start = time.time()
    self.model_info = self.model.fit(X_train, y_train, batch_size=64, \
                       epochs=epochs, verbose=2, validation_split=0.2, callbacks=[callback], class_weight=class_weight)
    end = time.time()
    
    print ("Model took %0.2f seconds to train"%(end - start))
  
  def validate(self, X_valid, y_valid):
    prediction = self.model(X_valid)
    prediction = [1 if i>=0.5 else 0 for i in prediction]
    return accuracy_score(prediction, y_valid)

# defining paramters for the tuning model
from tensorflow.keras.optimizers import *

optimizers = [Adadelta, SGD, Adam, RMSprop, Adagrad]
learning_rate = [0.1, 0.01]
batch_nom = [True, False]
loss = ['binary_crossentropy', 'mae', 'mse']
hidden_layers = [ [100,50], [100]]
optimizers = [RMSprop]
learning_rate = [0.001]
batch_nom = [True]
loss = ['binary_crossentropy']
hidden_layers = [[100,50]]
tf.config.list_physical_devices('GPU')
#hyper parameter tuning the models

All_models = {}
models = {}
counter = 1
for opt in optimizers:
  for lr in learning_rate:
    for batch_norm in batch_nom:
      for los in loss:
        for hid in hidden_layers:
          with tf.device('device:GPU:0'):
            model = NN_model(X_train, y_train)
            model.create_nn(num_input_nodes, 1, hidden_layers=hid, batch_nom=batch_norm, dropout=False)
            model.train(epochs=100,  loss=los)
            All_models['model'+str(counter)] = [opt, lr, batch_norm, los, hid, model.validate(X_valid, y_valid)]
            models['model'+str(counter)] = model
            counter+=1
w = pd.DataFrame(All_models).T.sort_values(by=5, ascending=False)
w.columns = ['Optimizer', 'learning_rate', 'batch_norm', 'loss', 'hidden_layer', 'accuracy']
w
# We got the parameters now
optimizers = RMSprop
learning_rate = 0.001
batch_nom = True
loss = 'binary_crossentropy'
hidden_layers = [100,50]
from sklearn.model_selection import StratifiedKFold

sta = StratifiedKFold(n_splits=5)
preds = []
for train, test in sta.split(X, y):
  X_train, X_test = X[train], X[test]
  y_train, y_test = y[train], y[test]
  model = NN_model(X_train, y_train)
  model.create_nn(num_input_nodes, 1, hidden_layers=hidden_layers, batch_nom=batch_norm, dropout=False)
  model.train(optimzer = optimizers, learning_rate=learning_rate, epochs=100,  loss=loss)
  preds.append([1 if i>=0.5 else 0 for i in model.model.predict(X_valid)])
# Combining all the predictions of each Stratify and and perform blending i.e taking mean
blending_prediction = pd.DataFrame(preds).T.mode(axis=1)
# Accuracy Score increased after blending
accuracy_score(y_valid, blending_prediction)
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
cf_matrix = confusion_matrix(y_valid, blending_prediction)
sns.heatmap(cf_matrix, annot=True)