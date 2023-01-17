# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

import requests
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD

# Due to memory allocation limit we are using 10 percent of actual dataset
url1 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data.gz'
url2 =  'https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data_10_percent.gz'

file_name1 = 'kddcup.data.gz'
file_name2 = 'kddcup.data_10_percent.gz'
data_path = tf.keras.utils.get_file(file_name2, origin=url2)

print(data_path)
# read the given data.gz using pandas read csv()

data = pd.read_csv(data_path, header= None)
data
names_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.names'
f1 = requests.get(names_url)
print(f1.text)
attack_types = 'https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/training_attack_types'
types = requests.get(attack_types)
print(types.text)
# Target types data extraction
attack_dict = {}
types_text_split = types.text.split()
for idx in range(0, len(types_text_split)):
    if idx < len(types_text_split) -1:
      attack_dict[types_text_split[idx]] = types_text_split[idx+1]
attack_dict['normal'] = 'normal'
attack_dict
# extract column names from .names file
col_names = []
f1_text_split = f1.text.split('\n')
for idx in range(1, len(f1_text_split)):
  col_name = f1_text_split[idx].split(':')[0]
  if idx == len(f1_text_split)-1:
    col_name = 'target'
  col_names.append(col_name)
col_names
# assign the actual columns to the dataframe
data.columns = col_names
data
# Map the class names based on target column. Lets check the unique values in target column
data.target.value_counts()
# map actual type to another column called 'target_type'
data['target_type'] = data.target.apply(lambda x : attack_dict[x[0:-1]] )
data.target_type.value_counts()
# check missing values 
data.info()
# Identifying categorical features
numeric_cols = data._get_numeric_data().columns # gets all the numeric column names

categorical_cols = list(set(data.columns)-set(numeric_cols))
categorical_cols
# lets look into deeply to identify if there are any other binary data exists or not
binary_cols = []
for col in numeric_cols:
  if len(data[col].unique()) <= 2:
      result = []
      s = data[col].value_counts()
      t = float(len(data[col]))
      for v in s.index:
          result.append("{}({}%)".format(v,round(100*(s[v]/t),1)))
      print("{} - [{}]".format(col, " , ".join(result)))
      binary_cols.append(col)
# combine all categorical column names
for col in binary_cols:
  categorical_cols.append(col)
categorical_cols

def plot_dist(col, ax):
    data[col].value_counts().plot(kind='bar', facecolor='y', ax=ax)
    ax.set_xlabel('{}'.format(col), fontsize=18)
    ax.set_title("{} on KDD Cup".format(col), fontsize= 18)
    plt.xticks(rotation=45)
    return ax
f, ax = plt.subplots(5,2, figsize = (22,40))
f.tight_layout(h_pad=15, w_pad=10, rect=[0, 0.08, 1, 0.93])

categorical_cols_plot = [ col for col in categorical_cols if col!='num_outbound_cmds']
k = 0
for i in range(5):
    for j in range(2):
        plot_dist(categorical_cols_plot[k], ax[i][j])
        k += 1
__ = plt.suptitle("Distributions of Categorical features", fontsize= 23)
# identify remaining numeric features by subtracting categorical columns
numeric_features = list(set(numeric_cols)-set(categorical_cols))
numeric_features
def plot_std_dist(_title):
  df_std = data[numeric_features].std()
  plt.figure(figsize=(25,6))
  plt.plot(list(df_std.index) ,list(df_std.values), 'yo', markersize=25)
  plt.xticks(rotation=90)
  plt.title(_title, fontsize= 18)
  plt.show()

plot_std_dist('Standard Deviation on Nuemeric features')
def apply_zscore(feature):
  mean = data[feature].mean()
  std = data[feature].std()
  data[feature] = (data[feature] - mean) / std

for feature in numeric_features:
  apply_zscore(feature)
plot_std_dist('Standard Deviation on Nuemeric features after applying zscore')
def apply_dummies(df, feature):
    get_dummies = pd.get_dummies(df[feature])
    for x in get_dummies.columns:
        dummy_name = f"{feature}-{x}"
        df[dummy_name] = get_dummies[x]
    df.drop(feature, axis=1, inplace=True)
    return None
data.shape
# apply one hot encoding for categorical columns except the target and target_types
for feature in categorical_cols:
  if feature not in ['target', 'target_type']:
    apply_dummies(data, feature)
data
# convert to numpy arrays
x_features = data.columns.drop(['target','target_type'])
x = data[x_features].values
print('Shape of Independent features data : ' + str(x.shape))
target_type_dummies = pd.get_dummies(data['target_type']) # Multi Class Classification (tartget_type is grouped attack types)
target = target_type_dummies.columns
num_classes = len(target)
y = target_type_dummies.values
print('Shape of Dependent features data : ' + str(y.shape))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=101)
print('Shape of Independent features Train data : ' + str(X_train.shape))
print('Shape of Dependent features Train data : ' + str(y_train.shape))
print('Shape of Independent features Test data: ' + str(X_test.shape))
print('Shape of Dependent features Test data: ' + str(y_test.shape))
# create model with most common parameters 
def seq_model():
    model = Sequential()
    model.add(Dense(x.shape[1],input_dim =x.shape[1],activation = 'relu',kernel_initializer='random_uniform'))
    model.add(Dense(1,activation='relu',kernel_initializer='random_uniform'))
    model.add(Dense(y.shape[1],activation='softmax'))   
    return model
def evaluate_model(model):
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    return None
def plot_results(history, optimizer, loss_fun):
    plt.figure(figsize=(25,6))
    # plot loss during training
    plt.subplot(121)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(122)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.suptitle(f"{optimizer} Optimizer with {loss_fun} as Loss function", fontsize= 23)
    plt.show()
    return None
# model with crossentropy loss and adam optimizer
model = seq_model()
model.compile(loss ='categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto')
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),callbacks=[monitor],verbose=2,epochs=50)
evaluate_model(model)
plot_results(history, 'Adam', 'categorical_crossentropy')
# model with crossentropy loss and SGD optimizer
model = seq_model()
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss ='categorical_crossentropy',optimizer = opt,metrics = ['accuracy'])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto')
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),callbacks=[monitor],verbose=2,epochs=50)
evaluate_model(model)
plot_results(history, 'SGD', 'categorical_crossentropy')