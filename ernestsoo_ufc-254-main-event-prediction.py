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
preprocessed = pd.read_csv("/kaggle/input/ufcdata/preprocessed_data.csv")
raw_fight= pd.read_csv("/kaggle/input/ufcdata/raw_total_fight_data.csv")
raw_fighter= pd.read_csv("/kaggle/input/ufcdata/raw_fighter_details.csv")
data = pd.read_csv("/kaggle/input/ufcdata/data.csv")
khabib_r = data['R_fighter'] == "Khabib Nurmagomedov"

'''
for column in data.columns:
    print(column)
    print(data[khabib_r][:1][column])
    print("/n/n")
'''
robert_r = data['R_fighter'] == "Robert Whittaker"

'''
for column in data.columns:
    print(column)
    print(data[robert_r][:1][column])
    print("/n/n")
'''
justin_b = data['B_fighter'] == "Justin Gaethje"

'''
for column in data.columns:
    print(column)
    print(data[justin_b][:1][column])
    print("/n/n")
'''
jared_b = data['B_fighter'] == "Jared Cannonier"

'''
for column in data.columns:
    print(column)
    print(data[jared_b][:1][column])
    print("/n/n")
'''
alexander_b = data['B_fighter'] == "Alexander Volkov"

'''
for column in data.columns:
    print(data[alexander_b][:1][column])
    print("/n/n")
'''
walt_r = data['R_fighter'] == "Walt Harris"

'''
for column in data.columns:
    print(data[walt_r][:1][column])
    print("/n/n")
'''
connor_b = data['B_fighter'] == "Conor McGregor"

'''
for column in data.columns:
    print(data[connor_r][:1][column])
    print("/n/n")
'''
dustin_b = data['B_fighter'] == "Dustin Poirier"

'''
for column in data.columns:
    print(data[dustin_r][:1][column])
    print("/n/n")
'''

test_dict = {}
test_dict_2 = {}
test_dict_3 = {}
test_dict_4 = {}

for column in preprocessed.columns:
    test_dict[column] = 0
    test_dict_2[column] = 0
    test_dict_3[column] = 0
    test_dict_4[column] = 0
test_dict_2['title_bout'] = False
test_dict_2['no_of_rounds'] = 3



for column in preprocessed.columns[:69]:
    if column in data[jared_b][:1].columns:
        test_dict_2[column] = data[jared_b][:1][column].values[0]
test_dict_3['title_bout'] = False
test_dict_3['no_of_rounds'] = 3



for column in preprocessed.columns[:69]:
    if column in data[alexander_b][:1].columns:
        test_dict_3[column] = data[alexander_b][:1][column].values[0]
test_dict['title_bout'] = True
test_dict['no_of_rounds'] = 5
# Blue: Justin Gaethje.
test_dict['B_current_win_streak'] = 4



for column in preprocessed.columns[:69]:
    if column in data[justin_b][:1].columns:
        test_dict[column] = data[justin_b][:1][column].values[0]
test_dict_4['title_bout'] = False
test_dict_4['no_of_rounds'] = 5


for column in preprocessed.columns[:69]:
    if column in data[connor_b][:1].columns:
        test_dict_4[column] = data[connor_b][:1][column].values[0]
for column in preprocessed.columns[69:]:
    if column in data[khabib_r][:1].columns:
        test_dict[column] = data[khabib_r][:1][column].values[0]
for column in preprocessed.columns[69:]:
    if column in data[robert_r][:1].columns:
        test_dict_2[column] = data[robert_r][:1][column].values[0]
for column in preprocessed.columns[69:]:
    if column in data[walt_r][:1].columns:
        test_dict_3[column] = data[walt_r][:1][column].values[0]
for column in preprocessed.columns[69:]:
    
    splitted = column.split("_")
    
    if len(splitted)>1:
        if splitted[0] == "R":
            splitted[0] == "B"
        
        if splitted[0] == "B":
            splitted[0] == "R"
        
    p_column = "_".join(splitted)
            
    if column in data[justin_b][:1].columns:
        test_dict_4[p_column] = data[justin_b][:1][column].values[0]
test_df = pd.DataFrame([test_dict])

test_df 
test_df_2 = pd.DataFrame([test_dict_2])

test_df_2
test_df_3 = pd.DataFrame([test_dict_3])

test_df_3
test_df_4 = pd.DataFrame([test_dict_4])

test_df_4
test_df_3_x = test_df_3.drop(["Winner","title_bout"], axis=1)

test_df_3_x
test_df_4_x = test_df_4.drop(["Winner","title_bout"], axis=1)

test_df_4_x
test_df_x = test_df.drop(["Winner","title_bout"], axis=1)

test_df_x
test_df_2_x = test_df_2.drop(["Winner","title_bout"], axis=1)

test_df_2_x
preprocessed['Winner'].value_counts()

red = 2380
blue = 1212
y = preprocessed["Winner"]

y_encoded = []

for label in y:
    if label == "Red":
        y_encoded.append(0)
    else:
        y_encoded.append(1)
        
from keras.utils import to_categorical
y_one_hot = to_categorical(y_encoded)

X = preprocessed.drop(["Winner","title_bout"], axis=1)

# Remove unneeded Columns
for column in preprocessed.columns[137:]:
    test_df_x.drop([column], axis=1)
    test_df_2_x.drop([column], axis=1)
    test_df_3_x.drop([column], axis=1)
    test_df_4_x.drop([column], axis=1)
    X.drop([column], axis=1)


print(y_one_hot[:5])
print(test_df_x.shape)
print(test_df_2_x.shape)
print(X.shape)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y_encoded,train_size=0.75)

x_val, x_test, y_val, y_test = train_test_split(x_test,y_test,train_size=0.5)
# demonstrate data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

test_df_x = scaler.transform(test_df_x)
test_df_2_x = scaler.transform(test_df_2_x)
test_df_3_x = scaler.transform(test_df_3_x)
x_train.shape[-1]
import keras

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_red = (1 / red)*(len(preprocessed))/2.0 
weight_for_blue = (1 / blue)*(len(preprocessed))/2.0

class_weight = {0: weight_for_red, 1: weight_for_blue}

print('Weight for class Red: {:.2f}'.format(weight_for_red))
print('Weight for class Blue: {:.2f}'.format(weight_for_blue))

import tensorflow as tf
import keras
#initializer = lambda i,  dtype, partition_info=None,shape=(x_train.shape[-1],): tf.keras.initializers.GlorotNormal()

def make_model(metrics = METRICS, output_bias=None, lambda_val= 0.02):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(
          128, activation='relu',
          input_shape=(x_train.shape[-1],),
          kernel_regularizer=keras.regularizers.l2(lambda_val)),
      keras.layers.Dropout(0.5),
      keras.layers.BatchNormalization(),
      keras.layers.Dense(
          128, activation='relu',
          kernel_regularizer=keras.regularizers.l2(lambda_val)),
      keras.layers.Dropout(0.5),
      keras.layers.BatchNormalization(),
      keras.layers.Dense(1, activation='sigmoid'),
      
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(lr=0.0001),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

  print(model.summary())
  return model

EPOCHS = 80
BATCH_SIZE = 32
VERBOSE = 1


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    verbose=VERBOSE,
    patience=10,
    restore_best_weights=True)

weighted_model = make_model()

weighted_history = weighted_model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping],
    validation_data=(x_val, y_val),
    # The class weights go here
    class_weight=class_weight,
    verbose=VERBOSE) 


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_loss(history, label, n):
  # Use a log scale to show the wide range of values.
  plt.semilogy(history.epoch,  history.history['loss'],
               color=colors[n], label='Train '+label)
  plt.semilogy(history.epoch,  history.history['val_loss'],
          color=colors[n], label='Val '+label,
          linestyle="--")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  
  plt.legend()

plot_loss(weighted_history, "loss", 0)
def plot_metrics(history):
  metrics =  ['loss', 'recall', 'precision', 'auc']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.5,1])
    else:
      plt.ylim([0,1])

    plt.legend()

plot_metrics(weighted_history)
def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Blue Wins Detected (True Negatives): ', cm[0][0])
  print('Blue Wins Incorrectly Detected (False Positives): ', cm[0][1])
  print('Red Win Missed (False Negatives): ', cm[1][0])
  print('Red Win Detected (True Positives): ', cm[1][1])
  print('Total Misclassification: ', np.sum(cm[1]))

y_pred= weighted_model.predict(x_test)

from sklearn.metrics import confusion_matrix
plot_cm(y_test, y_pred)


baseline_results = weighted_model.evaluate(x_test, y_test,
                                  batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(weighted_model.metrics_names, baseline_results):
  print(name, ': ', value)
print()

from sklearn.metrics import f1_score

y_pred_max = np.argmax(y_pred, axis = 1)
#y_test_max = np.argmax(y_test, axis = 1)

print(f1_score(y_test, y_pred_max, average="macro"))
print((f1_score(y_test, y_pred_max, average="macro")+f1_score(y_test, y_pred_max, average="micro"))/2)
ind_pred = weighted_model.predict(test_df_x)
ind_pred[0][0]
ind_pred_2 = weighted_model.predict(test_df_2_x)
ind_pred_2
ind_pred_3 = weighted_model.predict(test_df_3_x)
ind_pred_3
ind_pred_4 = weighted_model.predict(test_df_4_x)
ind_pred_4

if ind_pred[0][0] < 0.5:
    print("Red Wins: Khabib Nurmagomedov wins over Justin Gaethje")
else:
    print("Blue Wins: Justin Gaethje wins over Khabib Nurmagomedov")

if ind_pred_2[0][0] < 0.5:
    print("Red Wins: Robert Whittaker wins over Jared Cannonier")
else:
    print("Blue Wins: Jared Cannonier wins over Robert Whittaker")

if ind_pred_3[0][0] < 0.5:
    print("Red Wins: Walt Harris wins over Alexander Volkov")
else:
    print("Blue Wins: Alexander Volkov wins over Walt Harris")
if ind_pred_4[0][0] < 0.5:
    print("Red Wins: Dustin Poirier wins over Conor Mcgregor")
else:
    print("Blue Wins: Conor Mcgregor wins over Dustin Poirier")