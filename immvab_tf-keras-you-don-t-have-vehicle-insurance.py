import tensorflow as tf

from tensorflow import keras



import os

import tempfile



import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



from plotly.offline import init_notebook_mode, iplot 

import plotly.figure_factory as ff

import plotly.graph_objs as go

import plotly.offline as py

import pycountry

py.init_notebook_mode(connected=True)



import sklearn

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



mpl.rcParams['figure.figsize'] = (12, 10)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
df = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')

df.head()
colors = ['#835AF1']



fig = ff.create_distplot([df['Age']], ['Age'], colors=colors,

                         show_curve=True, show_hist=True)



# Add title

fig.update(layout_title_text='Distribution of Age')

fig.show()
df['Response'].value_counts()
# Preprocess Block



gender = {'Male': 0, 'Female': 1}

driving_license = {0: 0, 1: 1}

previously_insured = {0: 1, 1: 0}

vehicle_age = {'> 2 Years': 3, '1-2 Year': 2, '< 1 Year': 1}

vehicle_damage = {'Yes': 1, 'No': 0}



def preprocess(df):

    df['Gender'] = df['Gender'].map(gender)

    df['Driving_License'] = df['Driving_License'].map(driving_license)

    df['Previously_Insured'] = df['Previously_Insured'].map(previously_insured)

    df['Vehicle_Age'] = df['Vehicle_Age'].map(vehicle_age)

    df['Vehicle_Damage'] = df['Vehicle_Damage'].map(vehicle_damage)



    df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].apply(lambda x: np.int(x))

    df['Region_Code'] = df['Region_Code'].apply(lambda x: np.int(x))



    return df.drop('id', axis = 1)
df = preprocess(df)
df.describe()
train, val = train_test_split(df, test_size=0.1)

print(len(train), 'train examples')

print(len(val), 'validation examples')
train_labels = np.array(train['Response'])

val_labels = np.array(val['Response'])

train = train.drop('Response', axis = 1)

val = val.drop('Response', axis = 1)

bool_train_labels = train_labels != 0
scaler = StandardScaler()

train_features = scaler.fit_transform(train)

val_features = scaler.transform(val)
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



def make_model(metrics = METRICS, output_bias=None):

  if output_bias is not None:

    output_bias = tf.keras.initializers.Constant(output_bias)

  model = keras.Sequential([

      keras.layers.Dense(

          16, activation='relu',

          input_shape=(train_features.shape[-1],)),

      keras.layers.Dense(

          32, activation='relu'),

      keras.layers.Dropout(0.5),

      keras.layers.Dense(1, activation='sigmoid',

                         bias_initializer=output_bias),

  ])



  model.compile(

      optimizer=keras.optimizers.Adam(lr=1e-3),

      loss=keras.losses.BinaryCrossentropy(),

      metrics=metrics)



  return model
EPOCHS = 100

BATCH_SIZE = 2048



early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor='val_auc', 

    verbose=1,

    patience=10,

    mode='max',

    restore_best_weights=True)
model = make_model()

model.summary()
baseline_history = model.fit(

    train_features,

    train_labels,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    callbacks = [early_stopping],

    validation_data=(val_features, val_labels))
def plot_metrics(history):

  metrics =  ['loss', 'auc', 'precision', 'recall']

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

      plt.ylim([0.8,1])

    else:

      plt.ylim([0,1])



    plt.legend()
plot_metrics(baseline_history)
test = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')

test = preprocess(test)

test_features = scaler.fit_transform(test)

preds = model.predict(test_features, batch_size=BATCH_SIZE)
prediction = pd.read_csv('../input/health-insurance-cross-sell-prediction/sample_submission.csv')

prediction['Response'] = preds
prediction.to_csv('submission.csv',index=False)

prediction.head()