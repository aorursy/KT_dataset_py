# Use seaborn for pairplot

!pip install -q seaborn
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns





# Make numpy printouts easier to read.

np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.layers.experimental import preprocessing



print(tf.__version__)
url = '../input/yeh-concret-data/Concrete_Data_Yeh.csv'

column_names = ["cement","slag","flyash","water","superplasticizer","coarseaggregate","fineaggregate","age","csMPa"

]



raw_dataset = pd.read_csv(url)
dataset = raw_dataset.copy()

dataset.tail()
dataset.isna().sum()
dataset = dataset.dropna()
train_dataset = dataset.sample(frac=0.8, random_state=0)

test_dataset = dataset.drop(train_dataset.index)
sns.pairplot(train_dataset[["cement","slag","flyash","water","superplasticizer","coarseaggregate","fineaggregate","age","csMPa"]], diag_kind='kde')
train_dataset.describe().transpose()
train_features = train_dataset.copy()

test_features = test_dataset.copy()



train_labels = train_features.pop('csMPa')

test_labels = test_features.pop('csMPa')
train_dataset.describe().transpose()[['mean', 'std']]
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())
first = np.array(train_features[:1])



with np.printoptions(precision=2, suppress=True):

  print('First example:', first)

  print()

  print('Normalized:', normalizer(first).numpy())
superplasticizer = np.array(train_features['superplasticizer'])



superplasticizer_normalizer = preprocessing.Normalization(input_shape=[1,])

superplasticizer_normalizer.adapt(superplasticizer)
superplasticizer_model = tf.keras.Sequential([

    superplasticizer_normalizer,

    layers.Dense(units=1)

])



superplasticizer_model.summary()
superplasticizer_model.predict(superplasticizer[:10])
superplasticizer_model.compile(

    optimizer=tf.optimizers.Adam(learning_rate=0.1),

    loss='mean_absolute_error')
%%time

history = superplasticizer_model.fit(

    train_features['superplasticizer'], train_labels,

    epochs=100,

    # suppress logging

    verbose=0,

    # Calculate validation results on 20% of the training data

    validation_split = 0.2)
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
def plot_loss(history):

  plt.plot(history.history['loss'], label='loss')

  plt.plot(history.history['val_loss'], label='val_loss')

  plt.ylim([0, 100])

  plt.xlabel('Epoch')

  plt.ylabel('Error [csMPa]')

  plt.legend()

  plt.grid(True)
plot_loss(history)
test_results = {}



test_results['superplasticizer'] = superplasticizer_model.evaluate(

    test_features['superplasticizer'],

    test_labels, verbose=0)
x = tf.linspace(10, 70, 101)

y = superplasticizer_model.predict(x)
def plot_superplasticizer(x, y):

  plt.scatter(train_features['superplasticizer'], train_labels, label='Data')

  plt.plot(x, y, color='k', label='Predictions')

  plt.xlabel('superplasticizer')

  plt.ylabel('csMPa')

  plt.legend()
plot_superplasticizer(x,y)
linear_model = tf.keras.Sequential([

    normalizer,

    layers.Dense(units=1)

])
linear_model.predict(train_features[:10])
linear_model.layers[1].kernel
linear_model.compile(

    optimizer=tf.optimizers.Adam(learning_rate=0.1),

    loss='mean_absolute_error')
%%time

history = linear_model.fit(

    train_features, train_labels, 

    epochs=100,

    # suppress logging

    verbose=0,

    # Calculate validation results on 20% of the training data

    validation_split = 0.2)
plot_loss(history)
test_results['linear_model'] = linear_model.evaluate(

    test_features, test_labels, verbose=0)
def build_and_compile_model(norm):

  model = keras.Sequential([

      norm,

      layers.Dense(64, activation='relu'),

      layers.Dense(64, activation='relu'),

      layers.Dense(1)

  ])



  model.compile(loss='mean_absolute_error',

                optimizer=tf.keras.optimizers.Adam(0.001))

  return model
dnn_superplasticizer_model = build_and_compile_model(superplasticizer_normalizer)
dnn_superplasticizer_model.summary()
%%time

history = dnn_superplasticizer_model.fit(

    train_features['superplasticizer'], train_labels,

    validation_split=0.2,

    verbose=0, epochs=100)
plot_loss(history)
x = tf.linspace(2, 50, 10)

y = dnn_superplasticizer_model.predict(x)
plot_superplasticizer(x, y)
test_results['dnn_superplasticizer_model'] = dnn_superplasticizer_model.evaluate(

    test_features['superplasticizer'], test_labels,

    verbose=0)
dnn_model = build_and_compile_model(normalizer)

dnn_model.summary()
%%time





history = dnn_model.fit(

    train_features, train_labels,

    validation_split=0.2,

    verbose=0, epochs=100)
plot_loss(history)
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
pd.DataFrame(test_results, index=['Mean absolute error [csMPa]']).T
test_predictions = dnn_model.predict(test_features).flatten()



a = plt.axes(aspect='equal')

plt.scatter(test_labels, test_predictions)

plt.xlabel('True Values [csMPa]')

plt.ylabel('Predictions [csMPa]')

lims = [0, 50]

plt.xlim(lims)

plt.ylim(lims)

_ = plt.plot(lims, lims)

error = test_predictions - test_labels

plt.hist(error, bins=25)

plt.xlabel('Prediction Error [csMPa]')

_ = plt.ylabel('Count')
dnn_model.save('model')
print(test_features)
reloaded = tf.keras.models.load_model('model')



test_results['reloaded'] = reloaded.evaluate(

    test_features, test_labels, verbose=0)
new_data=train_features[:1].copy()

new_data=new_data.replace([500.0, 0.0, 0.0, 200.0,0.0,1125.0,613.0,3],[498.0, 0.0, 0.0, 200.0,0.0,1125.0,613.0,4])

predict=reloaded.predict(new_data)

print(predict)

print(new_data)

tf.keras.utils.plot_model(

    reloaded, to_file='model.png', show_shapes=False, show_layer_names=True,

    rankdir='TB', expand_nested=False, dpi=96)
pd.DataFrame(test_results, index=['Mean absolute error [c]']).T