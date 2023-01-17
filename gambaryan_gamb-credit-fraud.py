# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

from tensorflow import keras



import tempfile



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



import sklearn

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



mpl.rcParams['figure.figsize'] = (12, 12)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']



import warnings

warnings.filterwarnings('ignore')



raw_df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

raw_df.head()

raw_df.describe()




neg, pos = np.bincount(raw_df['Class'])

total = neg + pos

print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(

    total, pos, 100 * pos / total))
cleaned_df = raw_df.copy()



# Удалим столбец со временем

cleaned_df.pop('Time')



# Переведём стоимость в логарифмический масштаб.

eps=0.001 # 0 => 0.1¢ # чтобы избежать возможной ошибки при расчёте логарифма.

cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)

train_df, val_df = train_test_split(train_df, test_size=0.2)



train_labels = np.array(train_df.pop('Class'))

bool_train_labels = train_labels != 0

val_labels = np.array(val_df.pop('Class'))

test_labels = np.array(test_df.pop('Class'))



train_features = np.array(train_df)

val_features = np.array(val_df)

test_features = np.array(test_df)
scaler = StandardScaler()

train_features = scaler.fit_transform(train_features)



val_features = scaler.transform(val_features)

test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)

val_features = np.clip(val_features, -5, 5)

test_features = np.clip(test_features, -5, 5)



print('Training labels shape:', train_labels.shape)

print('Validation labels shape:', val_labels.shape)

print('Test labels shape:', test_labels.shape)



print('Training features shape:', train_features.shape)

print('Validation features shape:', val_features.shape)

print('Test features shape:', test_features.shape)
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
initial_bias = np.log([pos/neg])
model = make_model(output_bias = initial_bias)

model.summary()

model.predict(train_features[:10])
initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')

model.save_weights(initial_weights)
model = make_model()

model.load_weights(initial_weights)

baseline_history = model.fit(

    train_features,

    train_labels,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    callbacks = [early_stopping],

    #verbose=0,

    validation_data=(val_features, val_labels),

    verbose=0)
def plot_cm(labels, predictions, p=0.5):

  cm = confusion_matrix(labels, predictions > p)

  plt.figure(figsize=(5,5))

  sns.heatmap(cm, annot=True, fmt="d")

  plt.title('Confusion matrix @{:.2f}'.format(p))

  plt.ylabel('Actual label')

  plt.xlabel('Predicted label')



  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])

  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])

  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])

  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])

  print('Total Fraudulent Transactions: ', np.sum(cm[1]))
def plot_roc(name, labels, predictions, **kwargs):

  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)



  print('AUC:',sklearn.metrics.auc(fp,tp))



  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)

  plt.title("ROC-curve")

  plt.xlabel('False positives [%]')

  plt.ylabel('True positives [%]')

  plt.xlim([-0.5,20])

  plt.ylim([80,100.5])

  plt.grid(True)

  ax = plt.gca()

  ax.set_aspect('equal')
def plot_pr(name, labels, predictions, **kwargs):

  precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)



  plt.plot(100*precision, 100*recall, label=name, linewidth=2, **kwargs)

  plt.title("PR-curve")

  plt.xlabel('Recall [%]')

  plt.ylabel('Precetion [%]')

  plt.xlim([-0.5,20])

  plt.ylim([80,100.5])

  plt.grid(True)

  ax = plt.gca()

  ax.set_aspect('equal')
baseline_results = model.evaluate(test_features, test_labels,

                                  batch_size=BATCH_SIZE, verbose=0)

for _ in (zip(model.metrics_names,baseline_results)): print(_)
train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)

test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)
plot_cm(test_labels, test_predictions_baseline)
plt.subplot(2,2,1)

plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])

plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plt.legend(loc='lower right')



plt.subplot(2,2,2)

plot_pr("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])

plot_pr("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plt.legend(loc='lower right')
weight_for_0 = (1 / neg)*(total)/2.0 

weight_for_1 = (1 / pos)*(total)/2.0



class_weight = {0: weight_for_0, 1: weight_for_1}



print('Weight for class 0: {:.2f}'.format(weight_for_0))

print('Weight for class 1: {:.2f}'.format(weight_for_1))
weighted_model = make_model()

weighted_model.load_weights(initial_weights)



weighted_history = weighted_model.fit(

    train_features,

    train_labels,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    callbacks = [early_stopping],

    validation_data=(val_features, val_labels),

    # The class weights go here

    class_weight=class_weight,

    verbose=0) 
train_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)

test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)
weighted_results = weighted_model.evaluate(test_features, test_labels,

                                           batch_size=BATCH_SIZE, verbose=0)



plot_cm(test_labels, test_predictions_weighted)
plt.subplot(2,2,1)

plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])

plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')





plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])

plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')

plt.legend(loc='lower right')



plt.subplot(2,2,2)

plot_pr("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])

plot_pr("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')





plot_pr("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])

plot_pr("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')

plt.legend(loc='lower right')
pos_features = train_features[bool_train_labels]

neg_features = train_features[~bool_train_labels]



pos_labels = train_labels[bool_train_labels]

neg_labels = train_labels[~bool_train_labels]

BUFFER_SIZE = 100000







def make_ds(features, labels):

  ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()

  ds = ds.shuffle(BUFFER_SIZE).repeat()

  return ds



pos_ds = make_ds(pos_features, pos_labels)

neg_ds = make_ds(neg_features, neg_labels)





for features, label in pos_ds.take(1):

  print("Features:\n", features.numpy())

  print()

  print("Label: ", label.numpy())





resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])

resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)



for features, label in resampled_ds.take(1):

  print(label.numpy().mean())



resampled_steps_per_epoch = np.ceil(2.0*neg/BATCH_SIZE)

resampled_steps_per_epoch
"""resampled_model = make_model()

resampled_model.load_weights(initial_weights)



# Сбросим значение вектора смещения, т.к. набор данных теперь сбалансирован.

output_layer = resampled_model.layers[-1] 

output_layer.bias.assign([0])





resampled_history = resampled_model.fit(

    resampled_features,

    resampled_labels,

    batch_size=BATCH_SIZE,

    steps_per_epoch=300,

    epochs=EPOCHS,

    callbacks = [early_stopping],

    validation_data=(val_features, val_labels))""" 





resampled_model = make_model()

resampled_model.load_weights(initial_weights)



# Reset the bias to zero, since this dataset is balanced.

output_layer = resampled_model.layers[-1] 

output_layer.bias.assign([0])



val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()

val_ds = val_ds.batch(BATCH_SIZE).prefetch(2) 



resampled_history = resampled_model.fit(

    resampled_ds,

    epochs=10*EPOCHS,

    steps_per_epoch=20,#resampled_steps_per_epoch,

    callbacks = [early_stopping],

    validation_data=val_ds,

    verbose=0)
train_predictions_resampled = resampled_model.predict(train_features, batch_size=BATCH_SIZE)

test_predictions_resampled = resampled_model.predict(test_features, batch_size=BATCH_SIZE)
resampled_results = resampled_model.evaluate(test_features, test_labels,

                                             batch_size=BATCH_SIZE, verbose=0)

plot_cm(test_labels, test_predictions_resampled)
plt.subplot(2,2,1)

plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])

plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')





plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])

plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')





plot_roc("Train Resampled", train_labels, train_predictions_resampled, color=colors[2])

plot_roc("Test Resampled", test_labels, test_predictions_resampled, color=colors[2], linestyle='--')

plt.legend(loc='lower right')



plt.subplot(2,2,2)

plot_pr("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])

plot_pr("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')





plot_pr("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])

plot_pr("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')



plot_pr("Train Resampled", train_labels, train_predictions_resampled, color=colors[2])

plot_pr("Test Resampled", test_labels, test_predictions_resampled, color=colors[2], linestyle='--')

plt.legend(loc='lower right')
clf = RandomForestClassifier(n_estimators=50,class_weight="balanced", random_state=0)

clf.fit(train_features,train_labels)
train_predictions_RandomForest = clf.predict(train_features)

test_predictions_RandomForest = clf.predict(test_features)

plot_cm(test_labels, test_predictions_RandomForest)
plt.subplot(2,2,1)

plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])

plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')





plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])

plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')





plot_roc("Train Resampled", train_labels, train_predictions_resampled, color=colors[2])

plot_roc("Test Resampled", test_labels, test_predictions_resampled, color=colors[2], linestyle='--')



plot_roc("Train RandomForest", train_labels, train_predictions_RandomForest, color=colors[3])

plot_roc("Test RandomForest", test_labels, test_predictions_RandomForest, color=colors[3], linestyle='--')

plt.legend(loc='lower right')



plt.subplot(2,2,2)

plot_pr("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])

plot_pr("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')





plot_pr("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])

plot_pr("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')



plot_pr("Train Resampled", train_labels, train_predictions_resampled, color=colors[2])

plot_pr("Test Resampled", test_labels, test_predictions_resampled, color=colors[2], linestyle='--')



plot_pr("Train RandomForest", train_labels, train_predictions_RandomForest, color=colors[3])

plot_pr("Test RandomForest", test_labels, test_predictions_RandomForest, color=colors[3], linestyle='--')

plt.legend(loc='lower right')

print('\t\tBase Line\t\t Weighted Results\t\t Resampled Results\t\tRandom Forest')

for name, value_baseline_results, value_weighted_results, value_resampled_results in zip(model.metrics_names, baseline_results, weighted_results, resampled_results):

  print('{}\t\t:{:04f}\t\t{:04f}\t\t{:04f}\t\t'.format(name,value_baseline_results,value_weighted_results,value_resampled_results))

print()



print('Random Frorest:')

print("Accuracy:", sklearn.metrics.accuracy_score(test_labels, test_predictions_RandomForest),"\n",

"precision:", sklearn.metrics.precision_score(test_labels, test_predictions_RandomForest), "\n",

"recall", sklearn.metrics.recall_score(test_labels, test_predictions_RandomForest), "\n")