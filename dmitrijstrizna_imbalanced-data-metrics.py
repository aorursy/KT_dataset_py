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

from tensorflow import keras



import sklearn

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

import seaborn as sns

raw_df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

raw_df.head()
raw_df.columns
raw_df[['Amount', 'Class']].info()
neg, pos = np.bincount(raw_df['Class'])

raw_df['Class'].value_counts()
raw_df['Amount'].describe()
raw_df['Time'].describe()
cleaned_df = raw_df.copy()

del cleaned_df['Time']



# convert 'Amount' into log scale

eps=0.001 # for numeric stability: 0 => 0.1¢

cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)

cleaned_df.head()
# Use a utility from sklearn to split and shuffle our dataset.

train_df, test_df = train_test_split(cleaned_df, test_size=0.2)

train_df, val_df = train_test_split(train_df, test_size=0.2)



train_labels = np.array(train_df.pop('Class'))

bool_train_labels = train_labels != 0



val_labels = np.array(val_df.pop('Class'))

test_labels = np.array(test_df.pop('Class'))



train_features = np.array(train_df)

val_features = np.array(val_df)

test_features = np.array(test_df)
train_features[0,:]
# Normalize the input features using the sklearn StandardScaler. This will set the mean to 0 and standard deviation to 1.



# Note: The StandardScaler is only fit using the train_features to be sure the model is not peeking at the validation or test sets.



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)

test_features = scaler.transform(test_features)
# clip (cut) values outside -5, 5 interval

train_features = np.clip(train_features, -5, 5)

val_features = np.clip(val_features, -5, 5)

test_features = np.clip(test_features, -5, 5)
print('Training labels shape:', train_labels.shape)

print('Validation labels shape:', val_labels.shape)

print('Test labels shape:', test_labels.shape)



print('Training features shape:', train_features.shape)

print('Validation features shape:', val_features.shape)

print('Test features shape:', test_features.shape)

pos_df = pd.DataFrame(train_features[ bool_train_labels], columns = train_df.columns)

neg_df = pd.DataFrame(train_features[~bool_train_labels], columns = train_df.columns)



sns.jointplot(pos_df['V5'], pos_df['V6'],

              kind='hex', xlim = (-5,5), ylim = (-5,5))

plt.suptitle("Positive distribution")



sns.jointplot(neg_df['V5'], neg_df['V6'],

              kind='hex', xlim = (-5,5), ylim = (-5,5))

_ = plt.suptitle("Negative distribution")

EPOCHS = 100

BATCH_SIZE = 2048



# Note: Accuracy is not a helpful metric for this task. 

# You can 99.8%+ accuracy on this task by predicting False all the time.

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
model = make_model()

model.summary()
# test run on sample data

model.predict(train_features[:10])
initial_bias = np.log([pos/neg])

initial_bias
model = make_model(output_bias = initial_bias)

model.predict(train_features[:10])
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)

print("Loss: {:0.4f}".format(results[0]))
model.save_weights('initial_weights')

import subprocess

print(subprocess.check_output(['ls','-la']).decode())
model = make_model()

model.load_weights('initial_weights')
careful_bias_history = model.fit(

    train_features,

    train_labels,

    batch_size=BATCH_SIZE,

    epochs=20,

    validation_data=(val_features, val_labels), 

    verbose=1)
model = make_model()

model.load_weights("initial_weights")

model.layers[-1].bias.assign([0.0])

zero_bias_history = model.fit(

    train_features,

    train_labels,

    batch_size=BATCH_SIZE,

    epochs=20,

    validation_data=(val_features, val_labels), 

    verbose=0)
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



#plot_loss(zero_bias_history, "Zero Bias", 0)

#plot_loss(careful_bias_history, "Careful Bias", 1)
early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor='val_auc', 

    verbose=1,

    patience=10,

    mode='max',

    restore_best_weights=True)



model = make_model()

model.load_weights("initial_weights")

baseline_history = model.fit(

    train_features,

    train_labels,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    callbacks = [early_stopping],

    validation_data=(val_features, val_labels))
def plot_metrics(history, metric):

    #metrics =  ['loss', 'auc', 'precision', 'recall']

    #for n, metric in enumerate(metrics):

    #    name = metric.replace("_"," ").capitalize()

    #plt.subplot(2,2,n+1)

    name = metric.replace("_"," ").capitalize()

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
plot_metrics(baseline_history, 'loss')
plot_metrics(baseline_history, 'auc')
plot_metrics(baseline_history, 'precision')
plot_metrics(baseline_history, 'recall')
from sklearn.metrics import confusion_matrix



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
train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)

test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)
baseline_results = model.evaluate(test_features, test_labels,

                                  batch_size=BATCH_SIZE, verbose=0)

for name, value in zip(model.metrics_names, baseline_results):

    print(name, ': ', value)

print()



plot_cm(test_labels, test_predictions_baseline)
def plot_roc(name, labels, predictions, **kwargs):

    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)



    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)

    plt.xlabel('False positives [%]')

    plt.ylabel('True positives [%]')

    plt.xlim([-0.5,20])

    plt.ylim([80,100.5])

    plt.grid(True)

    ax = plt.gca()

    ax.set_aspect('equal')



plt.figure(figsize=[10,10])

plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])

plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plt.legend(loc='lower right')
# Scaling by total/2 helps keep the loss to a similar magnitude.

# The sum of the weights of all examples stays the same.

total = neg + pos

weight_for_0 = (1 / neg)*(total)/2.0 

weight_for_1 = (1 / pos)*(total)/2.0



class_weight = {0: weight_for_0, 1: weight_for_1}



print('Weight for class 0: {:.2f}'.format(weight_for_0))

print('Weight for class 1: {:.2f}'.format(weight_for_1))
weighted_model = make_model()

weighted_model.load_weights("initial_weights")



weighted_history = weighted_model.fit(

    train_features,

    train_labels,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    callbacks = [early_stopping],

    validation_data=(val_features, val_labels),

    # The class weights go here

    class_weight=class_weight) 
train_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)

test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)



weighted_results = weighted_model.evaluate(test_features, test_labels,

                                           batch_size=BATCH_SIZE, verbose=0)

for name, value in zip(weighted_model.metrics_names, weighted_results):

    print(name, ': ', value)

print()



plot_cm(test_labels, test_predictions_weighted)
plt.figure(figsize=[10,10])

plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])

plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')



plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])

plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')





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
resampled_steps_per_epoch = np.ceil(2.0*neg/BATCH_SIZE)

resampled_steps_per_epoch
resampled_model = make_model()

resampled_model.load_weights("initial_weights")



# Reset the bias to zero, since this dataset is balanced.

output_layer = resampled_model.layers[-1] 

output_layer.bias.assign([0])



val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()

val_ds = val_ds.batch(BATCH_SIZE).prefetch(2) 



resampled_history = resampled_model.fit(

    resampled_ds,

    epochs=EPOCHS,

    steps_per_epoch=resampled_steps_per_epoch,

    callbacks = [early_stopping],

    validation_data=val_ds)
plot_metrics(resampled_history, 'loss')
train_predictions_resampled = resampled_model.predict(train_features, batch_size=BATCH_SIZE)

test_predictions_resampled = resampled_model.predict(test_features, batch_size=BATCH_SIZE)



resampled_results = resampled_model.evaluate(test_features, test_labels,

                                           batch_size=BATCH_SIZE, verbose=0)

for name, value in zip(resampled_model.metrics_names, resampled_results):

    print(name, ': ', value)

print()



plot_cm(test_labels, test_predictions_resampled)
resampled_model = make_model()

resampled_model.load_weights("initial_weights")



# Reset the bias to zero, since this dataset is balanced.

output_layer = resampled_model.layers[-1] 

output_layer.bias.assign([0])



resampled_history = resampled_model.fit(

    resampled_ds,

    # These are not real epochs

    steps_per_epoch = 20,

    epochs=10*EPOCHS,

    callbacks = [early_stopping],

    validation_data=(val_ds))
plot_metrics(resampled_history, 'loss')
train_predictions_resampled = resampled_model.predict(train_features, batch_size=BATCH_SIZE)

test_predictions_resampled = resampled_model.predict(test_features, batch_size=BATCH_SIZE)



resampled_results = resampled_model.evaluate(test_features, test_labels,

                                             batch_size=BATCH_SIZE, verbose=0)

for name, value in zip(resampled_model.metrics_names, resampled_results):

    print(name, ': ', value)

print()



plot_cm(test_labels, test_predictions_resampled)
plt.figure(figsize=[10,10])

plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])

plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')



plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])

plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')



plot_roc("Train Resampled", train_labels, train_predictions_resampled,  color=colors[2])

plot_roc("Test Resampled", test_labels, test_predictions_resampled,  color=colors[2], linestyle='--')

plt.legend(loc='lower right')