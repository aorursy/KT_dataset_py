# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import tensorflow as tf

from tensorflow import keras



import os

import tempfile



import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



import sklearn

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load the credit card default data

cc_df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

cc_df.head(5)
# check the dataframe details

cc_df.describe()
# Check for nulls

cc_df.isnull().any()
cc_df.drop('Time', axis=1, inplace=True)

cc_df.head(5)
# Set Matplotlib parameters

mpl.rcParams['figure.figsize'] = (12, 10)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Examine the class

neg, pos = np.bincount(cc_df['Class'])

total = neg+pos

print('Examples in Positive Class:{}, vs. Examples in Negative Class:{}, Total Samples: {}'.format(pos, neg, pos+neg))
eps = 0.001

cc_df['Log Amount'] = np.log(cc_df.pop('Amount') + eps)

cc_df.head(5)
# As we have Imbalanced Dataset overfitting is a classical concern, but lets check how our results come out

# Use a utility from sklearn to split and shuffle our dataset.



train_df, test_df = train_test_split(cc_df, test_size=0.2)

train_df, val_df = train_test_split(train_df, test_size=0.2)



train_labels = np.array(train_df.pop('Class'))

bool_train_labels = train_labels != 0

val_labels = np.array(val_df.pop('Class'))

test_labels = np.array(test_df.pop('Class'))



train_features = np.array(train_df)

test_features = np.array(test_df)

val_features = np.array(val_df)
# Scale the data using Standard Scaler

sscaler = StandardScaler()



train_features = sscaler.fit_transform(train_features)

val_features = sscaler.fit_transform(val_features)

test_features = sscaler.fit_transform(test_features)



train_features = np.clip(train_features, -5, 5)

test_features = np.clip(test_features, -5, 5)

val_features = np.clip(val_features, -5, 5)



print('Training labels shape:', train_labels.shape)

print('Validation labels shape:', val_labels.shape)

print('Test labels shape:', test_labels.shape)



print('Training features shape:', train_features.shape)

print('Validation features shape:', val_features.shape)

print('Test features shape:', test_features.shape)
# Analyze data distribution

pos_df = pd.DataFrame(train_features[bool_train_labels], columns = train_df.columns)

neg_df = pd.DataFrame(train_features[~bool_train_labels], columns = train_df.columns)



sns.jointplot(pos_df['V5'], pos_df['V6'], kind='hex', xlim=(-5,5), ylim=(-5, 5))

plt.suptitle('Positive Distribution')

sns.jointplot(neg_df['V5'], neg_df['V6'], kind='hex', xlim=(-5,5), ylim=(-5, 5))

plt.suptitle('Negative Distribution')
# Define the Keras Model

METRICS = [ keras.metrics.TruePositives(name='tp'),

          keras.metrics.FalsePositives(name='fp'),

          keras.metrics.FalseNegatives(name='fn'),

          keras.metrics.TrueNegatives(name='tn'),

          keras.metrics.BinaryAccuracy(name='accuracy'),

          keras.metrics.Precision(name='precision'),

          keras.metrics.Recall(name='recall'),

          keras.metrics.AUC(name='auc'),

          ]

def make_model(metrics=METRICS, output_bias=None):

    if output_bias is not None:

        output_bias = tf.keras.initializers.Constant(output_bias)

    model = keras.Sequential([

        keras.layers.Dense(16, activation='relu', input_shape=(train_features.shape[-1],)),

        keras.layers.Dropout(0.5),

        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias ),

    ])

    

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3), 

                 loss=keras.losses.BinaryCrossentropy(),

                 metrics=metrics)

    

    return model
# Train the model

EPOCHS = 100

BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', verbose=1,

                                                 patience=10, mode='max', restore_best_weights=True)
model = make_model()

model.summary()
model.predict(train_features[:10])
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)

print("Loss: {:0.4f}".format(results[0]))
# Calculate the initial bias

initial_bias = np.log([pos/neg])

initial_bias
# Set the initial bias 

model = make_model(output_bias=initial_bias)

model.predict(train_features[:10])
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)

print("Loss: {0:.4f}".format(results[0]))
# Checkpoint the initial weights

initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')

model.save_weights(initial_weights)
# Model without Bias

model = make_model()

model.load_weights(initial_weights)

model.layers[-1].bias.assign([0.])

zero_bias_history = model.fit(train_features, train_labels,

                             batch_size=BATCH_SIZE, epochs=20, 

                             validation_data=(val_features, val_labels),

                             verbose=0)
# Model with Initial Bias

model = make_model()

model.load_weights(initial_weights)

careful_bias_history = model.fit(train_features, train_labels,

                             batch_size=BATCH_SIZE, epochs=20, 

                             validation_data=(val_features, val_labels),

                             verbose=0)
# Plot the losses

def plot_loss(history, label, n):

    # Use log scale

    plt.semilogy(history.epoch, history.history['loss'],

                color=colors[n], label='Train '+label)

    plt.semilogy(history.epoch, history.history['val_loss'],

                color=colors[n], label='Val '+label, linestyle='--')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.legend()

    
plot_loss(zero_bias_history, "Zero Bias", 0)

plot_loss(careful_bias_history, "Careful Bias", 1)
# Train the model

model = make_model()

model.load_weights(initial_weights)

baseline_history = model.fit(train_features, train_labels,

                            batch_size=BATCH_SIZE,

                            epochs=EPOCHS, callbacks=[early_stopping],

                            validation_data=(val_features, val_labels))
# Check Training History

# Plots model accuracy and loss to check for overfitting

def plot_metrics(history):

    metrics = ['loss', 'auc', 'precision', 'recall']

    for n, metric in enumerate(metrics):

        name = metric.replace("_"," ").capitalize()

        plt.subplot(2, 2, n+1)

        plt.plot(history.epoch, history.history[metric], color=colors[n], label='Train')

        plt.plot(history.epoch, history.history['val_'+metric], color=colors[n], label='Validation', linestyle='--')

        plt.xlabel('Epoch')

        plt.ylabel(name)

        if metric=='loss':

            plt.ylim([0, plt.ylim()[1]])

        elif metric=='auc':

            plt.ylim([0.8, 1])

        else:

            plt.ylim([0, 1])

            

        plt.legend()
plot_metrics(baseline_history)
# Evaluate Metrics

train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)

test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)
def plot_cm(labels, predictions, p=0.5):

    cm = confusion_matrix(labels, predictions > p)

    plt.figure(figsize=(6, 6))

    sns.heatmap(cm, annot=True, fmt='d')

    plt.title('Confusion matrix @{:.2f}'.format(p))

    plt.ylabel('Actual label')

    plt.xlabel('Predicted label')

    

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])

    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])

    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])

    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])

    print('Total Fraudulent Transactions: ', np.sum(cm[1]))
baseline_results = model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)

for name, value in zip(model.metrics_names, baseline_results):

    print(name, ':', value)

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
plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])

plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[1], linestyle='--')

plt.legend(loc='lower right')
# Scaling by total/2 helps keep the loss to a similar magnitude.

# The sum of the weights of all examples stays the same.

weight_for_0 = (1/ neg) * (total) /2.0

weight_for_1 = (1/pos) * (total) / 2.0



class_weight = {0: weight_for_0, 1: weight_for_1}



print('Weight for class 0: {:.2f}'.format(weight_for_0))

print('Weight for class 1: {:.2f}'.format(weight_for_1))



weighted_model = make_model()

weighted_model.load_weights(initial_weights)



weighted_history = weighted_model.fit(train_features, train_labels,

                                     batch_size =BATCH_SIZE,

                                     epochs=EPOCHS,

                                     callbacks=[early_stopping],

                                     validation_data=(val_features, val_labels),

                                     # Weights

                                     class_weight=class_weight)

# Check Training History

plot_metrics(weighted_history)
train_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)

test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)
weighted_results = weighted_model.evaluate(test_features, test_labels,

                                          batch_size=BATCH_SIZE, verbose=0)

for name, value in zip(weighted_model.metrics_names, weighted_results):

    print(name,':', value)

print()

plot_cm(test_labels, test_predictions_weighted)
plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])

plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')



plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])

plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')





plt.legend(loc='lower right')
pos_features = train_features[bool_train_labels]

neg_features = train_features[~bool_train_labels]



pos_labels = train_labels[bool_train_labels]

neg_labels = train_labels[~bool_train_labels]



pos_features.shape, neg_features.shape
ids = np.arange(len(pos_features))

choices = np.random.choice(ids, len(neg_features))

res_pos_features = pos_features[choices]

res_pos_labels = pos_labels[choices]

res_pos_features.shape
resampled_features = np.concatenate([res_pos_features, neg_features], axis=0 )

resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)



order = np.arange(len(resampled_labels))

np.random.shuffle(order)





resampled_features = resampled_features[order]

resampled_labels = resampled_labels[order]



resampled_features.shape
BUFFER_SIZE = 100000



def make_ds(features,labels):

    ds = tf.data.Dataset.from_tensor_slices((features, labels))

    ds = ds.shuffle(BUFFER_SIZE).repeat()

    return ds



pos_ds = make_ds(pos_features, pos_labels)

neg_ds = make_ds(neg_features, neg_labels)
for features, labels in pos_ds.take(1):

    print("Features:\n", features.numpy())

    print()

    print("Labels:\n", labels.numpy())
resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])

resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)
for features, labels in resampled_ds.take(1):

    print(labels.numpy().mean())
resampled_steps_per_epoch = np.ceil(2.0*neg/BATCH_SIZE)

resampled_steps_per_epoch
resampled_model = make_model()

resampled_model.load_weights(initial_weights)



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
plot_metrics(resampled_history )
resampled_model = make_model()

resampled_model.load_weights(initial_weights)



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
plot_metrics(resampled_history)
train_predictions_resampled = resampled_model.predict(train_features, batch_size=BATCH_SIZE)

test_predictions_resampled = resampled_model.predict(test_features, batch_size=BATCH_SIZE)
resampled_results = resampled_model.evaluate(test_features, test_labels,

                                             batch_size=BATCH_SIZE, verbose=0)

for name, value in zip(resampled_model.metrics_names, resampled_results):

  print(name, ': ', value)

print()



plot_cm(test_labels, test_predictions_resampled)
plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])

plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')



plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])

plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')



plot_roc("Train Resampled", train_labels, train_predictions_resampled,  color=colors[2])

plot_roc("Test Resampled", test_labels, test_predictions_resampled,  color=colors[2], linestyle='--')

plt.legend(loc='lower right')
pos_features.shape[0] + neg_features.shape[0]