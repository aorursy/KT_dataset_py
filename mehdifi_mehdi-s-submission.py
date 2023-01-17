# Import required libraries

import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow import keras



import sklearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns
# Loading training data

raw_train_df = pd.read_csv('/kaggle/input/fi-2020-q2-kaggle-competition/train.csv')



# Loading data for prediction

raw_predict_df = pd.read_csv('/kaggle/input/fi-2020-q2-kaggle-competition/test.csv')

# Taking a copy of the imported data

cleaned_train_df = raw_train_df.copy()



# Adding 0.01 to amounts to avoid zeo values and taking log

cleaned_train_df['Amount'] = np.log(cleaned_train_df['Amount'] + 0.01)



# Same proecdure on data for prediction

cleaned_test_df = raw_predict_df.copy()

cleaned_test_df['Amount'] = np.log(cleaned_test_df['Amount'] + 0.01)
# Splitting data for training, validation and testing

train_df, test_df = train_test_split(cleaned_train_df, test_size=0.2)

train_df, validation_df = train_test_split(train_df, test_size=0.2)
# Creating array of labels

train_labels = np.array(train_df['Class'])



# A logical array of training labels

train_labels_bool = train_labels != 0



# Labels for validation

validation_labels = np.array(validation_df['Class'])



# Labels for test

test_labels = np.array(test_df['Class'])
# Extracting features

train_features = np.array(train_df[train_df.columns[1:-1]])

validation_features = np.array(validation_df[validation_df.columns[1:-1]])

test_features = np.array(test_df[test_df.columns[1:-1]])



# Data structure for prediction data is different

predict_features = np.array(cleaned_test_df[cleaned_test_df.columns[2:]])
# Standardizing numeric features

scaler_obj = StandardScaler()

train_features = scaler_obj.fit_transform(train_features)

validation_features = scaler_obj.fit_transform(validation_features)

test_features = scaler_obj.fit_transform(test_features)

predict_features = scaler_obj.fit_transform(predict_features)



# Checking range of training data

max([max(train_features[i]) for i in range(train_features.shape[1])])

min([min(train_features[i]) for i in range(train_features.shape[1])])
# Limiting the range of features to [-5, 5]

train_features = np.clip(train_features, -5, 5)

validation_features = np.clip(validation_features, -5, 5)

test_features = np.clip(test_features, -5, 5)

predict_features = np.clip(predict_features, -5, 5)
# Helper function that generate model

def model_gen(dense_nodes=16, output_bias=None):

    METRICS = [

        keras.metrics.TruePositives(name='tp'),

        keras.metrics.TrueNegatives(name='tn'),

        keras.metrics.FalsePositives(name='fp'),

        keras.metrics.FalseNegatives(name='fn'),

        keras.metrics.BinaryAccuracy(name='accuracy'),

        keras.metrics.Precision(name='precision'),

        keras.metrics.Recall(name='recall'),

        keras.metrics.AUC(name='auc')

    ]



    if output_bias is not None:

        output_bias = tf.keras.initializers.Constant(output_bias)



    model = keras.Sequential([

        keras.layers.Dense(

            dense_nodes,

            activation='relu',

            input_shape=(train_features.shape[-1],)

        ),

        keras.layers.Dropout(0.5),

        keras.layers.Dense(

            1,

            activation='sigmoid',

            bias_initializer=output_bias

        ),

    ])



    model.compile(

        optimizer=keras.optimizers.Adam(lr=1e-3),

        loss=keras.losses.BinaryCrossentropy(),

        metrics=METRICS

    )



    return model
# Helper for graphing model performance

def graph_model(model):

    metrics = ['loss', 'auc', 'recall']

    for i, metric in enumerate(metrics):

        name = metric.replace("_", " ").capitalize()

        plt.subplot(2,2, i+1)

        plt.plot(

            model.epoch,

            model.history[metric],

            label='Train'

        )



        plt.plot(

            model.epoch,

            model.history['val_'+metric],

            linestyle="--",

            label='Val'

        )

        plt.xlabel('Epoch')

        plt.ylabel(name)

        plt.legend()
# Plot Confusion Matrix

def plot_cm(labels, predictions, p=0.5):

    cm = confusion_matrix(labels, predictions > p)

    plt.figure(figsize=(5, 5))

    sns.heatmap(cm, annot=True, fmt='d')

    plt.title(f"Confusion Matrix @ %{p*100}")

    plt.xlabel("Predicted Label")

    plt.ylabel("Actual Label")



    print(

        f"True Negatives: {cm[0][0]}\n"

        f"False Negatives: {cm[1][0]}\n"

        f"True Positives: {cm[1][1]}\n"

        f"False Positives: {cm[0][1]}"

    )



    plt.show()
# Plot ROC curve

def plot_roc(name, labels, predictions):

    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)



    plt.plot(100*fp, 100*tp, label=name, linewidth=2)

    plt.xlabel("False Positives (%)")

    plt.ylabel("True Positives (%)")

    plt.xlim([-0.5, 20])

    plt.ylim([80, 100.5])

    plt.grid(True)

    ax = plt.gca()

    ax.set_aspect('equal')
# Helper function that return tf dataset

def make_ds(features, labels, buffer_size=100000):

    ds = tf.data.Dataset.from_tensor_slices((features, labels))

    ds = ds.shuffle(buffer_size).repeat()

    return ds

# Helper function to over sample positive values, we do 50/50 on positive/negative records

def resample_ds():

    pos_features = train_features[train_labels_bool]

    neg_features = train_features[~train_labels_bool]



    pos_labels = train_labels[train_labels_bool]

    neg_labels = train_labels[~train_labels_bool]



    pos_ds = make_ds(pos_features, pos_labels)

    neg_ds = make_ds(neg_features, neg_labels)



    resampled_ds = tf.data.experimental.sample_from_datasets(

        [pos_ds, neg_ds],

        weights=[0.5, 0.5]

    )

    resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(1)

    steps = np.ceil(len(neg_labels)/BATCH_SIZE)

    return resampled_ds, steps
# Early stopper to avoid overfitting

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor='val_auc',

    verbose=1,

    patience=10,

    mode='max',

    restore_best_weights=True

)



# Setting parameters

BATCH_SIZE = 2048

EPOCH = 100

NODES = 24
# Using helper function to resample training data

resambled_ds, steps = resample_ds()



# Creating the model

resample_model = model_gen()



# Creating datasets for validation

val_ds = tf.data.Dataset.from_tensor_slices((validation_features, validation_labels)).cache()

val_ds = val_ds.batch(BATCH_SIZE).prefetch(1)



# Training the model

resample_history = resample_model.fit(

    resambled_ds,

    epochs=EPOCH,

    steps_per_epoch=steps,

    callbacks=[early_stopping],

    validation_data=val_ds

)



# Generating predictions on train/test data

train_predict = resample_model.predict(train_features, batch_size=BATCH_SIZE)

test_predict = resample_model.predict(test_features, batch_size=BATCH_SIZE)





# Plotting ROC and CM

plot_roc("Train", train_labels, train_predict)

plot_roc("Test", test_labels, test_predict)

plt.show()

plot_cm(train_labels, train_predict, p=0.6)
# Generating predictions

output_prediction = resample_model.predict(predict_features, batch_size=BATCH_SIZE)



# Setting threshold to categorize records

THRESHOLD = 0.8



predicted = [1 if x[0] > THRESHOLD else 0 for x in output_prediction.tolist()]

output = pd.DataFrame({

    'id': cleaned_test_df['Id'],

    'Predicted': predicted

})



output.to_csv('submission.csv', index=False)
