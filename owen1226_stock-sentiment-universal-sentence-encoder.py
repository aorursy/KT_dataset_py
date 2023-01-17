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
%reload_ext autoreload

%autoreload 2

%matplotlib inline
MODEL_NAME = 'sentiment'



ACTIVATION = 'sigmoid'

BATCH_SIZE = 32



CLASSES = ['negative', 'positive']
df = pd.read_csv('/kaggle/input/stockmarket-sentiment-dataset/stock_data.csv')

print(df.shape)

print(df.columns)

# df.head()
def tanh_to_sigmoid(x):

    return 0 if x == -1 else 1





df['Y'] = df['Sentiment'].apply(lambda x: tanh_to_sigmoid(x))
from sklearn.model_selection import train_test_split



df_train, df_test = train_test_split(df, test_size=0.02, random_state=1)



X_train, X_test = df_train['Text'].values, df_test['Text'].values

Y_train, Y_test = np.asarray(df_train['Y'].tolist()), np.asarray(df_test['Y'].tolist())



print('X:', X_train.shape, X_test.shape)

print(X_train[0])

print('Y:', Y_train.shape, Y_test.shape)

print(Y_train[0])
! pip install tensorflow_text
import tensorflow as tf

import tensorflow_hub as hub

import tensorflow.keras.layers as layers

from tensorflow.keras.models import Model

import tensorflow_text



# Enable to modify cache location with PATH variable: TFHUB_CACHE_DIR=/Users/Cache

use_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3", input_shape=[], dtype=tf.string, trainable=False)





class PreprocessText(layers.Layer):

    def __init__(self, name="preprocess_text", **kwargs):

        super(PreprocessText, self).__init__(name=name, **kwargs)



    def call(self, input):

        data_preprocessor.process_text(input)

        return self.preprocess(input)







def create_model(out_shape, activation='softmax'):

    # Functional

    inputs = layers.Input(shape=(1,), dtype=tf.string)

    

    X = use_layer(tf.squeeze(tf.cast(inputs, tf.string)))

    X = layers.Dense(512, activation='relu')(X)

    X = layers.Dropout(0.3)(X)

    X = layers.Dense(512, activation='relu')(X)

    X = layers.Dropout(0.3)(X)



    outputs = layers.Dense(out_shape, activation=activation)(X)



    model = Model(inputs=inputs, outputs=outputs) 

    return model
model = create_model(Y_train.shape[1] if ACTIVATION == 'softmax' else 1, activation=ACTIVATION)

    

model.summary()
from tensorflow.keras.optimizers import Adam



model.compile(loss = 'binary_crossentropy', 

              optimizer = Adam(lr=3e-4, decay=1e-6, beta_1=0.9, beta_2=0.999), 

              metrics = ['accuracy'])
BEST_WEIGHTS = (f'{MODEL_NAME}_use_best_weights.hdf5')

MODEL_FILE = (f'{MODEL_NAME}_use_model.h5')
from datetime import datetime

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard



monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto')

checkpoint = ModelCheckpoint(filepath=BEST_WEIGHTS, verbose=1, save_best_only=True)   # Save the best model

tensorboard = TensorBoard(log_dir='logs/{} - {}'.format(MODEL_NAME, datetime.now().strftime('%Y%m%d%H%M%S')))
hist = model.fit(

    X_train,

    Y_train,

    batch_size = BATCH_SIZE, 

    epochs = 30, 

    validation_data=(X_test, Y_test),

    validation_split=0.02,

    callbacks = [monitor, checkpoint, tensorboard],

    shuffle=True,

    verbose=1

)
model.load_weights(BEST_WEIGHTS)

model.save(MODEL_FILE)
import matplotlib.pyplot as plt



def plot_train_history(history):

    # plot the cost and accuracy 

    loss_list = history['loss']

    val_loss_list = history['val_loss']

    accuracy_list = history['accuracy']

    val_accuracy_list = history['val_accuracy']

    # epochs = range(len(loss_list))



    # plot the cost

    plt.plot(loss_list, 'b', label='Training cost')

    plt.plot(val_loss_list, 'r', label='Validation cost')

    plt.ylabel('cost')

    plt.xlabel('iterations')

    plt.title('Training and validation cost')

    plt.legend()

    

    plt.figure()

    

    # plot the accuracy

    plt.plot(accuracy_list, 'b', label='Training accuracy')

    plt.plot(val_accuracy_list, 'r', label='Validation accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('iterations')

    plt.title('Training and validation accuracy')

    plt.legend()





plot_train_history(hist.history)
score = model.evaluate(X_test, Y_test)



print ("Test Loss = " + str(score[0]))

print ("Test Accuracy = " + str(score[1]))
Y_test_pred = model.predict(X_test, verbose=1)
# Get threshold is binary classifation



if ACTIVATION == 'sigmoid':

    from sklearn.metrics import roc_curve, auc



    def calculate_optimal_threshold(Y, Y_pred):

        # ROC Curve

        fpr, tpr, thresholds = roc_curve(Y, Y_pred)

        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve')

        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

        plt.xlim([-0.025, 1.025])

        plt.ylim([-0.025, 1.025])

        plt.xlabel('False Positive Rate')

        plt.ylabel('True Positive Rate')

        plt.title('RoC Curve')

        print("AUC: ", roc_auc)



        # Calculate the optimal threshold

        i = np.arange(len(tpr)) # index for df

        roc_df = pd.DataFrame({'threshold' : pd.Series(thresholds, index = i), 

                               'fpr': pd.Series(fpr, index=i), 

                               '1-fpr' : pd.Series(1-fpr, index = i), 

                               'tpr': pd.Series(tpr, index = i), 

                               'diff': pd.Series(tpr - (1-fpr), index = i) })

        opt_threshold = roc_df.iloc[roc_df['diff'].abs().argsort()[:1]]

        print(opt_threshold)



        return opt_threshold['threshold'].values[0]





    threshold = calculate_optimal_threshold(Y_test, Y_test_pred)
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, precision_score, recall_score, classification_report



def analyze(Y, Y_pred, classes, activation="softmax"):

    if activation == "sigmoid":

        Y_cls = Y

        Y_pred_cls = (Y_pred > threshold).astype(float)

    elif activation == "softmax":

        Y_cls = np.argmax(Y, axis=1)

        Y_pred_cls = np.argmax(Y_pred, axis=1)

    

    

    accuracy = accuracy_score(Y_cls, Y_pred_cls)

    print("Accuracy score: {}\n".format(accuracy))

    

    

    rmse = np.sqrt(mean_squared_error(Y, Y_pred))

    print("RMSE score: {}\n".format(rmse))



    

    # plot Confusion Matrix

    print("Confusion Matrix:")

    cm = confusion_matrix(Y_cls, Y_pred_cls)

    print(cm)

    # Plot the confusion matrix as an image.

    plt.matshow(cm)

    # Make various adjustments to the plot.

    num_classes = len(classes)

    plt.colorbar()

    tick_marks = np.arange(num_classes)

    plt.xticks(tick_marks, range(num_classes))

    plt.yticks(tick_marks, range(num_classes))

    plt.xlabel('Predicted')

    plt.ylabel('True')

    

    

    # plot Classification Report

    print("Classification Report:")

    print(classification_report(Y_cls, Y_pred_cls, target_names=classes))







analyze(Y_test, Y_test_pred, CLASSES, ACTIVATION)
def show_mislabeled(X, Y, Y_pred, classes, activation="softmax", num_show = None):

    num_col = 5

    

    if activation == "sigmoid":

        Y_cls = Y

        Y_pred_cls = np.squeeze((Y_pred > threshold).astype(float))

    elif activation == "softmax":

        Y_cls = np.argmax(Y, axis=1)

        Y_pred_cls = np.argmax(Y_pred, axis=1)

    

    mislabeled_indices = np.where(Y_cls != Y_pred_cls)[0]

    print(f'{len(mislabeled_indices)} mislabeled\n')

    

    if num_show is None or num_show > len(mislabeled_indices):

        num_show = len(mislabeled_indices)

        

    for i, index in enumerate(mislabeled_indices[:num_show]):   

        print("{}\nPrediction: {}\nLabel: {}\n\n".format(X[index], 

                                                         classes[int(Y_pred_cls[index])], 

                                                         classes[int(Y_cls[index])]))







show_mislabeled(X_test, Y_test, Y_test_pred, CLASSES, ACTIVATION, 10)