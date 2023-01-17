import pandas as pd

import pyarrow.parquet as pq # Used to read the data

import os 

import numpy as np

from keras.layers import *

from keras.models import Model

from sklearn.model_selection import train_test_split 

from keras import backend as K 

from keras import optimizers

import tensorflow as tf

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from keras.callbacks import *

%matplotlib inline

import matplotlib.pyplot as plt
# select how many folds will be created

N_SPLITS = 5

# it is just a constant with the measurements data size

sample_size = 800000
def matthews_correlation_coeff(y_true, y_pred):

    '''Calculates the Matthews correlation coefficient measure for quality

    of binary classification problems.

    '''

    y_pred = tf.convert_to_tensor(y_pred, np.float32)

    y_true = tf.convert_to_tensor(y_true, np.float32)



    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    y_pred_neg = 1 - y_pred_pos



    y_pos = K.round(K.clip(y_true, 0, 1))

    y_neg = 1 - y_pos



    tp = K.sum(y_pos * y_pred_pos)

    tn = K.sum(y_neg * y_pred_neg)



    fp = K.sum(y_neg * y_pred_pos)

    fn = K.sum(y_pos * y_pred_neg)



    numerator = (tp * tn - fp * fn)

    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))



    return numerator / (denominator + K.epsilon())
# load the training set metadata, defines which signals are in which order in the data

train_meta = pd.read_csv('../input/vsb-power-line-fault-detection/metadata_train.csv')

# set index, it makes the data access much faster

train_meta = train_meta.set_index(['id_measurement', 'phase'])

train_meta.head()
# load the test set metadata, defines which signals are in which order in the data

test_meta = pd.read_csv('../input/vsb-power-line-fault-detection/metadata_train.csv')

# set index, it makes the data access much faster

test_meta = test_meta.set_index(['id_measurement', 'phase'])

test_meta.head()
!ls ../input/preprocessing-with-python-multiprocessing
df_test_pre = pd.read_csv("../input/preprocessing-with-python-multiprocessing/my_test_combined_scaled.csv.gz", compression="gzip")
df_test_pre.shape
1084640/160
df_train_pre = pd.read_csv("../input/preprocessing-with-python-multiprocessing/my_train_combined_scaled.csv.gz", compression="gzip")

df_train_pre.shape
df_train_pre.columns
df_train_pre.drop("Unnamed: 0", axis=1, inplace=True)

df_test_pre.drop("Unnamed: 0", axis=1, inplace=True)
#number of "observations" in test dataset

df_test_pre.shape[0]/160
df_train_pre.shape
#number of "observations" in training dataset

464640/160
train_meta.index.get_level_values('id_measurement').unique()
pd.set_option('display.max_rows', 5)

df_train_pre.iloc[0:160,:22]
df_train_pre.iloc[0:160,22:44]
df_train_pre.iloc[0:160,44:66]
df_train_pre.iloc[160:320]
pd.reset_option('display.max_rows')
from sklearn.metrics import matthews_corrcoef



# The output of this kernel must be binary (0 or 1), but the output of the NN Model is float (0 to 1).

# So, find the best threshold to convert float to binary is crucial to the result

# this piece of code is a function that evaluates all the possible thresholds from 0 to 1 by 0.01

def threshold_search(y_true, y_proba):

    best_threshold = 0

    best_score = 0

    scores = []

    for threshold in [i * 0.01 for i in range(100)]:

        yp_np = np.array(y_proba)

        yp_bool = yp_np >= threshold

        score = matthews_corrcoef(y_true, yp_bool)

        #score = K.eval(matthews_correlation(y_true.astype(np.float64), (y_proba > threshold).astype(np.float64)))

        scores.append(score)

        if score > best_score:

            print("found better score:"+str(score)+", th="+str(threshold))

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'matthews_correlation': best_score}

    scores_df = pd.DataFrame({"score": scores})

    print("scores plot:")

    scores_df.plot()

    plt.show()

    return search_result
def create_model(input_data):

    input_shape = input_data.shape

    inp = Input(shape=(input_shape[1], input_shape[2],), name="input_signal")

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True, name="lstm1"), name="bi1")(inp)

    x = Bidirectional(CuDNNLSTM(64, return_sequences=False, name="lstm2"), name="bi2")(x)

    #other kernels have used also a custom Attention layer but I leave it out for simplicity here

#    x = Attention(input_shape[1])(x)

    x = Dense(128, activation="relu", name="dense1")(x)

    x = Dense(64, activation="relu", name="dense2")(x)

    x = Dense(1, activation='sigmoid', name="output")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation_coeff])

    return model
#if any of the 3 signals for a measurement id is labeled as faulty, this labels the whole set of 3 as faulty

y = (train_meta.groupby("id_measurement").sum()/3 > 0)["target"]
#to see the number of targets matches the number of rows in training dataset

y.shape
#if using all signal values separately, the number of rows would be 8712, or 2904*3.

#X = df_train_pre.values.reshape(8712, 160, 22)

#but with the current data format I show above, it is 2904 rows, or "observations"

X = df_train_pre.values.reshape(2904, 160, 66)

X.shape
X_test = df_test_pre.values.reshape(6779, 160, 66)
eval_preds = np.zeros(X.shape[0])

label_predictions = []



splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=123).split(X, y))

for idx, (train_idx, val_idx) in enumerate(splits):

    K.clear_session()

    print("Beginning fold {}".format(idx+1))

    train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]



    model = create_model(X)

    #checkpoint to save model with best validation score. keras seems to add val_xxxxx as name for metric to use here

    ckpt = ModelCheckpoint('weights.h5', save_best_only=True, save_weights_only=True, monitor='val_matthews_correlation_coeff', verbose=1, mode='max')

    earlystopper = EarlyStopping(patience=25, verbose=1) 

    model.fit(train_X, train_y, batch_size=128, epochs=50, validation_data=[val_X, val_y], callbacks=[ckpt, earlystopper])

    # loads the best weights saved by the checkpoint

    model.load_weights('weights.h5')



    print("finding threshold")

    predictions = model.predict(val_X, batch_size=512)

    best_threshold = threshold_search(val_y, predictions)['threshold']

    

    print("predicting test set")

    pred = model.predict(X_test, batch_size=300, verbose=1)

    pred_bool = pred > best_threshold

    labels = pred_bool.astype("int32")

    label_predictions.append(labels)

    

label_predictions = [pred.flatten() for pred in label_predictions]



import scipy



# Ensemble with voting

labels = np.array(label_predictions)

#convert list of predictions into set of columns

labels = np.transpose(labels, (1, 0))

#take most common value (0 or 1) or each row

labels = scipy.stats.mode(labels, axis=-1)[0]

labels = np.squeeze(labels)



submission = pd.read_csv('../input/vsb-power-line-fault-detection/sample_submission.csv')

labels3 = np.repeat(labels, 3)

submission['target'] = labels3

submission.to_csv('_voted_submission.csv', index=False)

submission.head()
sum(labels3)