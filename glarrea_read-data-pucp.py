import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras import regularizers 
%matplotlib inline
from keras import backend as K
def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=2):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
        
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

df = pd.read_csv('../input/train_v2.csv')
print(df.shape)
df.head(10)
np.random.seed(34)
sample = 10000
df = df.sample(sample)
df.shape
img_size = 96

def read_img(path):
    x = cv2.imread('../input/train-jpg/'+path+'.jpg')
    x = cv2.resize(x, (img_size, img_size))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x
from joblib import Parallel, delayed

with Parallel(n_jobs=12, prefer='threads', verbose=1) as ex:
    x = ex(delayed(read_img)(file) for file in df.image_name)
    
x = np.stack(x)
x.shape
labels = sorted({ee for e in df.tags.unique() for ee in e.split(' ')})
labels
for lbl in labels:
    df[lbl] = df.tags.str.contains(lbl)
df.head(10)
y =  df.iloc[:,2:].astype(np.int).values
y
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
x_train.shape, x_val.shape
y_val.shape
x_train.shape
base_model = ResNet50(include_top=False, input_shape=(img_size,img_size,3), pooling='avg')
base_model.trainable = False
base_model.summary()
def plot_img(x, y):
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15,6))
    lbls = [lbl for lbl,prob in zip(labels, y) if prob == 1]
    ax1.imshow(x)
    ax1.set_axis_off()
    ax1.set_title('\n'.join(lbls), size=14)
    ax2.bar(np.arange(len(y)), y)
    ax2.set_xticks(np.arange(len(y)))
    ax2.set_xticklabels(labels, rotation=90)
    plt.show()
idx = np.random.choice(len(x_train))
sample_x, sample_y = x_train[idx], y_train[idx]
plot_img(sample_x, sample_y)
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

top_model = Sequential([
    Dense(256, activation='relu', input_shape=(2048,)),
    Dense(128, activation='relu'),
    Dense(96, activation='relu',kernel_regularizer=regularizers.l2(0.01)),
    Dense(48, activation='relu',kernel_regularizer=regularizers.l2(0.01)),
    Dense(17, activation='sigmoid')
])
top_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0005), metrics=[fbeta_score])
top_model.summary()
final_model = Sequential([base_model, top_model])
final_model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=[fbeta_score])
final_model.summary()
precomputed_train = base_model.predict(x_train, batch_size=256, verbose=1)
precomputed_train.shape
precomputed_val = base_model.predict(x_val, batch_size=256, verbose=1)
precomputed_val.shape
#Si no se realiza precompute
#log1 = final_model.fit(x_train, y_train, batch_size=64, epochs=20,validation_data=[x_val, y_val])
log = top_model.fit(precomputed_train, y_train, epochs=40, batch_size=256, validation_data=[precomputed_val, y_val])
def show_results(log):
    fig, axes = plt.subplots(1, 2, figsize=(14,4))
    ax1, ax2 = axes
    ax1.plot(log.history['loss'], label='train')
    ax1.plot(log.history['val_loss'], label='validation')
    ax1.set_xlabel('epoch'); ax1.set_ylabel('loss')
    ax2.plot(log.history['fbeta_score'], label='train')
    ax2.plot(log.history['val_fbeta_score'], label='validation')
    ax2.set_xlabel('epoch'); ax2.set_ylabel('val_fbeta_score')
    for ax in axes: ax.legend()
show_results(log)

#visualizando los resultados de validación
for it in range(5):
    idx = np.random.choice(len(x_val))
    sample_x, sample_y = x_val[idx], y_val[idx]
    plot_img(sample_x, sample_y)
    plot_img(sample_x, final_model.predict(sample_x[None])[0])

