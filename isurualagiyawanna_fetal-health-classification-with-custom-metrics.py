import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
print(tf.__version__)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    if filenames:
        csv_path = os.path.join(dirname, filenames[0])
print(csv_path)
from tensorflow.keras import backend as K

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))
df = pd.read_csv(csv_path)
df.head()
df_cols = df.columns.values
Y = df[df_cols[-1]].values
X = df[df_cols[:-1]].values

Y = Y - 1
Y = Y.astype(int)

X, Y = shuffle(X, Y)
Ntrain = int(len(Y) * 0.7)

Xtrain, Xtest = X[:Ntrain], X[Ntrain:]
Ytrain, Ytest = Y[:Ntrain], Y[Ntrain:]
print(Ytrain.shape)
print(Xtrain.shape)
from collections import Counter
classes = list(set(Ytrain))
class_data = dict(Counter(Ytrain))
class_data
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(Ytrain),
                                                 Ytrain)
class_weights = {i : class_weights[i] for i in range(len(set(Ytrain)))}
class_weights
Xscalar = StandardScaler()
Xscalar.fit(Xtrain)

Xtrain = Xscalar.transform(Xtrain)
Xtest = Xscalar.transform(Xtest)
num_epoches = 80
batch_size = 32
val_split = 0.1
def classifier1():
    n_features = Xtrain.shape[1]
    inputs = Input(shape=(n_features,))
    x = Dense(512, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs, outputs)
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['acc',f1_score,precision, recall] 
    )
    history = model.fit(
                    Xtrain,
                    Ytrain,
                    batch_size=batch_size,
                    epochs=num_epoches,
                    validation_split=val_split,
                    class_weight=class_weights
                    )
    return history, model
    
def plot_metrics(history):
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    
    loss_train = np.cumsum(loss_train) / np.arange(1,num_epoches+1)
    loss_val = np.cumsum(loss_val) / np.arange(1,num_epoches+1)
    plt.plot(loss_train, 'r', label='Training loss')
    plt.plot(loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    acc_train = history.history['acc']
    acc_val = history.history['val_acc']
    
    plt.plot(acc_train, 'r', label='Training loss')
    plt.plot(acc_val, 'b', label='validation loss')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    precision_train = history.history['precision']
    precision_val = history.history['val_precision']
    
    plt.plot(precision_train, 'r', label='Training loss')
    plt.plot(precision_val, 'b', label='validation loss')
    plt.title('Training and Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    recall_train = history.history['recall']
    recall_val = history.history['val_recall']
    
    plt.plot(recall_train, 'r', label='Training loss')
    plt.plot(recall_val, 'b', label='validation loss')
    plt.title('Training and Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    f1_score_train = history.history['f1_score']
    f1_score_val = history.history['val_f1_score']
    
    plt.plot(f1_score_train, 'r', label='Training loss')
    plt.plot(f1_score_val, 'b', label='validation loss')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

history1, model1 = classifier1()
plot_metrics(history1)
P = model1.predict(Xtest)
Ypred = P.argmax(axis=1)
model1.evaluate(Xtest, Ytest)
