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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.utils import resample
from keras.utils.np_utils import to_categorical
from keras.layers import Convolution1D, MaxPool1D, Dense, Input, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model
root = '/kaggle/input/heartbeat/'
train_df = pd.read_csv(root+'mitbih_train.csv', header=None)
test_df = pd.read_csv(root+'mitbih_test.csv', header=None)
train_df.head()
train_df.info()
class_dist = train_df[187].astype(int).value_counts()
class_dist
class_dist.mean()
plt.figure(figsize=(10, 7))
p = class_dist.plot(kind='pie',
                    labels=['N','S','V','F','Q'],
                    autopct='%1.1f%%')
p.add_artist(plt.Circle((0,0), 0.7, color='white'))
plt.title('Class Distribution')
plt.legend()
plt.show()
plt.savefig('Origial Class Distribution.PNG')
#train_df_new
df_0 = train_df[train_df[187]==0].sample(n=20000, random_state=8)
df_1 = resample(train_df[train_df[187]==1], n_samples=20000,replace=True,
                                           random_state=8)
df_2 = resample(train_df[train_df[187]==2], n_samples=20000,replace=True,
                                           random_state=8)
df_3 = resample(train_df[train_df[187]==3], n_samples=20000,replace=True,
                                           random_state=8)
df_4 = resample(train_df[train_df[187]==4], n_samples=20000,replace=True,
                                           random_state=8)
train_df_new = pd.concat([df_0, df_1, df_2, df_3, df_4])
plt.figure(figsize=(10, 7))
p = train_df_new[187].value_counts().plot(kind='pie',
                    labels=['N','S','V','F','Q'],
                    autopct='%1.1f%%')
p.add_artist(plt.Circle((0,0), 0.7, color='white'))
plt.title('Class Distribution: Post Random-Sampling')
plt.legend()
plt.show()
plt.savefig('post random sampling class dist.PNG')
c = train_df_new.groupby(187, group_keys=False)\
        .apply(lambda train_df_new: train_df_new.sample(1))
c
fig, axes = plt.subplots(5, 1, figsize=(16, 11))

leg = iter(['N', 'S', 'V', 'F', 'U'])
colors = iter(['skyblue', 'red', 'lightgreen', 'orange', 'black'])
for i, ax in enumerate(axes.flatten()):
    ax.plot(c.iloc[i, :186].T, color=next(colors))
    ax.legend(next(leg))
plt.title('Sample of different heart-beat types')
plt.show()
plt.savefig('heart beat sample.PNG')
def plot_hist(class_num, min_val = 5, size = 70, title=''):
    img = train_df_new.loc[train_df_new[187]==class_num].values
    img = img[:, min_val: size]
    img_flatten = img.flatten()

    final1 = np.arange(min_val, size)
    for _ in range(img.shape[0]-1):
        tempo1 = np.arange(min_val, size)
        final1 = np.concatenate((final1, tempo1))
    print(len(final1))
    print(len(img_flatten))
    plt.hist2d(final1, img_flatten, bins=(80, 80), cmap=plt.cm.jet)
    plt.title('2D Histogram- '+title)

    plt.show()
    plt.savefig('2D Histogram- '+title+'.PNG')
plot_hist(0, title='Normal Heart Beat')
plot_hist(1, 5, 50, title='Supraventricular ectopic beats')
plot_hist(2, 30, 70, title='Ventricular ectopic beats')
plot_hist(3, 20, 58, title='Fusion beats')
plot_hist(4, 15, 70, title='Unknown beats')
def add_gaussian_noise(signal):
    noise = np.random.normal(0, 0.05, 186)
    return signal+noise
plt.figure(figsize=(14, 7))
tempo = c.iloc[0, :186]
bruiter = add_gaussian_noise(tempo)

# tempo
plt.subplot(2,1,1)
plt.plot(tempo)

plt.title('ECG: BEFORE Gaussion noise additon')

#bruiter
plt.subplot(2,1,2)
plt.plot(bruiter)

plt.title('ECG: AFTER Gaussion noise additon')


plt.show()
plt.savefig('ECG Gaussian noise transformation.PNG')
# data prepapration : Labels
target_train = train_df_new[187]
target_test = test_df[187]

y_train = to_categorical(target_train)
y_test = to_categorical(target_test)
# data prepapration : Features
X_train = train_df_new.iloc[:,:186].values[:,:, np.newaxis]
X_test = test_df.iloc[:,:186].values[:,:, np.newaxis]
def train_model(X_train, y_train, X_test, y_test):
    
    # input signal image shape
    im_shape = (X_train.shape[1], 1)
    
    # Input layer
    inputs_cnn = Input(shape = (im_shape),
                       name='inputs_cnn')
    
    # Block 1
    conv1_1 = Convolution1D(64, (6), activation='relu',
                            input_shape=im_shape)(inputs_cnn)
    conv1_1 = BatchNormalization()(conv1_1)
    
    pool1 = MaxPool1D(pool_size=(3), strides=(2),
                            padding='same')(conv1_1)
    
    # Block 2
    conv2_1 = Convolution1D(64, (3), activation='relu',
                            input_shape=im_shape)(pool1)
    conv2_1 = BatchNormalization()(conv2_1)
    
    pool2 = MaxPool1D(pool_size=(3), strides=(2),
                    padding='same')(conv2_1)

    # Block 3
    conv3_1 = Convolution1D(64, (3), activation='relu',
                            input_shape=im_shape)(pool2)
    conv3_1 = BatchNormalization()(conv3_1)
    
    pool3 = MaxPool1D(pool_size=(3), strides=(2),
                            padding='same')(conv3_1)
    # Flatten
    flatten = Flatten()(pool3)
    
    # Dense Block
    dense1 = Dense(64, activation='relu')(flatten)
    dense2 = Dense(32, activation='relu')(dense1)
    
    # Output Block
    output = Dense(5, activation='softmax', name='output')(dense2)
    
    # compile model
    model = Model(inputs= inputs_cnn, outputs= output)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=8),
                ModelCheckpoint(filepath='best_model.h5',
                                monitor='val_loss',
                                save_best_only=True)]
    # training
    print('Training...')
    history=model.fit(X_train, y_train, epochs=40, batch_size=32,
                      validation_data=(X_test, y_test),
                      callbacks=callbacks)
    
    model.load_weights('best_model.h5')
    
    return (model, history)
def evaluate_model(history, X_test, y_test, model):
    scores = model.evaluate((X_test), y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    print(history)
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()
    
    fig2, ax_loss = plt.subplots()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model - Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()
    plt.savefig('step evalutaion.PNG')
    target_names = [str(i) for i in range(5)]
    
    y_true = []
    for element in y_test:
        y_true.append(np.argmax(element))
    prediction_proba = model.predict(X_test)
    prediction = np.argmax(prediction_proba, axis=1)
    cnf_matrix = confusion_matrix(y_true, prediction)
        
model, history = train_model(X_train, y_train, X_test, y_test)
evaluate_model(history, X_test, y_test, model)
y_pred = model.predict(X_test)
y_test
np.arange(len(y_pred))
y_pred_clean = np.zeros_like(y_pred)
for idx, i in enumerate(np.argmax(y_pred,axis=1)):
    y_pred_clean[idx][i] = 1
print(classification_report(y_test, y_pred_clean))
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred_clean, axis=1))
print(conf_matrix)
plt.figure(figsize=(10, 7))
sns.heatmap(np.corrcoef(conf_matrix))
plt.title('Confusion Matrix Corrleation-Coefficient')
plt.savefig('Confusion Matrix Correlation Coefficient.png')