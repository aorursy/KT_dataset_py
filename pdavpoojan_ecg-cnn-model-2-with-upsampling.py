import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical

from sklearn.utils import class_weight

import warnings

from keras.layers import Dense, Convolution1D, MaxPool1D, Flatten, Dropout

from keras.layers import Input

from keras.models import Model

from keras.layers.normalization import BatchNormalization

import tensorflow.keras as keras

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Softmax, Add, Flatten, Activation, Dropout

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

from tensorflow.keras.optimizers import Adam

import keras

from keras.callbacks import EarlyStopping, ModelCheckpoint

warnings.filterwarnings('ignore')
train_df=pd.read_csv('/kaggle/input/heartbeat/mitbih_train.csv',header=None)

test_df=pd.read_csv('/kaggle/input/heartbeat/mitbih_test.csv',header=None)
from sklearn.utils import resample

df_1=train_df[train_df[187]==1]

df_2=train_df[train_df[187]==2]

df_3=train_df[train_df[187]==3]

df_4=train_df[train_df[187]==4]

df_0=(train_df[train_df[187]==0]).sample(n=20000,random_state=42)



df_1_upsample=resample(df_1,replace=True,n_samples=20000,random_state=123)

df_2_upsample=resample(df_2,replace=True,n_samples=20000,random_state=124)

df_3_upsample=resample(df_3,replace=True,n_samples=20000,random_state=125)

df_4_upsample=resample(df_4,replace=True,n_samples=20000,random_state=126)



train_df=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])
target_train=train_df[187]

target_test=test_df[187]

y_train=to_categorical(target_train)

y_test=to_categorical(target_test)
X_train=train_df.iloc[:,:186].values

X_test=test_df.iloc[:,:186].values

#for i in range(len(X_train)):

#    X_train[i,:186]= add_gaussian_noise(X_train[i,:186])

X_train = X_train.reshape(len(X_train), X_train.shape[1],1)

X_test = X_test.reshape(len(X_test), X_test.shape[1],1)
#CNN model

im_shape=(X_train.shape[1],1)

inp=Input(shape=(im_shape), name='inputs_cnn')

C = Conv1D(filters=32, kernel_size=5, strides=1)(inp)



C11 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(C)

A11 = Activation("relu")(C11)

C12 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A11)

S11 = Add()([C12, C])

A12 = Activation("relu")(S11)

M11 = MaxPooling1D(pool_size=5, strides=2)(A12)





C21 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M11)

A21 = Activation("relu")(C21)

C22 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A21)

S21 = Add()([C22, M11])

A22 = Activation("relu")(S11)

M21 = MaxPooling1D(pool_size=5, strides=2)(A22)





C31 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M21)

A31 = Activation("relu")(C31)

C32 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A31)

S31 = Add()([C32, M21])

A32 = Activation("relu")(S31)

M31 = MaxPooling1D(pool_size=5, strides=2)(A32)





C41 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M31)

A41 = Activation("relu")(C41)

C42 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A41)

S41 = Add()([C42, M31])

A42 = Activation("relu")(S41)

M41 = MaxPooling1D(pool_size=5, strides=2)(A42)





C51 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M41)

A51 = Activation("relu")(C51)

C52 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A51)

S51 = Add()([C52, M41])

A52 = Activation("relu")(S51)

M51 = MaxPooling1D(pool_size=5, strides=2)(A52)



F1 = Flatten()(M51)



D1 = Dense(32)(F1)

A6 = Activation("relu")(D1)

D2 = Dense(32)(A6)

D3 = Dense(5)(D2)

A7 = Softmax()(D3)



model = Model(inputs=inp, outputs=A7)



model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])



callbacks = [

         ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]



history=model.fit(X_train, y_train,epochs=150,callbacks=callbacks, batch_size=16,validation_data=(X_test,y_test))

model.load_weights('best_model.h5')
def evaluate_model(history,X_test,y_test,model):

    scores = model.evaluate((X_test),y_test, verbose=0)

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

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.title('Model- Loss')

    plt.legend(['Training', 'Validation'], loc='upper right')

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.show()

    target_names=['0','1','2','3','4']

    

    y_true=[]

    for element in y_test:

        y_true.append(np.argmax(element))

    prediction_proba=model.predict(X_test)

    prediction=np.argmax(prediction_proba,axis=1)

    cnf_matrix = confusion_matrix(y_true, prediction)
evaluate_model(history,X_test,y_test,model)

y_pred=model.predict(X_test)
import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure(figsize=(10, 10))

plot_confusion_matrix(cnf_matrix, classes=['N', 'S', 'V', 'F', 'Q'],normalize=True,

                      title='Confusion matrix, with normalization')

plt.show()
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
from keras.models import model_from_json



# serialize model to JSON

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")

 

# later...

 

# load json and create model

json_file = open('model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model.h5")

print("Loaded model from disk")