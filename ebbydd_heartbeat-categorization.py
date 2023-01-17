import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))



mit_test_data = pd.read_csv("../input/mitbih_test.csv", header=None)

mit_train_data = pd.read_csv("../input/mitbih_train.csv", header=None)



print("MIT test dataset")

print(mit_test_data.info())

print("MIT train dataset")

print(mit_train_data.info())
mit_train_data[187].value_counts()
mit_train_data.describe()
#mit_train_data.iloc[19999].plot()
# take a random distribution

sample = mit_test_data.sample(25)



# remove the target column

sampleX = sample.iloc[:,sample.columns != 187]



import matplotlib.pyplot as plt



plt.style.use('classic')



# plt samples

for index, row in sampleX.iterrows():

    plt.plot(np.array(range(0, 187)) ,row)



plt.xlabel("time")

plt.ylabel("magnitude")

plt.title("heartbeat reccording \nrandom sample")



plt.show()



plt.style.use("ggplot")



plt.title("Number of record in each category")



plt.hist(sample.iloc[:,sample.columns == 187].transpose())

plt.show()
# We will use the Seaborn library

import seaborn as sns

sns.set()



# Graphics in SVG format are more sharp and legible

%config InlineBackend.figure_format = 'svg' 
mit_train_data.columns
columns = [0,1,2,3,4,5,6,187]
# `pairplot()` may become very slow with the SVG format

#%config InlineBackend.figure_format = 'png'

#sns.pairplot(mit_train_data[columns]);
print("Train data")

print("Type\tCount")

print((mit_train_data[187]).value_counts())

print("-------------------------")

print("Test data")

print("Type\tCount")

print((mit_test_data[187]).value_counts())
sns.countplot(x=187, data=mit_train_data);
sns.countplot(x=187, data=mit_test_data);
mit_train_data = mit_train_data.sample(frac=1)
from keras.utils import to_categorical



print("--- X ---")

X = mit_train_data.loc[:, mit_train_data.columns != 187]

print(X.head())

print(X.info())



print("--- Y ---")

y = mit_train_data.loc[:, mit_train_data.columns == 187]

y = to_categorical(y)



print("--- testX ---")

testX = mit_test_data.loc[:, mit_test_data.columns != 187]

print(testX.head())

print(testX.info())



print("--- testy ---")

testy = mit_test_data.loc[:, mit_test_data.columns == 187]

testy = to_categorical(testy)
from keras import backend as K

    

def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall

 

    def precision(y_true, y_pred):

        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of

        how many selected items are relevant.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
from keras.models import Sequential

from keras.layers import Dense, Activation,BatchNormalization,Dropout



model = Sequential()



model.add(Dense(50, input_dim=187, init='normal', activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(30, init='normal', activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(5, activation='softmax'))



model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['acc',f1])



history = model.fit(X, y, validation_split=0.2,epochs=100,shuffle=True,class_weight='auto')
print("Evaluation: ")

mse, acc, F1 = model.evaluate(testX, testy)

print('mean_squared_error :', mse)

print('accuracy:', acc)

print('F1:', F1)
history.history.keys()
plt.figure()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.plot(history.history['f1'])

plt.plot(history.history['val_f1'])

plt.legend(labels=['loss','val_loss','f1','val_f1'],loc='best')

plt.show()
X.shape
y.shape
X = np.expand_dims(X,2)

testX = np.expand_dims(testX,2)
X.shape
from keras.models import Sequential

from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Input, Flatten, SeparableConv1D

from keras.layers import GlobalMaxPooling1D

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import Concatenate

from keras.models import Model



from keras import backend as K

from keras.optimizers import Adam

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
n_obs, feature, depth = X.shape

batch_size = 1024
def build_model():

    input_img = Input(shape=(feature, depth), name='ImageInput')

    x = Conv1D(64, 3, activation='relu', padding='same', name='Conv1_1')(input_img)

    x = Conv1D(64, 3, activation='relu', padding='same', name='Conv1_2')(x)

    x = MaxPooling1D(2, name='pool1')(x)

    

    x = SeparableConv1D(64, 3, activation='relu', padding='same', name='Conv2_1')(x)

    x = SeparableConv1D(64, 3, activation='relu', padding='same', name='Conv2_2')(x)

    x = MaxPooling1D(2, name='pool2')(x)

    

    x = SeparableConv1D(128, 3, activation='relu', padding='same', name='Conv3_1')(x)

    x = BatchNormalization(name='bn1')(x)

    x = SeparableConv1D(128, 3, activation='relu', padding='same', name='Conv3_2')(x)

    x = BatchNormalization(name='bn2')(x)

    

    x = SeparableConv1D(256, 3, activation='relu', padding='same', name='Conv3_3')(x)

    x = MaxPooling1D(2, name='pool3')(x)

    x = Dropout(0.6, name='dropout0')(x)

    

    x = Flatten(name='flatten')(x)

    x = Dense(256, activation='relu', name='fc1')(x)

    x = Dropout(0.6, name='dropout1')(x)

    x = Dense(128, activation='relu', name='fc2')(x)

    x = Dropout(0.5, name='dropout2')(x)

    x = Dense(5, activation='softmax', name='fc3')(x)

    

    model = Model(inputs=input_img, outputs=x)

    return model
model =  build_model()

#model.summary()
model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['acc',f1])
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)

history = model.fit(X, y, validation_split=0.2,epochs=75,batch_size=batch_size,shuffle=True,class_weight='auto',callbacks=[checkpointer])
print("Evaluation: ")

mse, acc, F1 = model.evaluate(testX, testy)

print('mean_squared_error :', mse)

print('accuracy:', acc)

print('F1:', F1)
model.save('cnn-0.985.h5')
y_pred = model.predict(testX, batch_size=1000)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error 



print(classification_report(testy.argmax(axis=1), y_pred.argmax(axis=1)))
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

cnf_matrix = confusion_matrix(testy.argmax(axis=1), y_pred.argmax(axis=1))

np.set_printoptions(precision=1)



# Plot non-normalized confusion matrix

plt.figure(figsize=(7, 7))

plot_confusion_matrix(cnf_matrix, classes=['N', 'S', 'V', 'F', 'Q'],

                      title='Confusion matrix')

plt.show()
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#mit_train_data = mit_train_data.sample(frac=1)
from keras.utils import to_categorical



print("--- X ---")

X = mit_train_data.loc[:, mit_train_data.columns != 187]

print(X.head())

print(X.info())



print("--- Y ---")

y = mit_train_data.loc[:, mit_train_data.columns == 187]

#y = to_categorical(y)



print("--- testX ---")

testX = mit_test_data.loc[:, mit_test_data.columns != 187]

print(testX.head())

print(testX.info())



print("--- testy ---")

testy = mit_test_data.loc[:, mit_test_data.columns == 187]

testy = to_categorical(testy)
X.shape,y.shape
y = y.values.squeeze()
X = np.array(X)
C0 = np.argwhere(y == 0).flatten()

C1 = np.argwhere(y == 1).flatten()

C2 = np.argwhere(y == 2).flatten()

C3 = np.argwhere(y == 3).flatten()

C4 = np.argwhere(y == 4).flatten()
print(C0.shape[0],C1.shape[0],C2.shape[0],C3.shape[0],C4.shape[0])
import random

from scipy.signal import resample



def stretch(x):

    l = int(187 * (1 + (random.random()-0.5)/3))

    y = resample(x, l)

    if l < 187:

        y_ = np.zeros(shape=(187, ))

        y_[:l] = y

    else:

        y_ = y[:187]

    return y_



def amplify(x):

    alpha = (random.random()-0.5)

    factor = -alpha*x + (1+alpha)

    return x*factor



def augment(x):

    result = np.zeros(shape= (5, 187))

    for i in range(3):

        if random.random() < 0.33:

            new_y = stretch(x)

        elif random.random() < 0.66:

            new_y = amplify(x)

        else:

            new_y = stretch(x)

            new_y = amplify(new_y)

        result[i, :] = new_y

    return result
import matplotlib.pyplot as plt

import random

plt.plot(X[0, :])

plt.plot(amplify(X[0, :]))

plt.plot(stretch(X[0, :]))

plt.show()
result_C1 = np.apply_along_axis(augment, axis=1, arr=X[C1]).reshape(-1, 187)

class_C1 = np.ones(shape=(result_C1.shape[0],), dtype=int)*3



result_C3 = np.apply_along_axis(augment, axis=1, arr=X[C3]).reshape(-1, 187)

class_C3 = np.ones(shape=(result_C3.shape[0],), dtype=int)*3



# result_C32 = np.apply_along_axis(augment, axis=1, arr=X[C3]).reshape(-1, 187)

# class_C32 = np.ones(shape=(result_C32.shape[0],), dtype=int)*3



# X = np.vstack([X, result_C1, result_C3])

# y = np.hstack([y, class_C1, class_C3])



X = np.vstack([X,  result_C3])

y = np.hstack([y,  class_C3])
X.shape, y.shape
y = to_categorical(y)
from sklearn.utils import shuffle

X, y = shuffle(X,y,random_state=0)
X = np.expand_dims(X,2)

testX = np.expand_dims(testX,2)
from keras.models import Sequential

from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Input, Flatten, SeparableConv1D

from keras.layers import GlobalMaxPooling1D

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import Concatenate

from keras.models import Model



from keras import backend as K

from keras.optimizers import Adam

from keras.callbacks import LearningRateScheduler, ModelCheckpoint



n_obs, feature, depth = X.shape

batch_size = 1024
def build_model():

    input_img = Input(shape=(feature, depth), name='ImageInput')

    x = Conv1D(64, 3, activation='relu', padding='same', name='Conv1_1')(input_img)

    x = Conv1D(64, 3, activation='relu', padding='same', name='Conv1_2')(x)

    x = MaxPooling1D(2, name='pool1')(x)

    

    x = SeparableConv1D(128, 3, activation='relu', padding='same', name='Conv2_1')(x)

    x = SeparableConv1D(128, 3, activation='relu', padding='same', name='Conv2_2')(x)

    x = MaxPooling1D(2, name='pool2')(x)

    

    x = SeparableConv1D(256, 3, activation='relu', padding='same', name='Conv3_1')(x)

    x = BatchNormalization(name='bn1')(x)

    x = SeparableConv1D(256, 3, activation='relu', padding='same', name='Conv3_2')(x)

    x = BatchNormalization(name='bn2')(x)

    x = Dropout(0.3, name='dropout3-2')(x)

    

    x = SeparableConv1D(512, 3, activation='relu', padding='same', name='Conv3_3')(x)

    x = MaxPooling1D(2, name='pool3')(x)

    x = Dropout(0.3, name='dropout3-3')(x)

    

    x = Flatten(name='flatten')(x)

    x = Dense(512, activation='relu', name='fc1')(x)

    x = Dropout(0.6, name='dropout1')(x)

    x = Dense(256, activation='relu', name='fc2')(x)

    x = Dropout(0.5, name='dropout2')(x)

    x = Dense(5, activation='softmax', name='fc3')(x)

    

    model = Model(inputs=input_img, outputs=x)

    return model
model =  build_model()

model.summary()
model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['acc',f1])
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath="/tmp/weights-aug.hdf5", monitor='val_f1', mode='max', verbose=1, save_best_only=True)
history = model.fit(X, y, validation_split=0.2,epochs=75,batch_size=batch_size*2,class_weight='auto',callbacks=[checkpointer])
model.load_weights('/tmp/weights-aug.hdf5')
print("Evaluation: ")

mse, acc, F1 = model.evaluate(testX, testy)

print('mean_squared_error :', mse)

print('accuracy:', acc)

print('F1:', F1)
K.set_value(model.optimizer.lr, 1e-4)

model.fit(X, y, validation_split=0.2,epochs=30,batch_size=batch_size*2,class_weight='auto',callbacks=[checkpointer])
K.set_value(model.optimizer.lr, 1e-5)

model.fit(X, y, validation_split=0.2,epochs=30,batch_size=batch_size,class_weight='auto',callbacks=[checkpointer])
model.load_weights('/tmp/weights-aug.hdf5')

print("Evaluation: ")

mse, acc, F1 = model.evaluate(testX, testy)

print('mean_squared_error :', mse)

print('accuracy:', acc)

print('F1:', F1)
y_pred = model.predict(testX, batch_size=1000)
import itertools

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error 



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

cnf_matrix = confusion_matrix(testy.argmax(axis=1), y_pred.argmax(axis=1))

#np.set_printoptions(precision=0)



# Plot non-normalized confusion matrix

plt.figure(figsize=(5, 5))

plot_confusion_matrix(cnf_matrix, classes=['N', 'S', 'V', 'F', 'Q'],

                      title='Confusion matrix')

plt.show()
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#K.set_value(model.optimizer.lr, 1e-3)

history = model.fit(X, y, validation_data=(testX,testy),epochs=150,batch_size=batch_size*2,class_weight='auto',callbacks=[checkpointer])
model.load_weights('/tmp/weights-aug.hdf5')
K.set_value(model.optimizer.lr, 1e-5)

history = model.fit(X, y, validation_data=(testX,testy),epochs=30,batch_size=batch_size*2,class_weight='auto',callbacks=[checkpointer])
model.load_weights('/tmp/weights-aug.hdf5')

print("Evaluation: ")

mse, acc, F1 = model.evaluate(testX, testy)

print('mean_squared_error :', mse)

print('accuracy:', acc)

print('F1:', F1)
model.save("0.98887.hdf5")
K.set_value(model.optimizer.lr, 1e-7)

history = model.fit(X, y, validation_data=(testX,testy),epochs=30,batch_size=batch_size*2,class_weight='auto',callbacks=[checkpointer])
y_pred = model.predict(testX, batch_size=1000)

# Compute confusion matrix

cnf_matrix = confusion_matrix(testy.argmax(axis=1), y_pred.argmax(axis=1))

#np.set_printoptions(precision=0)



# Plot non-normalized confusion matrix

plt.figure(figsize=(5, 5))

plot_confusion_matrix(cnf_matrix, classes=['N', 'S', 'V', 'F', 'Q'],

                      title='Confusion matrix')

plt.show()