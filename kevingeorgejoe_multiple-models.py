import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



import keras

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.models import Sequential





from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



from sklearn.model_selection import train_test_split



from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv("../input/output_file.csv")



y = data["label"]

y = y - 1

X = data.drop(columns=["label"], axis=1)



del data



x_train, x_val, y_train, y_val = train_test_split(X.values, y.values, test_size=0.10, random_state=42)
# network parameters 

batch_size = 128

num_classes = 657

epochs = 5 # Further Fine Tuning can be done



# input image dimensions

img_rows, img_cols = 30, 40
# preprocess the train data 

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32')

x_train /= 255



# preprocess the validation data

x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)

x_val = x_val.astype('float32')

x_val /= 255



input_shape = (img_rows, img_cols, 1)



# convert the target variable 

y_train = keras.utils.to_categorical(y_train, num_classes)

y_val = keras.utils.to_categorical(y_val, num_classes)
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
model = Sequential()



# add first convolutional layer

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))



# add second convolutional layer

model.add(Conv2D(64, (3, 3), activation='relu'))



# add one max pooling layer 

model.add(MaxPooling2D(pool_size=(2, 2)))



# add one dropout layer

model.add(Dropout(0.25))



# add flatten layer

model.add(Flatten())



# add dense layer

model.add(Dense(128, activation='relu'))



# add another dropout layer

model.add(Dropout(0.5))



# add dense layer

model.add(Dense(num_classes, activation='softmax'))



# complile the model and view its architecur

model.compile(loss=keras.losses.categorical_crossentropy,  optimizer=keras.optimizers.Adadelta(), metrics=[f1])



model.summary()
%%time

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=128, verbose=1, validation_data=(x_val, y_val))

accuracy = model.evaluate(x_val, y_val, verbose=0)

print('Test accuracy:', accuracy[1])
# list all data in history

print(history.history.keys())
# summarize history for accuracy

plt.plot(history.history['f1'])

plt.plot(history.history['val_f1'])

plt.title('model f1')

plt.ylabel('f1')

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
data = pd.read_csv("../input/output_file.csv")



y = data["label"]

y = y - 1

X = data.drop(columns=["label"], axis=1)



del data



x_train, x_val, y_train, y_val = train_test_split(X.values, y.values, test_size=0.10, random_state=42)
%%time

rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, min_samples_leaf=5)

rf.fit(x_train, y_train)



pred = rf.predict(x_val)



print(accuracy_score(pred, y_val))
%%time

mlb = MultinomialNB()

mlb.fit(x_train, y_train)

pred = mlb.predict(x_val)

print(accuracy_score(pred, y_val))