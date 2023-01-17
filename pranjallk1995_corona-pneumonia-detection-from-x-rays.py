# importing ImageDataGenerator



from keras.preprocessing.image import ImageDataGenerator



# creating generator



datagen = ImageDataGenerator(rescale=1. / 255)



# preparing iterators for each dataset



train_it = datagen.flow_from_directory('../input/covid-xrays/X-rays/train', class_mode='categorical')

val_it = datagen.flow_from_directory('../input/covid-xrays/X-rays/val', class_mode='categorical')

test_it = datagen.flow_from_directory('../input/covid-xrays/X-rays/test', class_mode='categorical')
# confirming that the iterators work



batchX, batchy = train_it.next()

print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
# importing libraries



from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout

from keras.optimizers import SGD



# build a sequential model



model = Sequential()

model.add(InputLayer(input_shape=(256, 256, 3)))



# 1st conv block



model.add(Conv2D(8, (3, 3), activation='relu', strides=(1, 1), padding='same'))

model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

model.add(BatchNormalization())



# 2nd conv block



model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same'))

model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

model.add(BatchNormalization())



# 3rd conv block



model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'))

model.add(BatchNormalization())



# 4th conv block



model.add(Conv2D(16, (3, 3), activation='relu', strides=(1,1), padding='same'))

model.add(BatchNormalization())



# 5th conv block



model.add(Conv2D(32, (3, 3), activation='relu', strides=(1,1), padding='same'))

model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

model.add(BatchNormalization())



# 6th conv block



model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='same'))

model.add(BatchNormalization())



# 7th conv block



model.add(Conv2D(32, (3, 3), activation='relu', strides=(1,1), padding='same'))

model.add(BatchNormalization())



# 8th conv block



model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='same'))

model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

model.add(BatchNormalization())



# 9th conv block



model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1), padding='same'))

model.add(BatchNormalization())



# 10th conv bock



model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='same'))

model.add(BatchNormalization())



# ANN block



model.add(Flatten())

model.add(Dense(units=100, activation='relu'))

model.add(Dense(units=100, activation='relu'))

model.add(Dropout(0.25))



# output layer



model.add(Dense(units=3, activation='softmax'))





# compile model



opt = SGD(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])



# model summary



model.summary()
# fit on data for 50 epochs



history = model.fit_generator(train_it, epochs=50, validation_data=val_it)
# saving model for futute use



model.save("My_model")



# saving weights seperately



model.save_weights("PKModel.h5")
#importing matplotlib



import matplotlib.pyplot as plt



# plotting graph



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# loading model



from keras.models import load_model



model = load_model("My_model")
# getting actual values



test_it2 = datagen.flow_from_directory('../input/covid-xrays/X-rays/val', class_mode='categorical', batch_size=95)

idx2label_dict = {test_it2.class_indices[k]: k for k in test_it2.class_indices}
model.load_weights('PKModel.h5')

y_pred = model.predict_classes(test_it2)

y_true = test_it2.classes

print(y_pred.shape,y_true.shape)
from sklearn.metrics import confusion_matrix

import seaborn as sn

import pandas as pd

import numpy as np
def get_key(mydict,val): 

    for key, value in mydict.items(): 

         if val == value: 

             return key 
def calReCall(y_true,y_pred,idx2label_dict,class_name):

    cm = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(cm, index = [idx2label_dict[int(i)] for i in "012"],

                  columns = [idx2label_dict[int(i)] for i in "012"])

    plt.figure(figsize = (10,7))

    sn.heatmap(df_cm, annot=True,linewidths=.5)

    id = get_key(idx2label_dict,class_name)

    out = np.sum(cm, axis=1)

    return cm[id][id]/out[id]
recallVal = calReCall(y_true,y_pred,idx2label_dict,'Covid-19')

print("Recall: ",recallVal)