batch_size = 32

epochs = 1000

learning_rate = 0.001

img_height = 256

img_width = 256

train_dir = '../input/rgbeurosat/RBG/train'

val_dir = '../input/rgbeurosat/RBG/val'

test_dir = '../input/rgbeurosat/RBG/test'
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_data_generator = ImageDataGenerator(

    rescale = 1./255,

    width_shift_range = 0.15,

    height_shift_range = 0.15,

    horizontal_flip=True,

    zoom_range=0.3

)
validation_data_generator = ImageDataGenerator(

    rescale=1./255

)
train_data = train_data_generator.flow_from_directory(

    batch_size=batch_size,

    directory=train_dir,

    shuffle=True,

    target_size=(img_height, img_width),

    class_mode='categorical'

)
val_data = validation_data_generator.flow_from_directory(

    batch_size=batch_size,

    directory=val_dir,

    shuffle=True,

    target_size=(img_height, img_width),

    class_mode='categorical'

)
# importing libraries



from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout

from keras.optimizers import SGD, Adam



# build a sequential model



model = Sequential()

model.add(InputLayer(input_shape=(img_height, img_width, 3)))



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



model.add(Dense(units=10, activation='softmax'))





# compile model



opt = Adam(lr= learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])



# model summary



model.summary()
from keras.callbacks import ModelCheckpoint



mcp_save = ModelCheckpoint('PKModelGIS.h5', save_best_only=True, monitor='val_loss', mode='min')
history = model.fit_generator(

    train_data,

    epochs=epochs,

    validation_data=val_data,

    steps_per_epoch=24,

    validation_steps=10,

    callbacks=[mcp_save]

)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'vaidation'], loc='upper left')

plt.show()
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'vaidation'], loc='upper left')

plt.show()
test_data_generator = ImageDataGenerator(

    rescale=1./255

)
test_data = test_data_generator.flow_from_directory(

    batch_size=1,

    directory=test_dir,

    shuffle=False,

    target_size=(img_height, img_width),

    class_mode='categorical'

)

idx2label_dict = {test_data.class_indices[k]: k for k in test_data.class_indices}
model.load_weights('PKModelGIS.h5')

y_pred = model.predict_generator(test_data, steps=2700)

y_true = test_data.classes

print(y_pred.shape,y_true.shape)
import numpy as np



y_pred = np.argmax(y_pred, axis = 1)
from sklearn.metrics import confusion_matrix

import seaborn as sn

import pandas as pd
def get_key(mydict,val): 

    for key, value in mydict.items(): 

         if val == value: 

             return key 
def calReCall(y_true,y_pred,idx2label_dict,class_name):

    cm = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(cm, index = [idx2label_dict[int(i)] for i in "0123456789"],

                  columns = [idx2label_dict[int(i)] for i in "0123456789"])

    plt.figure(figsize = (10,7))

    sn.heatmap(df_cm, annot=True,linewidths=.5)

    id = get_key(idx2label_dict,class_name)

    out = np.sum(cm, axis=1)

    return cm[id][id]/out[id]
recallVal = calReCall(y_true,y_pred,idx2label_dict,'Residential')

print("Recall: ",recallVal)
cm = confusion_matrix(y_true, y_pred)

print(cm)



cm_sum = np.sum(cm)

col_sum = np.sum(cm, axis = 0)

row_sum = np.sum(cm, axis = 1)



class_acc = []



row = len(cm)



for x in range(0,row):

    tp = cm[x][x] 

    fp = row_sum[x] + col_sum[x] - cm[x][x]

    fn = col_sum[x] + row_sum[x] - cm[x][x]

    tn = cm_sum - row_sum[x]- col_sum[x] + cm[x][x]

    

    temp = (tp+tn)/(tp+fn+fp+tn)

    class_acc.append(temp)

    

print(class_acc)



diag_sum = cm.trace()

print(diag_sum/cm_sum)