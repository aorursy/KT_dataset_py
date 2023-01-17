import pandas as pd
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
label = train_df["label"]
train_df = train_df.drop(["label"], axis=1)
train_df = train_df.astype('float32') / 255

test_df = test_df.astype('float32') / 255
train_images = train_df.values.reshape((42000,28,28,1))

test_images = test_df.values.reshape((28000,28,28,1))
from sklearn import model_selection
train_x,valid_x,train_y,valid_y = model_selection.train_test_split(train_images,label,

                                                                   test_size=0.1,stratify=label,random_state=0)
from keras.utils import to_categorical
train_labels = to_categorical(train_y)

test_labels = to_categorical(valid_y)
from keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,

                         height_shift_range=0.08, zoom_range=0.08)
from keras import models

from keras import layers

from keras.layers.normalization import BatchNormalization

model = models.Sequential()

model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(28,28,1)))

model.add(layers.MaxPooling2D((2,2)))

model.add(BatchNormalization())

model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(128,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(BatchNormalization())

model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(128,(3,3),activation='relu'))

#model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Dropout(0.4))

model.add(layers.Flatten())

model.add(layers.Dense(128,activation='relu'))

model.add(BatchNormalization())

model.add(layers.Dense(10,activation='softmax'))

model.summary()
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit_generator(gen.flow(train_x,train_labels, batch_size=64),

                              epochs = 20, validation_data = (valid_x,test_labels),

                              steps_per_epoch=train_x.shape[0] // 64)

#model.fit(train_x,train_labels,epochs=20,batch_size=64,validation_data=(valid_x, test_labels))
valid_loss, valid_acc = model.evaluate(valid_x, test_labels)

valid_acc
pred_test = model.predict(test_images)

ytestpred = pred_test.argmax(axis=1)
df = pd.read_csv('../input/sample_submission.csv')

df['Label'] = ytestpred

df.head()
df.to_csv('submission.csv', index=False)