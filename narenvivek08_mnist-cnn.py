import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
data=pd.read_csv('../input/digit-recognizer/train.csv')
valid=pd.read_csv('../input/digit-recognizer/test.csv')
data.head()
valid.head()
len(data)
test = data.iloc[33600:, :]
train = data.iloc[:33600, :]
train_x=train.drop('label', axis=1).to_numpy()/255.0
train_x=train_x.reshape(-1,28,28,1)
train_y=train['label'].to_numpy()

test_x=test.drop('label',axis=1).to_numpy()/255.0
test_x=test_x.reshape(-1,28,28,1)
test_y=test['label'].to_numpy()

valid_x=valid.to_numpy()/255.0
valid_x=valid_x.reshape(-1,28,28,1)


plt.imshow(train_x[700].reshape(28,28))
plt.show()
print(train_y[700])
print(train_x.shape,train_y.shape)
print(test_x.shape,test_y.shape)
print(valid_x.shape)
train_datagen = ImageDataGenerator(rescale=1.0,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rotation_range=40,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1.0)
model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    
    tf.keras.layers.Flatten(),
    
    # 128 neuron hidden layer
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])
model.summary()
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit_generator(train_datagen.flow(train_x, train_y, batch_size=64),
                              steps_per_epoch=len(train_x) / 64,
                              epochs=30,
                              validation_data=validation_datagen.flow(test_x, test_y, batch_size=64),
                              validation_steps=len(test_x) / 64
                             )


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
predictions=model.predict(valid_x)
predictions[0]
result=np.argmax(predictions,axis=1)
result
result_df=pd.Series(result,name='Label')
result_df
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result_df],axis = 1)

submission.to_csv("mnist_submission.csv",index=False)
