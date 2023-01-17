# Importing necessary libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D
import seaborn as sns
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score 
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
sns.set()
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.3,
                                   zoom_range=0.3,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('../input/hand-gesture-to-emoji-data/emoji_data/data/Train',
                                                    target_size=(150, 150),
                                                    batch_size=2,
                                                    color_mode='grayscale',
                                                    class_mode="sparse",
                                                    shuffle=True)

validation_generator = test_datagen.flow_from_directory('../input/hand-gesture-to-emoji-data/emoji_data/data/Valid',
                                                        target_size=(150, 150),
                                                        batch_size=2,
                                                        color_mode='grayscale',
                                                        class_mode="sparse")
# Metrics for checking the model performance while training
import tensorflow as tf
def f1score(y, y_pred):
  return f1_score(y, tf.math.argmax(y_pred, axis=1), average='micro')

def custom_f1score(y, y_pred):
  return tf.py_function(f1score, (y, y_pred), tf.double)
# Callback for stopping the learning if the model reaches a f1-score > 0.999 on the CV/test data.
class stop_training_callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_custom_f1score') > 0.98 and logs.get('custom_f1score') > 0.98):
      self.model.stop_training = True
# Model architecture.
K.clear_session()
ip = Input(shape = (150,150,1))
z = Conv2D(filters = 32, kernel_size = (64,64), padding='same', input_shape = (150,150,1), activation='relu')(ip)
z = Conv2D(filters = 64, kernel_size = (16,16), padding='same', input_shape = (150,150,1), activation='relu')(z)
z = Conv2D(filters = 128, kernel_size = (8,8), padding='same', input_shape = (150,150,1), activation='relu')(z)
z = MaxPool2D(pool_size = (4,4))(z)
z = Flatten()(z)
z = Dense(32, activation='relu')(z)
op = Dense(5, activation='softmax')(z)
model = Model(inputs=ip, outputs=op)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.00001), metrics=[custom_f1score])
model.summary()
callbacks = [stop_training_callback()]
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=1, callbacks=callbacks)
model.fit(train_generator,
          epochs=10,
          validation_data=validation_generator,
          callbacks=callbacks,
          verbose=1)

test_generator = test_datagen.flow_from_directory('../input/hand-gesture-to-emoji-data/emoji_data/data/Valid',
                                                    target_size=(150, 150),
                                                    batch_size=1,
                                                    color_mode='grayscale',
                                                    class_mode="sparse")
num = 0
y_pred = []
y_true = []
for img, y_actual in test_generator:
    if num==600:
        break
    pred_label = model.predict(img).argmax()
    y_pred.append(pred_label)
    y_true.append(y_actual[0])
    num+=1
# Confusion matrix to check the misclassified points in test/CV data.
cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, index=np.arange(5), columns=np.arange(5))
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True)
plt.xlabel('predicted class labels')
plt.ylabel('actual class labels')
plt.title('confusion matrix for first 600 images in validation data')
plt.show()
# sample images from test/CV data and their actual and predicted class labels.
plt.figure(figsize=(16,16))
for i,j in enumerate(test_generator):
    if i==16:
        break
    pred_label = model.predict(j[0]).argmax()
    actual_label = j[1][0]
    plt.subplot(4,4,i+1)
    img = j[0][0][:,:,0]
    plt.imshow(img, cmap='gray')
    plt.title(f'actual: {actual_label} | predicted: {pred_label}')
    plt.axis('off')
plt.show()
model.save_weights('model_weights.h5')
