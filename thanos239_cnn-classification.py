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
df_train = pd.read_csv("../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
df_test = pd.read_csv("../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")
df_train.head()
df_test.head()
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (10,5)) 
sns.countplot(df_train['label'])
y_train = df_train['label']
y_test = df_test['label']
del df_train['label']
del df_test['label']
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)
print(df_train.shape)
x_train= df_train.values.reshape(-1,28,28,1)
x_test=df_test.values.reshape(-1,28,28,1)
x_train.shape
x_test.shape
f, ax = plt.subplots(2,5) 
f.set_size_inches(10, 10)
k = 0
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(x_train[k].reshape(28, 28) , cmap = "gray")
        k += 1
    plt.tight_layout()
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.12,
    zoom_range=0.12,
    horizontal_flip=False,
    fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    
    tf.keras.layers.Dense(24, activation=tf.nn.softmax)]
    )
model.summary()
model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=128),
                              steps_per_epoch=len(x_train) / 128,
                              epochs=30,
                              validation_data=test_datagen.flow(x_test, y_test, batch_size=128),
                              validation_steps=len(x_test) / 128)
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")
%matplotlib inline
import matplotlib.pyplot as plt
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
data_test = pd.read_csv("../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")
y=data_test['label']
predictions = model.predict_classes(x_test)
for i in range(len(predictions)):
    if(predictions[i] >= 9):
        predictions[i] += 1
predictions[:5]   

from sklearn.metrics import classification_report,confusion_matrix
classes = ["Class " + str(i) for i in range(25) if i != 9]
print(classification_report(y, predictions, target_names = classes))
conf_m = confusion_matrix(y,predictions)
conf_m
conf_m = pd.DataFrame(conf_m , index = [i for i in range(25) if i != 9] , columns = [i for i in range(25) if i != 9])

conf_m.head()
plt.figure(figsize = (15,15))
sns.heatmap(conf_m,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='')
print(predictions[:10])
y=np.array(y)
print(y[:10])
correct = np.nonzero(predictions == y)[0]
incorrect = np.nonzero(predictions != y)[0]

#printing correct classes
i = 0
for c in correct[:6]:
    plt.subplot(3,2,i+1)
    plt.imshow(x_test[c].reshape(28,28), cmap="gray", interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(predictions[c], y[c]))
    plt.tight_layout()
    i += 1
#printing incorrectly predicted images

i = 0
for c in incorrect[:6]:
    plt.subplot(3,2,i+1)
    plt.imshow(x_test[c].reshape(28,28), cmap="gray", interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(predictions[c], y[c]))
    plt.tight_layout()
    i += 1
