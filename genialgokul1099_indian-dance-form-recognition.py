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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Lambda,Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.optimizers import RMSprop,Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
train = pd.read_csv('../input/indian-dance-form-recognition/dataset/train.csv')
test = pd.read_csv('../input/indian-dance-form-recognition/dataset/test.csv')

base = '../input/indian-dance-form-recognition/dataset'
train_dir = os.path.join(str(base)+'/train/')
test_dir = os.path.join(str(base)+'/test/')
train_fname = os.listdir(train_dir)
test_fname = os.listdir(test_dir)
img_wid = 224
img_hei = 224


def train_data_preparation(list_of_images,train,train_dir):
    x=[]
    y=[]
    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(train_dir+image),(img_wid,img_hei),interpolation = cv2.INTER_CUBIC))
        if image in list(train['Image']):
            y.append(train.loc[train['Image'] == image,'target'].values[0])
    return x,y         
import cv2
training_data,training_labels = train_data_preparation(train_fname,train,train_dir)
def test_preparation_Data(list_of_images,test_dir):
    x=[]
    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(test_dir+image),(img_wid,img_hei),interpolation = cv2.INTER_CUBIC))
    return x    
testing_data = test_preparation_Data(test_fname,test_dir)

def show_batch(image_batch,image_label):
    plt.figure(figsize=(12,12))
    for n in range(30):
        ax = plt.subplot(6,6,n+1)
        plt.imshow(image_batch[n])
        plt.title(image_label[n].title())
        plt.axis('off')
        
        
show_batch(training_data,training_labels)
le = LabelEncoder()
training_labels = le.fit_transform(training_labels)
from sklearn.model_selection import train_test_split
X_train,X_valid,Y_train,Y_valid = train_test_split(training_data,training_labels,test_size=0.33,random_state= 42)

train_datagen = image.ImageDataGenerator(rescale=1./255,
                                  featurewise_center = False,
                                  samplewise_center = False,
                                  rotation_range = 40,
                                  zoom_range = 0.20,
                                  width_shift_range = 0.10,
                                  height_shift_range = 0.10,
                                  horizontal_flip=True,
                                  vertical_flip = False)
test_datagen = image.ImageDataGenerator(rescale  = 1./255)

train_datagen.fit(X_train)
test_datagen.fit(X_valid)
test_datagen.fit(testing_data)
X_train = np.array(X_train)
testing__data = np.array(testing_data)
X_valid = np.array(X_valid)

print(X_train.shape)
print(X_valid.shape)
print(Y_train.shape)
print(Y_valid.shape)

vggmodel = VGG16(weights = "imagenet",include_top =False,input_shape = (224,224,3),pooling = "max")
vggmodel.summary()
vggmodel.trainable=False
model = Sequential([
    vggmodel,
    Dense(units=1024,activation = "relu",kernel_initializer="uniform"),
    Dropout(0.25),
    Dense(units=512,activation = 'relu'),
    Dropout(0.25),
    Dense(units = 8,activation ="softmax")
])
reducelearningrate = ReduceLROnPlateau(monitor = 'loss',
                                      factor = 0.1,
                                      patience = 2,
                                      cooldown=2,
                                      min_lr = 0.01,
                                      verbose=1)
class callbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get("accuracy") >= 0.99):
            print("99% accuracy reached,so stopping traiing!!")
            self.model.stop_training = True
callbacks = callbacks()            
earlystop = tf.keras.callbacks.EarlyStopping(
                 monitor='val_loss', patience = 3,
                 min_delta=0.001
             )            
            
model.compile(optimizer = "adam",loss=tf.keras.losses.categorical_crossentropy,metrics = ["accuracy"])

history = model.fit_generator(train_datagen.flow(X_train,to_categorical(Y_train,8),batch_size=16),

                              validation_data = test_datagen.flow(X_valid,to_categorical(Y_valid,8),batch_size=16),verbose=2,epochs=100,
                             callbacks = [reducelearningrate,callbacks])
import matplotlib.pyplot as plt
plt.plot(history.history["accuracy"],label = "accuracy")
plt.plot(history.history["val_accuracy"],label = "val_accuracy")
plt.legend(loc="best")
plt.show()
plt.plot(history.history["loss"],label = "loss")
plt.plot(history.history["val_loss"],label= "val_loss")
plt.legend(loc="best")
plt.show()
testing_data=np.array(testing_data)
prediction = model.predict(testing_data)
prediction=[np.argmax(i) for i in prediction]
target=le.inverse_transform(prediction)
submission = pd.DataFrame({ 'Image': test.Image, 'target': target })
submission.to_csv('output.csv', index=False)
submission
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
basemodel = InceptionResNetV2(include_top=False,
                             weights="imagenet",
                             input_tensor=None,
                             input_shape=(224,224,3),
                             pooling="avg")
basemodel.trainable=False
basemodel.summary()
model_3 = Sequential()
model_3.add(basemodel)
model_3.add(Dense(256, activation='relu'))
model_3.add(BatchNormalization())


model_3.add(Dense(64, activation='relu'))
model_3.add(BatchNormalization())
model_3.add(Dropout(0.5))



model_3.add(Dense(8,activation='softmax'))

model_3.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model_3.summary()
history = model_3.fit_generator(train_datagen.flow(X_train,to_categorical(Y_train,8),batch_size=16),

                              validation_data = test_datagen.flow(X_valid,to_categorical(Y_valid,8),batch_size=16),verbose=2,epochs=100,
                             callbacks = [reducelearningrate,callbacks,earlystop])
testing_data=np.array(testing_data)
prediction = model_3.predict(testing_data)
prediction=[np.argmax(i) for i in prediction]
target=le.inverse_transform(prediction)
submission = pd.DataFrame({ 'Image': test.Image, 'target': target })
submission.to_csv('output.csv', index=False)