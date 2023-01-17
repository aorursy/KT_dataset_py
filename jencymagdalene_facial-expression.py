# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Label extraction on training data

import os

labels = []

for i in os.listdir('../input/facial-expression-dataset-image-folders-fer2013/data/train/0'):

    labels.append(0)

for i in os.listdir('../input/facial-expression-dataset-image-folders-fer2013/data/train/1'):

    labels.append(1)

for i in os.listdir('../input/facial-expression-dataset-image-folders-fer2013/data/train/2'):

    labels.append(2)

for i in os.listdir('../input/facial-expression-dataset-image-folders-fer2013/data/train/3'):

    labels.append(3)

for i in os.listdir('../input/facial-expression-dataset-image-folders-fer2013/data/train/4'):

    labels.append(4)

for i in os.listdir('../input/facial-expression-dataset-image-folders-fer2013/data/train/5'):

    labels.append(5)

for i in os.listdir('../input/facial-expression-dataset-image-folders-fer2013/data/train/6'):

    labels.append(6)

   
#feature extraction on training data

import cv2

loc1 = '../input/facial-expression-dataset-image-folders-fer2013/data/train/0'

loc2 = '../input/facial-expression-dataset-image-folders-fer2013/data/train/1'

loc3 = '../input/facial-expression-dataset-image-folders-fer2013/data/train/2'

loc4 = '../input/facial-expression-dataset-image-folders-fer2013/data/train/3'

loc5 = '../input/facial-expression-dataset-image-folders-fer2013/data/train/4'

loc6 = '../input/facial-expression-dataset-image-folders-fer2013/data/train/5'

loc7 = '../input/facial-expression-dataset-image-folders-fer2013/data/train/6'

features = []

from tqdm import tqdm

for i in tqdm(os.listdir(loc1)):

    features.append(cv2.imread(os.path.join(loc1,i),0))

    

for i in tqdm(os.listdir(loc2)):

    features.append(cv2.imread(os.path.join(loc2,i),0))

    

for i in tqdm(os.listdir(loc3)):

    features.append(cv2.imread(os.path.join(loc3,i),0))

    

for i in tqdm(os.listdir(loc4)):

    features.append(cv2.imread(os.path.join(loc4,i),0))

    

for i in tqdm(os.listdir(loc5)):

    features.append(cv2.imread(os.path.join(loc5,i),0))

    

for i in tqdm(os.listdir(loc6)):

    features.append(cv2.imread(os.path.join(loc6,i),0))

    

for i in tqdm(os.listdir(loc7)):

    features.append(cv2.imread(os.path.join(loc7,i),0))
#Label extraction on testing data

import os

labels_test = []

for i in os.listdir('../input/facial-expression-dataset-image-folders-fer2013/data/test/0'):

    labels_test.append(0)

for i in os.listdir('../input/facial-expression-dataset-image-folders-fer2013/data/test/1'):

    labels_test.append(1)

for i in os.listdir('../input/facial-expression-dataset-image-folders-fer2013/data/test/2'):

    labels_test.append(2)

for i in os.listdir('../input/facial-expression-dataset-image-folders-fer2013/data/test/3'):

    labels_test.append(3)

for i in os.listdir('../input/facial-expression-dataset-image-folders-fer2013/data/test/4'):

    labels_test.append(4)

for i in os.listdir('../input/facial-expression-dataset-image-folders-fer2013/data/test/5'):

    labels_test.append(5)

for i in os.listdir('../input/facial-expression-dataset-image-folders-fer2013/data/test/6'):

    labels_test.append(6)
#feature extraction on testing data

import cv2

loc1 = '../input/facial-expression-dataset-image-folders-fer2013/data/test/0'

loc2 = '../input/facial-expression-dataset-image-folders-fer2013/data/test/1'

loc3 = '../input/facial-expression-dataset-image-folders-fer2013/data/test/2'

loc4 = '../input/facial-expression-dataset-image-folders-fer2013/data/test/3'

loc5 = '../input/facial-expression-dataset-image-folders-fer2013/data/test/4'

loc6 = '../input/facial-expression-dataset-image-folders-fer2013/data/test/5'

loc7 = '../input/facial-expression-dataset-image-folders-fer2013/data/test/6'

test_features = []

from tqdm import tqdm

for i in tqdm(os.listdir(loc1)):

    test_features.append(cv2.imread(os.path.join(loc1,i),0))

    

for i in tqdm(os.listdir(loc2)):

    test_features.append(cv2.imread(os.path.join(loc2,i),0))

    

for i in tqdm(os.listdir(loc3)):

    test_features.append(cv2.imread(os.path.join(loc3,i),0))

    

for i in tqdm(os.listdir(loc4)):

    test_features.append(cv2.imread(os.path.join(loc4,i),0))

    

for i in tqdm(os.listdir(loc5)):

    test_features.append(cv2.imread(os.path.join(loc5,i),0))

    

for i in tqdm(os.listdir(loc6)):

    test_features.append(cv2.imread(os.path.join(loc6,i),0))

    

for i in tqdm(os.listdir(loc7)):

    test_features.append(cv2.imread(os.path.join(loc7,i),0))
import pandas as pd

train_data = pd.DataFrame()

test_data = pd.DataFrame()
train_data['emotion'] = labels

train_data['pixel_values'] = features

test_data['emotion'] = labels_test

test_data['pixel_values'] = test_features
train_data.head()
test_data.head()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']



def setup_axe(axe,df,title):

    df['emotion'].value_counts(sort=False).plot(ax=axe, kind='bar', rot=0)

    axe.set_xticklabels(emotion_labels)

    axe.set_xlabel("Emotions")

    axe.set_ylabel("Count")

    axe.set_title(title)

    

    # set individual bar lables using above list

    for i in axe.patches:

         axe.text(i.get_x()-.05, i.get_height()+120, \

                str(round((i.get_height()), 2)), fontsize=14, color='red',

                    rotation=0)



import matplotlib.pyplot as plt   

fig, axes = plt.subplots(1,2, figsize=(20,8), sharey=True)

setup_axe(axes[0],train_data,'train')

setup_axe(axes[1],test_data,'test')

plt.show()
import numpy as np

from keras.utils import np_utils 

features = np.array(features).reshape(-1,48,48,1)

test_features = np.array(test_features).reshape(-1,48,48,1)



features = features/255

test_features = test_features/255



labels = np_utils.to_categorical(labels)

labels_test =np_utils.to_categorical(labels_test)
print('Training features shape ',features.shape)

print('Training labels shape',labels.shape)

print('Testing features shape ',test_features.shape)

print('Testing labels shape',labels_test.shape)
from keras.models import Sequential

from keras.layers import Dense , Activation , Dropout ,Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.metrics import categorical_accuracy

from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint

from keras.optimizers import *

from keras.layers.normalization import BatchNormalization

from sklearn.metrics import accuracy_score





model = Sequential()

input_shape = (48,48,1)

model.add(Conv2D(64, (5, 5), input_shape=input_shape,activation='relu', padding='same'))

model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))

model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))

model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(128))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(7))

model.add(Activation('softmax'))

  

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

model.summary()
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min', restore_best_weights=True)
history = model.fit(x=features, 

            y=labels, 

            batch_size=64,

            steps_per_epoch=len(features) / 64,

            epochs=30, 

            verbose=1, 

            callbacks = [es],

            validation_data=(test_features,labels_test),

            shuffle=True)
import matplotlib.pyplot as plt

# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Plot training & validation loss values



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

test_true = np.argmax(labels_test, axis=1)

test_pred = np.argmax(model.predict(test_features), axis=1)

print("CNN Model Accuracy on testing data: {:.4f}".format(accuracy_score(test_true, test_pred)))
from sklearn import metrics

# Predicted values

y_pred = test_pred

# Actual values

y_act = test_true 

# Printing the confusion matrix

# The columns will show the instances predicted for each label,

# and the rows will show the actual number of instances for each label.

print(metrics.confusion_matrix(y_act, y_pred, labels=[0,1,2,3,4,5,6]))

# Printing the precision and recall, among other metrics

print(metrics.classification_report(y_act, y_pred, labels=[0,1,2,3,4,5,6]))
from sklearn.metrics import confusion_matrix

import seaborn as sns



cm = confusion_matrix(y_act, y_pred)

# Normalise

cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=emotion_labels, yticklabels=emotion_labels)

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show(block=False)
objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

y_pos = np.arange(len(objects))

print(y_pos)
import matplotlib.pyplot as plt

def emotion_analysis(emotions):

    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.9)

    plt.tick_params(axis='x', which='both', pad=10,width=4,length=10)

    plt.xticks(y_pos, objects)

    plt.ylabel('percentage')

    plt.title('emotion')

    plt.show()
from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from skimage import io

def predict_image(pic):

    img = image.load_img(pic, grayscale=True, target_size=(48, 48))

    show_img=image.load_img(pic, grayscale=False, target_size=(200, 200))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis = 0)



    x /= 255



    custom = model.predict(x)

    

    emotion_analysis(custom[0])



    x = np.array(x, 'float32')

    x = x.reshape([48, 48]);



    plt.gray()

    plt.imshow(show_img)

    plt.show()



    m=0.000000000000000000001

    a=custom[0]

    for i in range(0,len(a)):

        if a[i]>m:

            m=a[i]

            ind=i

        

    print('Expression Prediction:',objects[ind])
predict_image('../input/angryman/angry.jpg')
predict_image('../input/hchildimg/happychild.jpg')
predict_image('../input/fearful/fear.jpg')