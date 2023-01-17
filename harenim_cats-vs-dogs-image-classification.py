# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
   # for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
labels = []
for i in os.listdir('../input/dogs-cats-images/dataset/training_set/cats'):
    labels.append(0)
for i in os.listdir('../input/dogs-cats-images/dataset/training_set/dogs'):
    labels.append(1)
import cv2
loc1 = '../input/dogs-cats-images/dataset/training_set/cats'
loc2 = '../input/dogs-cats-images/dataset/training_set/dogs'
features = []
from tqdm import tqdm
for i in tqdm(os.listdir(loc1)):
    f1 = cv2.imread(os.path.join(loc1,i))
    f1 = cv2.resize(f1,(100,100))
    features.append(f1)
    
for i in tqdm(os.listdir(loc2)):
    f2 = cv2.imread(os.path.join(loc2,i))
    f2 = cv2.resize(f2,(100,100))
    features.append(f2)
from keras import layers, models, optimizers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
model.summary()
from keras.preprocessing.image import ImageDataGenerator

# Rescale pixel values from [0, 255] to [0, 1]
train_datagen = ImageDataGenerator(rescale=1./255) 
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    "../input/dogs-cats-images/dataset/training_set/",
    target_size=(100, 100), 
    batch_size=50,
    class_mode='binary') # because we use binary_crossentropy loss we need binary labels

validation_generator = test_datagen.flow_from_directory(
    "../input/dogs-cats-images/dataset/test_set/",
    target_size=(100, 100),
    batch_size=50,
    class_mode='binary')
history = model.fit_generator(
    train_generator,
    steps_per_epoch=160, 
    epochs=20,
    validation_data=validation_generator,
    validation_steps=40) # 40 x 50 == 2000
import matplotlib.pyplot as plt

def plot_accuracy_and_loss(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
plot_accuracy_and_loss(history)

test_generator = test_datagen.flow_from_directory(
    "../input/dogs-cats-images/dataset/test_set/",
    target_size=(100, 100),
    batch_size=50,
    class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=40)
print('test acc:', test_acc)
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt

#filename="../input/dogss-test/images.jpg"
def predict_images(filename):
    img = load_img(filename, target_size=(100, 100))
    plt.imshow(img)
    plt.show()
    img = img_to_array(img)

    img = img.reshape(1, 100, 100, 3)

    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]

    result = model.predict(img)
    return result


 

filename="../input/dogss-test/images.jpg"
result=predict_images(filename)
if(result==0.):
        print("cat")
else :
        print("dog")

from matplotlib import pyplot
from matplotlib.image import imread
# define location of dataset
#folder = '../input/catsdogs/'
# plot first few images
#for i in range(7):
    
   # filename = folder + 'c' + str(i+1) + '.jpg'
filename="../input/catsdogs/c1.jpg"
result=predict_images(filename)
if(result==0.):
    print("cat")
else :
    print("dog")
    
    
	
model.save('catsvsdogs.h5')
import numpy as np
Y = np.array(labels)
X = np.array(features)
from keras.utils import np_utils
Xt = (X - X.mean())/X.std()        #Normalised the data
Yt = np_utils.to_categorical(Y)    #Categorical representation
Xt = Xt.reshape(8000,30000)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(Xt,Yt, test_size = 0.1, random_state = 2)
from sklearn.ensemble import RandomForestClassifier
rmodel = RandomForestClassifier()
rmodel.fit(x_train,y_train)
print(rmodel.score(x_train,y_train))
print(rmodel.score(x_test,y_test))
import matplotlib.pyplot as plt
plt.imshow(x_test[70].reshape(100,100,3))
plt.show()
p = rmodel.predict(x_test[70].reshape(1,30000))
if(np.argmax(p)==0):
    print("cat")
else:
    print("dog")
#RANDOMFORESTCLASSIFIER
import matplotlib.pyplot as plt
for i in range(60,65):
    #plt.subplot(40, 40, i+1)
    plt.imshow(x_test[i+1].reshape(100,100,3))
    
    p = rmodel.predict(x_test[i+1].reshape(1,30000))
    #plt.tight_layout()
    plt.show()

    if(np.argmax(p)==0):
        #plt.xlabel(x_test[i+1] + '(' + "{}".format(cat) + ')' )
        print("cat")
    else:
        #plt.xlabel(filename + '(' + "{}".format(dog) + ')' )
        print("dog")


#np.argmax(p)
filename="../input/dogss-test/images.jpg"
img = load_img(filename, target_size=(100, 100))
plt.imshow(img)
plt.show()
img = img_to_array(img)

p = rmodel.predict(img.reshape(1,30000))
if(np.argmax(p)==0):
    print("cat")
else:
    print("dog")
#CNN
from keras.models import load_model
model = load_model('./catsvsdogs.h5')
#filename="../input/dogss-test/images.jpg"
filename="../input/catsdogs/c3.jpg"
result=predict_images(filename)
if(result==0.):
    print("cat")
else :
    print("dog")
filename="../input/catsdogs/c6.jpg"
result=predict_images(filename)
if(result==0.):
    print("cat")
else :
    print("dog")