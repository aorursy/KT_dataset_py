import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import glob
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import Callback

data = []
labels = []
size = (30,30)
classes = len(glob.glob('../input/gtsrb-german-traffic-sign/Meta/*.png'))
for i in range(classes):
    path = "../input/gtsrb-german-traffic-sign/Train/" +str(i)
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '/'+ a)
            image = image.resize(size)
            image = np.array(image) /255.0
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")
data = np.array(data)
labels = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
#building the model
import keras
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu',))
model.add(BatchNormalization())
model.add(Dropout(rate=0.7))
model.add(Dense(classes, activation='softmax'))
#Compilation of the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
class MyCallback(Callback):

    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy') > 0.9995  and logs.get('accuracy') > 0.995) :
                print("val_accurcy more than 99.95% and accuracu more than 99.5%")
                self.model.stop_training = True
mcallback = MyCallback()

history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), callbacks = [mcallback])
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
test_set = pd.read_csv('/kaggle/input/gtsrb-german-traffic-sign/Test.csv')
labels = test_set["ClassId"].values
imgs = test_set["Path"].values
test_data=[]
for img in imgs:
    image = Image.open('/kaggle/input/gtsrb-german-traffic-sign/'+img)
    image = image.resize(size)
    test_data.append(np.array(image)/255.0)
test=np.array(test_data)
pred = model.predict_classes(test)
#Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))
#model.save("simple_traffic_signs_classifier.h5")
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy

classifier = load_model('simple_traffic_signs_classifier.h5')

signs = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }

for filename in glob.glob('../input/testing/*'):
    img = Image.open(filename)
    img = img.resize((30,30))
    plt.imshow(img)
    plt.show()
    img = numpy.array(img)
    img = img[...,:3]
    img = img/255.0
    img = numpy.expand_dims(img, axis=0)
    
    print('your image shape should be (1, 30, 30, 3)',img.shape)
    pred = classifier.predict_classes([img])[0]
    sign = signs[pred+1]
    print(sign)


