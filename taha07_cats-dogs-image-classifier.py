# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import random
from tensorflow.keras.preprocessing import image


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.listdir("/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set/")
print("Number of Cats images in Training Directory {}".format(len(os.listdir("/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set/cats"))))
print("Number of Dogs images in Training Directory {}".format(len(os.listdir("/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set/dogs"))))

print("Number of images in Training Directory {}".format(len(os.listdir("/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set/cats")) + len(os.listdir("/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set/dogs"))))
from tensorflow.keras.preprocessing.image import load_img
cat_dir = "/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set/cats"
dog_dir = "/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set/dogs"
fig = plt.figure(figsize=(12,9))
fig.set_size_inches(15,15)
for i in range(5):
    plt.subplot(1,5,i+1)
    cat_img = os.listdir(cat_dir)
    sample = random.choice(cat_img)
    img = load_img(cat_dir +"/" +sample)
    plt.imshow(img)
    plt.xlabel("Cat")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()

fig = plt.figure(figsize=(12,8))
fig.set_size_inches(15,15)

for i in range(5):
    plt.subplot(1,5,i+1)
    dog_img = os.listdir(dog_dir)
    sample = random.choice(dog_img)
    img = load_img(dog_dir +"/" +sample)
    plt.imshow(img)
    plt.xlabel("Dog")
    plt.xticks([])
    plt.yticks([])
    
plt.tight_layout()
    
labels = ["cats","dogs"]
img_size = 120

def get_data(data_dir):
    data = []
    for label in labels:
        category = labels.index(label)
        path = os.path.join(data_dir,label)
        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path,img))
            img_arr_resize = cv2.resize(img_arr,(img_size,img_size))
            data.append([img_arr_resize,category])
    return np.array(data)



train = get_data("/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set/")
test =   get_data("/kaggle/input/dogs-cats-images/dog vs cat/dataset/test_set/")
x_train = []
y_train =[]

x_test = []
y_test = []   

for feature,label in train:
    x_train.append(feature)
    y_train.append(label)

for feature,label in test:
    x_test.append(feature)
    y_test.append(label)

plt.figure(figsize=(10,5))
plt.style.use("ggplot")
plt.subplot(1,2,1)
sns.countplot(y_train)
plt.title("Train")

plt.subplot(1,2,2)
sns.countplot(y_test)
plt.title("Test")

plt.show()
x_train = np.array(x_train)/255
x_test = np.array(x_test)/255

x_train = x_train.reshape(-1,img_size,img_size,3)
y_train = np.array(y_train)

x_test = x_test.reshape(-1,img_size,img_size,3)
y_test = np.array(y_test)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=45,width_shift_range = 0.2,height_shift_range=0.2,
                            zoom_range = 0.2,shear_range = 0.2,horizontal_flip= True)
datagen.fit(x_train)
from tensorflow.keras.applications.vgg16 import VGG16

vgg = VGG16(weights = "imagenet",include_top=False,input_shape = (120,120,3))
for layer in vgg.layers:
    layer.trainable = False
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Model

x = vgg.output
x = Flatten()(x)
prediction = Dense(1,activation = "sigmoid")(x)
model = Model(inputs = vgg.input,outputs = prediction)
model.summary()
from tensorflow.keras.optimizers import SGD
model.compile(optimizer = SGD(lr =0.0001,momentum = 0.9),loss='binary_crossentropy',metrics=['accuracy'])
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
checkpoint = ModelCheckpoint("cats_dogs.h5",monitor="val_accuracy",verbose=1,save_best_only=True,
                             save_weights_only=False,mode="auto",period = 1)
earlyStop = EarlyStopping(monitor = "val_acc",patience=10,verbose=1)

history = model.fit(datagen.flow(x_train,y_train,batch_size=32),epochs=40,
                    validation_data = datagen.flow(x_test,y_test),callbacks = [checkpoint,earlyStop])
print("Loss of the Model :{}".format(model.evaluate(x_test,y_test)[0]))
print("Accuracy of the Model :{}%".format(model.evaluate(x_test,y_test)[1]*100))
plt.style.use("ggplot")
fig = plt.figure(figsize=(12,6))
epochs = range(1,41)
plt.subplot(1,2,1)
plt.plot(epochs,history.history["accuracy"],"go-")
plt.plot(epochs,history.history["val_accuracy"],"ro-")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train","val"],loc = "upper left")
#plt.show()

#fig = plt.figure(figsize=(12,8))    
plt.subplot(1,2,2)
plt.plot(epochs,history.history["loss"],"go-")
plt.plot(epochs,history.history["val_loss"],"ro-")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train","val"],loc = "upper left")
plt.show()

prediction = []
y_pred = model.predict(x_test)

for i in y_pred:
    if i < 0.5:
        prediction.append(0)
    else:
        prediction.append(1)
from sklearn.metrics import classification_report
print(classification_report(y_test, prediction, target_names = ['Cats (Class 0)','Dogs (Class 1)']))
from mlxtend.plotting  import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,prediction)
plot_confusion_matrix(conf_mat = cm,figsize=(8,8),class_names = labels)
correct = np.nonzero(prediction == y_test)[0]
incorrect = np.nonzero(prediction != y_test)[0]
plt.figure(figsize=(12,8))
i = 0
for c in correct[:4]:
    plt.subplot(1,4,i+1)
    plt.imshow(x_test[c])
    plt.xlabel("Predicted Class {} \n Actual Class {}".format(prediction[c], y_test[c]))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    i += 1
plt.figure(figsize=(12,8))
i = 0
for c in incorrect[:4]:
    plt.subplot(1,4,i+1)
    plt.imshow(x_test[c])
    plt.xlabel("Predicted Class {} \n Actual Class {}".format(prediction[c], y_test[c]))
    plt.xticks([])
    plt.yticks([]) 
    plt.tight_layout()
    i += 1
