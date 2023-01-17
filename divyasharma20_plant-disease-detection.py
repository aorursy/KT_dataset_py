# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.utils import plot_model
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
!pip list
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)
train_data_dir =  "../input/plantv/PlantVillage/train"
validation_data_dir ="../input/plantv/PlantVillage/val"
img_width, img_height = 224, 224 
batch = 64
epochs = 2

# input_sizes = {
#     'alexnet' : (224,224),
#     'densenet': (224,224),
#     'resnet' : (224,224),  256, 256
#     'inception' : (299,299),
#     'squeezenet' : (224,224),#not 255,255 acc. to https://github.com/pytorch/pytorch/issues/1120
#     'vgg' : (224,224)
# }
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=10,
    preprocessing_function=preprocess_input
)

val_datagen = ImageDataGenerator(
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=10, 
    preprocessing_function=preprocess_input
    )
train_generator = train_datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(img_width, img_height),
)

val_generator = val_datagen.flow_from_directory(
    directory=validation_data_dir,
    target_size=(img_width, img_height),
)
type(train_generator)
resnet50 = ResNet50(include_top=False, weights="imagenet", input_shape=(img_width, img_height, 3))
resnet50.trainable = False

x = resnet50.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(39, activation="softmax")(x)
from keras.applications.resnet152 import ResNet152
resnet50 = tf.keras.applications.ResNet152(include_top=False, weights="imagenet", input_shape=(img_width, img_height, 3))
resnet50.trainable = False

x = resnet50.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(39, activation="softmax")(x)
from keras.applications.inception_v3 import InceptionV3, preprocess_input
resnet50 = InceptionV3(include_top=False, weights="imagenet", input_shape=(img_width, img_height, 3))
resnet50.trainable = False

x = resnet50.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(39, activation="softmax")(x)
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
resnet50 = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(img_width, img_height, 3))
resnet50.trainable = False

x = resnet50.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(39, activation="softmax")(x)
from keras.applications.densenet import DenseNet169, preprocess_input
resnet50 = DenseNet169(include_top=False, weights="imagenet", input_shape=(224,224,3))
resnet50.trainable = False

x = resnet50.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(39, activation="softmax")(x)
# base_model = InceptionV3(include_top=False, weights="imagenet", input_tensor=Input(shape=(299,299,3)))
# model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
# image_size = (299, 299)
model = Model(resnet50.input, x)
model.summary()
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
from keras.optimizers import Adam, SGD
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) #0.0001
model.compile(optimizer=SGD(lr=0.0001,momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy']) #0.0001
checkpoint = ModelCheckpoint("resnet_2.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
#A callback is a set of functions to be applied at given stages of the training procedure. 
#You can use callbacks to get a view on internal states and statistics of the model during training.
history = model.fit_generator(
    train_generator,
    epochs=30,
    steps_per_epoch=50,
    validation_data=val_generator,
    validation_steps=25,
    callbacks = [checkpoint, early]
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
dic = {}
dic['incep_resnet_v2'] = 93.34
dic['inception_v3'] = 96.75
dic['resnet_50'] = 97.52
dic['densenet_169'] = 95.75
import pandas as pd
df = pd.DataFrame()
df['Architecture'] = list(dic.keys())
df['Accuracy']=list(dic.values())
import seaborn as sn
sn.catplot("Architecture", "Accuracy", data=df, kind="bar")

scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")
import pickle
print("[INFO] Saving model...")
pickle.dump(model,open('DenseNet169_model98.pkl', 'wb'))
import io
import base64
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array

default_image_size = tuple((256, 256))

def convert_image(image_data):
    try:
        image = Image.open(image_data)
        if image is not None :
            image = image.resize(default_image_size, Image.ANTIALIAS)   
            image_array = img_to_array(image)
            return np.expand_dims(image_array, axis=0), None
        else :
            return None, "Error loading image file"
    except Exception as e:
        return None, str(e)
image_path = "../input/plantv/PlantVillage/train/Tomato___Leaf_Mold/3d232a86-283d-45ec-b5cf-99829d78afa8___Crnl_L.Mold 6550.JPG"
image_array, err_msg = convert_image(image_path)
image_array = train_datagen.flow_from_directory(
    directory="../input/plantv/PlantVillage/val/Corn_(maize)___Common_rust_/",
    target_size=(img_width, img_height),
)
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
# from keras.applications.vgg16 import VGG16
# from keras.models import Model

image = load_img("../input/plantv/PlantVillage/train/Tomato___Leaf_Mold/90ceddb7-9fe6-4e76-b33e-d83983cf8ef5___Crnl_L.Mold 6540.JPG", target_size=(224,224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
import pickle
model_file = "resnet50_model.pkl"
model = pickle.load(open(model_file,'rb'))
pred = model.predict(image)
dt = f"{label_binarizer.inverse_transform(pred)[0]}"
dt
np.argmax(pred)
from os import listdir
image_labels = listdir("../input/plantv/PlantVillage/train")
image_labels
'Tomato___Late_blight'.split("___")[1]
for i,x in enumerate(image_labels):
    print(i,"==>",x)
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(image_labels)
# pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)
n_classes
import cv2
default_image_size =  tuple((256, 256))
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
from keras.preprocessing.image import img_to_array
image_list, label_list = [], []
directory_root = '../input/plantv/PlantVillage'
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    print(root_dir)
    for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for plant_folder in root_dir :
        if plant_folder == "val":
            plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")

            for disease_folder in plant_disease_folder_list :
                # remove .DS_Store from list
                if disease_folder == ".DS_Store" :
                    plant_disease_folder_list.remove(disease_folder)

            for plant_disease_folder in plant_disease_folder_list:
                print(f"[INFO] Processing {plant_disease_folder} ...")
                plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")

                for single_plant_disease_image in plant_disease_image_list :
                    if single_plant_disease_image == ".DS_Store" :
                        plant_disease_image_list.remove(single_plant_disease_image)

                for image in plant_disease_image_list[:200]:
                    image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                    if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                        image_list.append(convert_image_to_array(image_directory))
                        label_list.append(plant_disease_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")
np_image_list = np.array(image_list, dtype=np.float16) / 225.0
import pickle
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
image_labels1 = label_binarizer.fit_transform(image_labels)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
list(label_binarizer.classes_)
label_binarizer.inverse_transform(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]))
from sklearn.preprocessing import LabelEncoder
lst = [x for x in image_labels]
lst
from numpy import array
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit_transform(lst) #["image_labels","good","bad"]
pickle.dump(le,open('label_transformEnco2.pkl', 'wb'))
n_classes = len(le.classes_)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.5, random_state = 42) 
model_file = "densnet169more_model.pkl"
model = pickle.load(open(model_file,'rb'))
scores = model.evaluate(x_test, y_test,batch_size=128)
print(f"Test Accuracy: {scores[1]*100}")
#test loss, test acc
scores
from flask import Flask, render_template , redirect, url_for, request
import random
app = Flask(__name__)
app.static_folder = 'static'
@app.route('/')
def homepage():
    return """Hello world!"""

if __name__ == '__main__':
    app.run( debug=True)
import tkinter as tk
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Hello World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def say_hi(self):
        print("hi there, everyone!")

root = tk.Tk()
app = Application(master=root)
app.mainloop()
from flask import Flask, request, jsonify, abort
import traceback
import pandas as pd
import numpy as np
import socket
import pickle
import flask

app = Flask(__name__)
# model = pickle.load(open("model3.pkl","rb"))

@app.route('/api',methods=['GET','POST'])
def predict():
    data = request.get_json(force=True)
    predict_request = [data['Open'], data['Low'], data['High'], data['Adj Close']]
    predict_request = np.array(predict_request)
    y_hat = 0 #model(predict_request)
    output = [y_hat[0]]
    return flask.jsonify(results=response)

if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))
    port = sock.getsockname()[1]
    sock.close()
    app.run(port=port, debug=True)
