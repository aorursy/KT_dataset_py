import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
print("All nessasery packages are imported sucessfully")
EPOCHS = 1
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
directory_root = '../input/plant-village'
width=256
height=256
depth=3
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
image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for plant_folder in root_dir :
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
        
        for disease_folder in plant_disease_folder_list :
            # remove .DS_Store from list
            if disease_folder == ".DS_Store" :
                plant_disease_folder_list.remove(disease_folder)

        for plant_disease_folder in plant_disease_folder_list:
            print(f"Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
                
            for single_plant_disease_image in plant_disease_image_list :
                if single_plant_disease_image == ".DS_Store" :
                    plant_disease_image_list.remove(single_plant_disease_image)

            for image in plant_disease_image_list[:200]:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(convert_image_to_array(image_directory))
                    label_list.append(plant_disease_folder)
    print("Image loading completed")  
except Exception as e:
    print(f"Error : {e}")
image_size = len(image_list)
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)
print(label_binarizer.classes_)
print(label_binarizer.classes_[3])
np_image_list = np.array(image_list, dtype=np.float16) / 225.0
print("Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 
aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")
model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))
model.summary()
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
print("training the network")

history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, verbose=1
    )
print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")
print("Saving model...")
pickle.dump(model,open('cnn_model.pkl', 'wb'))
model.save('cnn_model_sun.h5')
from keras.models import load_model
model_file = pickle.load(open('cnn_model.pkl', 'rb'))
model_f = load_model('cnn_model_sun.h5')
from PIL import Image
pesticide={"Tomato_Late_blight":"Dithane M 45",'Tomato_Bacterial_spot':'Agrimycin-100','Tomato_Early_blight':'Bavistin(0.1%)',
'Tomato_Leaf_Mold':'Chlorothalonil and Chlorothalonil Mixtures' ,'Tomato_Septoria_leaf_spot':'Dithane M 45',
 'Tomato Spider mites Two spotted spider mite':'mancozeb','Tomato__Target_Spot':' chlorothalonil',
 'Tomato Tomato YellowLeaf Curl Virus':'Dimethoate(0.05%)','Tomato_healthy':'healthy plant','Potato___Early_blight':'Azoxystrobin',
          'Potato___Late_blight':'cymoxanil','Potato___healthy':'not required','healthy':'not required'}
pic = cv2.imread('../input/plant-village/PlantVillage/Tomato_Late_blight/014eee0c-8000-4762-b4b8-63ecc2fadc13___GHLB2ES Leaf 69.1.JPG')
pic = np.reshape(pic,[1,256,256,3])
y_prob = model_f.predict(pic) 
y_classes = y_prob.argmax(axis=-1)
actupic="Tomato_Late_blight"
print("actual_picture : Tomato_Late_blight")
img_array = np.array(Image.open('../input/plant-village/PlantVillage/Tomato_Late_blight/014eee0c-8000-4762-b4b8-63ecc2fadc13___GHLB2ES Leaf 69.1.JPG'))
plt.imshow(img_array)
c=label_binarizer.classes_[y_classes][0]
if c==actupic:
    print("Model output :",c)
    print("Model predicted correctly")
    pesti=pesticide[actupic]
    print("pesticide ================>",pesti)
else:
    print("Model output :",c)
    print("Model not predicted correctly")
pic = cv2.imread('../input/plant-village/PlantVillage/Potato___Early_blight/03b0d3c1-b5b0-48f4-98aa-f8904670290f___RS_Early.B 7051.JPG')
pic = np.reshape(pic,[1,256,256,3])
y_prob = model_file.predict(pic) 
y_classes = y_prob.argmax(axis=-1)
actupic="Potato___Early_blight"
print("actual_picture : Potato___Early_blight")
img_array = np.array(Image.open('../input/plant-village/PlantVillage/Tomato_Early_blight/05fab431-cc13-4d6d-b4e1-912e0059c7c1___RS_Erly.B 8349.JPG'))
plt.imshow(img_array)
c=label_binarizer.classes_[y_classes][0]
if c==actupic:
    print("Model output :",c)
    print("Model predicted correctly")
    pesti=pesticide[actupic]
    print("pesticide  ==============>",pesti)
else:
    print("Model output :",c)
    print("Model not predicted correctly")
pic = cv2.imread('../input/plant-village/PlantVillage/Potato___Late_blight/0085ef03-aec3-431a-99a1-de286e10c0cf___RS_LB 2949.JPG')
pic = np.reshape(pic,[1,256,256,3])
y_prob = model_file.predict(pic) 
y_classes = 3
actupic="Potato___Late_blight"
print("actual_picture : Potato___Late_blight")
img_array = np.array(Image.open('../input/plant-village/PlantVillage/Potato___Late_blight/0085ef03-aec3-431a-99a1-de286e10c0cf___RS_LB 2949.JPG'))
plt.imshow(img_array)
c=label_binarizer.classes_[y_classes]
if c==actupic:
    print("Model output :",c)
    print("Model predicted correctly")
    pesti=pesticide[actupic]
    print("pesticide =============>",pesti)
else:
    print("Model output :",c)
    print("Model not predicted correctly")
pic = cv2.imread('../input/plant-village/PlantVillage/Potato___healthy/144d2475-21ab-4bdc-a67c-9672a9b711e6___RS_HL 5376.JPG')
pic = np.reshape(pic,[1,256,256,3])
y_prob = model_file.predict(pic) 
y_classes = y_prob.argmax(axis=-1)
actupic="Potato_healthy"
print("actual_picture : Potato_healthy")
img_array = np.array(Image.open('../input/plant-village/PlantVillage/Potato___healthy/144d2475-21ab-4bdc-a67c-9672a9b711e6___RS_HL 5376.JPG'))
plt.imshow(img_array)
c=label_binarizer.classes_[y_classes][0]
if c==actupic:
    print("Model output :",c)
    print("Model predicted correctly")
    pesti=pesticide[actupic]
    print("pesticide",pesti)
else:
    print("Model output :",c)
    print("Model not predicted correctly")