import numpy as np
import pickle
import cv2
import pandas as pd
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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
EPOCHS = 30
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((28, 28))
image_size = 0
directory_root = '../input/train/train'
width= 28
height= 28
depth= 1 # the number of channels in our input images. 1 for grayscale single channel images, 3  for standard RGB images
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir, 0)
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
    print("[INFO] Loading Images")
    root_dir = listdir(directory_root)
    for directory in root_dir:
        print(f"[INFO] processing {directory}")
        directory_content = listdir(f"{directory_root}/{directory}")
        for image in directory_content:
            image_list.append(convert_image_to_array(f"{directory_root}/{directory}/{image}"))
            label_list.append(directory)
    print("[INFO] Image loading complete")
except Exception as e:
    print(e)
image_size = len(image_list)
print(f"[INFO] Image size = {image_size}")
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
n_classes = len(label_binarizer.classes_)
print(label_binarizer.classes_)
np_image_list = np.array(image_list, dtype=np.float16) / 225.0
print(np_image_list.shape)
print(image_labels.shape)
print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 
aug = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")
model = Sequential()
inputShape = (height, width, depth)
# if we are using "channels first", update the input shape
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
# first set of CONV => RELU => POOL layers
model.add(Conv2D(20, (5, 5), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# second set of CONV => RELU => POOL layers
model.add(Conv2D(50, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))
# softmax classifier
model.add(Dense(n_classes))
model.add(Activation("softmax"))
model.summary()
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the network
print("[INFO] training network...")
print(len(x_train))
history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS
    )
acc = history.history['acc']
val_acc = history.history['val_acc']
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
print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Accuracy: {scores[1]*100} Loss: {scores[0]}")
print("CNN Error: %.2f%%" % (100- (scores[1]*100) ) )
import numpy as np
y_pred = model.predict(x_test)
acc = sum([np.argmax(y_test[i])==np.argmax(y_pred[i]) for i in range(10000)])/10000
print(acc*100)
test_image_list, test_image_fileName_list = [], []
try:
    print("[INFO] Loading Test Images")
    test_image_directory = '../input/test/test'
    root_dir = listdir(test_image_directory)
    for image in root_dir:
        test_image_list.append(convert_image_to_array(f"{test_image_directory}/{image}"))
        test_image_fileName_list.append(image)
    print("[INFO] Image loading complete")
except Exception as e:
    print(e)
np_test_image_list = np.array(test_image_list, dtype=np.float16) / 225.0
print(f"[INFO] Test Image Shape : {np_test_image_list.shape}")
predictions = []
for arr in np_test_image_list:
    arr = np.expand_dims(arr, axis=0)
    prediction_output = label_binarizer.inverse_transform(model.predict(arr))
    predictions.append(prediction_output[0])
print("[INFO] Creating pandas dataframe")
submission_data = {"ImageID":test_image_fileName_list,"Category":predictions}
submission_data_frame = pd.DataFrame(submission_data)
submission_data_frame
print("[INFO] Saving Predicition to CSV")
submission_data_frame.to_csv('sample-submission.csv',columns=["ImageID","Category"], index = False)