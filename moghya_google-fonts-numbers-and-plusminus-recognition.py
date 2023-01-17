import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
TRAIN_CSV_FILE = "../input/numbers-and-plusminus-in-google-fonts/dataset.csv"
TRAIN_IMAGES_PATH = "../input/numbers-and-plusminus-in-google-fonts/images/dataset/"
TEST_CSV_FILE = "../input/test-data-numbers-and-plusminus-in-google-fonts/test_dataset.csv"
TEST_IMAGES_PATH = "../input/test-data-numbers-and-plusminus-in-google-fonts/testimages/testimages/"
print('No of train images:',len(os.listdir(TRAIN_IMAGES_PATH)))
print('No of test images:',len(os.listdir(TEST_IMAGES_PATH)))
from PIL import Image

def class_of(char):
    if char=='+':
        return 10
    elif char=='-':
        return 11
    else:
        return int(char)

def PIL_to_array(img):
    arr = np.array(img.getdata(),np.float32)/255
    arr = arr.reshape(92,142)
    return arr


def convert_to_vector_data(csv_file,image_dataset_path):
    inp = []
    out = []
    f = open(csv_file,'r')
    rows = f.readlines()
    for row in rows:
        row = row.strip(' \n ').split(',')
        file_name = row[0]
        char = row[1]
        img = Image.open(image_dataset_path+file_name).convert('L')
        arr = PIL_to_array(img)
        inp.append(arr)
        out.append(class_of(char))
    f.close()
    return inp,out
X_train,y_train = convert_to_vector_data(TRAIN_CSV_FILE,TRAIN_IMAGES_PATH)
X_test,y_test = convert_to_vector_data(TEST_CSV_FILE,TEST_IMAGES_PATH)
X_train,y_train = np.array(X_train),np.array(y_train)
X_test,y_test = np.array(X_test),np.array(y_test)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
np.unique(y_test)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, pooling
from keras.optimizers import Adam
from keras.utils import np_utils
X_train = X_train.reshape(X_train.shape[0],92, 142,1)
X_test = X_test.reshape(X_test.shape[0],92,142,1)
Y_train = np_utils.to_categorical(y_train, 12)
Y_test = np_utils.to_categorical(y_test, 12)
Y_train.shape
X_train.shape
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(92,142,1)))
model.add(pooling.MaxPooling2D(pool_size=(2,2)))
print(model.output_shape)
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
print(model.output_shape)
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
print(model.output_shape)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))
print(model.output_shape)
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)
test_predictions = model.predict_classes(X_test)
test_predictions.shape
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,test_predictions))
train_predictions = model.predict_classes(X_train)
print(accuracy_score(y_train,train_predictions))
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
print(os.listdir('.'))
