# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
main_dir = "../input/"

train_dir = "dogs-vs-cats/"

path = os.path.join(main_dir,train_dir)

import zipfile



import zipfile

with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/train.zip", 'r') as zip_ref:

    zip_ref.extractall("./kaggle/input/data/train/")
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/test1.zip", 'r') as zip_ref:

    zip_ref.extractall("./kaggle/input/data/test/")
import cv2

import matplotlib.pyplot as plt



train_data_path = "./kaggle/input/data/train/train/"



for f in os.listdir(train_data_path):

    # Load an color image in grayscale

    print(f)

    img = cv2.imread(train_data_path + str(f),cv2.IMREAD_GRAYSCALE)

    resized_img = cv2.resize(img, dsize =(80,80))

    plt.imshow(resized_img)

    break
#label the train data form the file names

X_train_orig = []

y_train_orig = []



train_data_path = "./kaggle/input/data/train/train/"



for f in os.listdir(train_data_path):

    y_train_orig.append(int(f.split('.')[0] == "dog"))

    img = cv2.imread(train_data_path+str(f), cv2.IMREAD_GRAYSCALE)

    X_train_orig.append(cv2.resize(img, dsize=(80,80)))

    

print(len(X_train_orig))

print(len(y_train_orig))
unique_elements, counts_elements = np.unique(np.array(y_train_orig), return_counts=True)

print(counts_elements)



plt.xticks(unique_elements, ("Cat", "Dog"))

plt.bar(unique_elements, counts_elements)

plt.title("No. of cat and dog examples in train dataset")

plt.show()
# resize and normalize X for train



X_train_orig = np.reshape(X_train_orig, (-1, 80,80,1))

print(X_train_orig.shape)



X_train = X_train_orig/255.0
#divide X_train into train cv datasets

np.random.seed(1)



#shuffle the indexes for diversity

shuffled_indexes = np.arange(X_train.shape[0])

np.random.shuffle(shuffled_indexes)



X_train = np.array(X_train)[shuffled_indexes]



y_train = np.array(y_train_orig)[shuffled_indexes]





X_train = X_train[:int(len(X_train)*0.8)]



X_cv = X_train[int(len(X_train)*0.8):]



y_train = y_train[:int(len(y_train)*0.8)]



y_cv = y_train[int(len(y_train)*0.8):]



print(len(X_train))

print(len(X_cv))

print(len(y_train))

print(len(y_cv))
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(128, (3,3), activation="relu", input_shape=(80, 80, 1)),

    tf.keras.layers.MaxPool2D((3,3)),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),

    tf.keras.layers.MaxPool2D((3,3)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=18, activation="relu"),

    tf.keras.layers.Dense(units=9, activation="relu"),

    tf.keras.layers.Dense(units=2, activation="softmax")

])



model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# callback to stop the training if accuracy reaches above 99.8



class Acc_call_back(tf.keras.callbacks.Callback):

    def on_epoch_finished(self, epoch, logs={}):

        if logs.get("accuracy") > 0.998:

            print("Accuracy reached more than 99.8%, stopping training")

            self.model.stop_training = True

    
cb = Acc_call_back()

model.fit(X_train, y_train, epochs=25, callbacks = [cb])
model.evaluate(X_cv, y_cv)
print(X_cv.shape)
preds = model.predict(X_cv)



print(preds[0])
#read test data



test_data_path = "./kaggle/input/data/test/test1/"



X_test_orig = []



for f in os.listdir(test_data_path):

    img = cv2.imread(test_data_path+str(f), cv2.IMREAD_GRAYSCALE)

    X_test_orig.append(cv2.resize(img, dsize=(80,80)))



print(len(X_test_orig))
# Nrmalize and reshape test data.

X_test = np.reshape(X_test_orig,(-1, 80,80,1))



X_test = X_test/255.0

predictions = model.predict(X_test)
result = pd.DataFrame(np.arange(1,len(predictions)+1))



result.columns = ["index"]



result["label"] = np.argmax(predictions,axis=1)



result.to_csv("./kaggle/input/data/result.csv",index=False)

                      

                      