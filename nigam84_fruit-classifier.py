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
#importing useful libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dropout,Dense,Flatten,MaxPooling2D
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import cv2
from keras.utils import np_utils
from sklearn.utils import shuffle
import glob
import pandas as pd
import os
import csv
# As the labels in ImageDataGenerator has numeric values so for prediction result we need name of classes of 
# fruit and vegetable which will be easily understood by user

# initial dictionary for converting categories to numerical values

initial_dic={
"apple braeburn" : 0 ,
"apple crimson snow" : 1 ,
"apple golden 1" : 2 ,
"apple golden 2" : 3 ,
"apple golden 3" : 4 ,
"apple granny smith" : 5 ,
"apple pink lady" : 6 ,
"apple red 1" : 7 ,
"apple red 2" : 8 ,
"apple red 3" : 9 ,
"apple red delicious" : 10 ,
"apple red yellow 1" : 11 ,
"apple red yellow 2" : 12 ,
"apricot" : 13 ,
"avocado" : 14 ,
"avocado ripe" : 15 ,
"banana" : 16 ,
"banana lady finger" : 17 ,
"banana red" : 18 ,
"beetroot" : 19 ,
"blueberry" : 20 ,
"cactus fruit" : 21 ,
"cantaloupe 1" : 22 ,
"cantaloupe 2" : 23 ,
"carambula" : 24 ,
"cauliflower" : 25 ,
"cherry 1" : 26 ,
"cherry 2" : 27 ,
"cherry rainier" : 28 ,
"cherry wax black" : 29 ,
"cherry wax red" : 30 ,
"cherry wax yellow" : 31 ,
"chestnut" : 32 ,
"clementine" : 33 ,
"cocos" : 34 ,
"dates" : 35 ,
"eggplant" : 36 ,
"ginger root" : 37 ,
"granadilla" : 38 ,
"grape blue" : 39 ,
"grape pink" : 40 ,
"grape white" : 41 ,
"grape white 2" : 42 ,
"grape white 3" : 43 ,
"grape white 4" : 44 ,
"grapefruit pink" : 45 ,
"grapefruit white" : 46 ,
"guava" : 47 ,
"hazelnut" : 48 ,
"huckleberry" : 49 ,
"kaki" : 50 ,
"kiwi" : 51 ,
"kohlrabi" : 52 ,
"kumquats" : 53 ,
"lemon" : 54 ,
"lemon meyer" : 55 ,
"limes" : 56 ,
"lychee" : 57 ,
"mandarine" : 58 ,
"mango" : 59 ,
"mango red" : 60 ,
"mangostan" : 61 ,
"maracuja" : 62 ,
"melon piel de sapo" : 63 ,
"mulberry" : 64 ,
"nectarine" : 65 ,
"nectarine flat" : 66 ,
"nut forest" : 67 ,
"nut pecan" : 68 ,
"onion red" : 69 ,
"onion red peeled" : 70 ,
"onion white" : 71 ,
"orange" : 72 ,
"papaya" : 73 ,
"passion fruit" : 74 ,
"peach" : 75 ,
"peach 2" : 76 ,
"peach flat" : 77 ,
"pear" : 78 ,
"pear abate" : 79 ,
"pear forelle" : 80 ,
"pear kaiser" : 81 ,
"pear monster" : 82 ,
"pear red" : 83 ,
"pear williams" : 84 ,
"pepino" : 85 ,
"pepper green" : 86 ,
"pepper red" : 87 ,
"pepper yellow" : 88 ,
"physalis" : 89 ,
"physalis with husk" : 90 ,
"pineapple" : 91 ,
"pineapple mini" : 92 ,
"pitahaya red" : 93 ,
"plum" : 94 ,
"plum 2" : 95 ,
"plum 3" : 96 ,
"pomegranate" : 97 ,
"pomelo sweetie" : 98 ,
"potato red" : 99 ,
"potato red washed" : 100 ,
"potato sweet" : 101 ,
"potato white" : 102 ,
"quince" : 103 ,
"rambutan" : 104 ,
"raspberry" : 105 ,
"redcurrant" : 106 ,
"salak" : 107 ,
"strawberry" : 108 ,
"strawberry wedge" : 109 ,
"tamarillo" : 110 ,
"tangelo" : 111 ,
"tomato 1" : 112 ,
"tomato 2" : 113 ,
"tomato 3" : 114 ,
"tomato 4" : 115 ,
"tomato cherry red" : 116 ,
"tomato maroon" : 117 ,
"tomato yellow" : 118 ,
"walnut" : 119 ,
}



# result dictionary which converts result to category of fruits

result_dic={
0 : "apple braeburn",
1 : "apple crimson snow",
2 : "apple golden 1",
3 : "apple golden 2",
4 : "apple golden 3",
5 : "apple granny smith",
6 : "apple pink lady",
7 : "apple red 1",
8 : "apple red 2",
9 : "apple red 3",
10 : "apple red delicious",
11 : "apple red yellow 1",
12 : "apple red yellow 2",
13 : "apricot",
14 : "avocado",
15 : "avocado ripe",
16 : "banana",
17 : "banana lady finger",
18 : "banana red",
19 : "beetroot",
20 : "blueberry",
21 : "cactus fruit",
22 : "cantaloupe 1",
23 : "cantaloupe 2",
24 : "carambula",
25 : "cauliflower",
26 : "cherry 1",
27 : "cherry 2",
28 : "cherry rainier",
29 : "cherry wax black",
30 : "cherry wax red",
31 : "cherry wax yellow",
32 : "chestnut",
33 : "clementine",
34 : "cocos",
35 : "dates",
36 : "eggplant",
37 : "ginger root",
38 : "granadilla",
39 : "grape blue",
40 : "grape pink",
41 : "grape white",
42 : "grape white 2",
43 : "grape white 3",
44 : "grape white 4",
45 : "grapefruit pink",
46 : "grapefruit white",
47 : "guava",
48 : "hazelnut",
49 : "huckleberry",
50 : "kaki",
51 : "kiwi",
52 : "kohlrabi",
53 : "kumquats",
54 : "lemon",
55 : "lemon meyer",
56 : "limes",
57 : "lychee",
58 : "mandarine",
59 : "mango",
60 : "mango red",
61 : "mangostan",
62 : "maracuja",
63 : "melon piel de sapo",
64 : "mulberry",
65 : "nectarine",
66 : "nectarine flat",
67 : "nut forest",
68 : "nut pecan",
69 : "onion red",
70 : "onion red peeled",
71 : "onion white",
72 : "orange",
73 : "papaya",
74 : "passion fruit",
75 : "peach",
76 : "peach 2",
77 : "peach flat",
78 : "pear",
79 : "pear abate",
80 : "pear forelle",
81 : "pear kaiser",
82 : "pear monster",
83 : "pear red",
84 : "pear williams",
85 : "pepino",
86 : "pepper green",
87 : "pepper red",
88 : "pepper yellow",
89 : "physalis",
90 : "physalis with husk",
91 : "pineapple",
92 : "pineapple mini",
93 : "pitahaya red",
94 : "plum",
95 : "plum 2",
96 : "plum 3",
97 : "pomegranate",
98 : "pomelo sweetie",
99 : "potato red",
100 : "potato red washed",
101 : "potato sweet",
102 : "potato white",
103 : "quince",
104 : "rambutan",
105 : "raspberry",
106 : "redcurrant",
107 : "salak",
108 : "strawberry",
109 : "strawberry wedge",
110 : "tamarillo",
111 : "tangelo",
112 : "tomato 1",
113 : "tomato 2",
114 : "tomato 3",
115 : "tomato 4",
116 : "tomato cherry red",
117 : "tomato maroon",
118 : "tomato yellow",
119 : "walnut"
}
# using ImageDataGenerator for data augmentation and loading images in batches in memory
'''PS. Tried loading all data at once but memory limit exceeded'''

train_dir = '/kaggle/input/fruits/fruits-360/Training/'

# Performing scaling and rotation etc on training
generator = ImageDataGenerator(rotation_range=40,
        rescale=1./255,
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2)


train_generator = generator.flow_from_directory(
    directory=r"/kaggle/input/fruits/fruits-360/Training/",
    target_size=(100, 100),
    color_mode="rgb",
    batch_size=16,
    class_mode="categorical",
    shuffle=True,
    seed=0
)


# Performing scaling and rotation etc on training
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = generator.flow_from_directory(
    directory=r"/kaggle/input/fruits/fruits-360/Test/",
    target_size=(100, 100),
    color_mode="rgb",
    batch_size=16,
    class_mode="categorical",
    shuffle=True,
    seed=0
)
# creating the model
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(100,100,3)))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=16,kernel_size=(3,3),activation="relu"))
model.add(Conv2D(filters=16,kernel_size=(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256,activation="relu"))
model.add(Dropout(0.25))

model.add(Dense(120,activation="softmax"))
model.summary()
checkpoint = ModelCheckpoint("fruits_classifier.h5", monitor = 'val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(), metrics = ['acc'])
history = model.fit_generator(train_generator, 
                              epochs=10, 
                              shuffle=False, 
                              validation_data=test_generator,
                              callbacks=[checkpoint]
                              )
# Plots of Training and validation accuracy/loss during training of model
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plt.figure()
plt.plot(epochs, acc, 'b', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('Accuracy.jpg')
plt.figure()
plt.plot(epochs, loss, 'b', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('Loss.jpg')
# Showing image and label from training set
x,y = train_generator.next()
for i in range(0,1):
    image = x[i]
    plt.imshow(image)
    plt.title(np.argmax(y[i]))
    plt.show()
# Test prediction of a batch

x_test,y_test=test_generator.next()
y_pred = model.predict(x_test)

results_pred = y_pred.argmax(axis=1)
results_test = y_test.argmax(axis=1)


# plotting the test fruits in the batch of 16 with thier predictions 
fig = plt.figure(figsize=(16, 9))
for i in range(5):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(x_test[i])
    label_pred = results_pred[i]
    label_true = results_test[i]
    ax.set_title("{} ({})".format(label_pred, label_true))
# Test prediction of a batch with name labels

x_test,y_test=test_generator.next()
y_pred = model.predict(x_test)

results_pred = y_pred.argmax(axis=1)
results_test = y_test.argmax(axis=1)


# plotting the test fruits in the batch of 16 with thier predictions 
fig = plt.figure(figsize=(16, 9))
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(x_test[i])
    label_pred = result_dic[results_pred[i]]
    label_true = result_dic[results_test[i]]
    ax.set_title("{} ({})".format(label_pred, label_true))
# serialize model to JSON
model_json = model.to_json()
with open("fruit_clf.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("fruit_clf.h5")
print("Saved model to disk")
# load json and create model
from keras.models import model_from_json

json_file = open('/kaggle/working/fruit_clf.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
fruit_classifier = model_from_json(loaded_model_json)
# load weights into new model
fruit_classifier.load_weights("/kaggle/working/fruit_clf.h5")
print("Loaded model from disk")
# Test prediction of a batch with name labels from loaded model

x_test,y_test=test_generator.next()
y_pred = fruit_classifier.predict(x_test)

results_pred = y_pred.argmax(axis=1)
results_test = y_test.argmax(axis=1)


# plotting the test fruits in the batch of 16 with thier predictions 
fig = plt.figure(figsize=(16, 9))
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(x_test[i])
    label_pred = result_dic[results_pred[i]]
    label_true = result_dic[results_test[i]]
    ax.set_title("{} ({})".format(label_pred, label_true))
