import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
DATASET_DIR = os.listdir("../input/flowers-recognition/flowers/flowers") 
DATASET_DIR
labels = ['daisy', 'rose', 'dandelion', 'sunflower', 'tulip'] #labels
DIR = "../input/flowers-recognition/flowers/flowers/" # path
link = []
for label in labels:
    path = os.path.join(DIR, label) # combine path and labels
    link.append(path) # append in link
print(link)


for i in range(len(link)):
    new = os.listdir(link[i])
    i+=1
    print(f"length : {len(new)}") # each folder total image count 
IMG_SIZE = 224 # image size

data = [] 

def get_data(data_dir):
    for category in labels:
        path = os.path.join(data_dir, category) #combine path
        class_num = labels.index(category) # index no of labels
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR) # color image array
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # resize
                data.append([resized_array, class_num])
            except Exception as e: # exception
                print(e)
    return np.array(data) # return array
new_data = get_data("/kaggle/input/flowers-recognition/flowers/flowers/") # path for function
# visualize each class
import seaborn as sns

l = []
for i in new_data:
    l.append(labels[i[1]])
sns.set_style('whitegrid')
countplot = sns.countplot(l)

for p in countplot.patches:
    countplot.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.45, p.get_height()+0.1), ha='center', va='bottom', color= 'black') # show the count no.
import matplotlib.pyplot as plt # pip install matplotlib
import random
random.shuffle(new_data) # shuffle data
# random image visualization
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    r = random.randint(0, len(new_data))
    img = (new_data[r][0])
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.title(f"Flower: {labels[new_data[r][1]]}")
    fig.tight_layout()
plt.show()
# split data
from sklearn.model_selection import train_test_split
# separate item and label
X = []
y = []
for item, label in new_data:
    X.append(item)
    y.append(label)
X = np.array(X) / 255 # Normaliation( Now the array will remain 0-1)
y = np.array(y)
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 3) # Reshape array with channel 3
from sklearn.preprocessing import LabelBinarizer # LabelBinarizer
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(y)

# if rose, then it will be 0 1 0 0 0
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) # trainning and testing data
import tensorflow as tf 
from tensorflow.keras.layers import Flatten, Conv2D, Activation, Dense, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(2, 2, padding="same"))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(2, 2, padding="same"))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(2, 2, padding="same"))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(5, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=32) # training part with 10 epochs
score = model.evaluate(x_test, y_test)
model.save("cnn.model") # save model
# ploting 
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, "b", label="trainning accuracy")
plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
plt.legend()
plt.show()

plt.plot(epochs, loss, "b", label="trainning loss")
plt.plot(epochs, val_loss, "r", label="validation loss")
plt.legend()
plt.show()
predictions = model.predict_classes(x_test)
predictions[:10] # predict 10 images
y_test_inv_label = label_binarizer.inverse_transform(y_test)
# predict function
labels = ['daisy', 'rose', 'dandelion', 'sunflower', 'tulip']

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

model = tf.keras.models.load_model("cnn.model")
prediction = model.predict_classes([prepare("../input/prediction-images/rose.jpg")])
print(labels[(prediction[0])])  # predict rose
prediction = model.predict_classes([prepare("../input/prediction-images/daisy.jpg")])
print(labels[(prediction[0])]) # predict daisy
prediction = model.predict_classes([prepare("../input/prediction-images/dandelion1.jpg")])
print(labels[(prediction[0])]) # predict daisy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
confusion_matrix(y_test_inv_label, predictions)
print(classification_report(y_test_inv_label, predictions, target_names=labels))
print(accuracy_score(y_test_inv_label, predictions))
