import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from sklearn.metrics import confusion_matrix# confusion matrix to carry out error analysis

import seaborn as sn                        # heatmap

from sklearn.utils import shuffle           # shuffle the data

import matplotlib.pyplot as plt             # 2D plotting library

import cv2                                  # image processing library

import tensorflow as tf                     # best library ever

        # Any results you write to the current directory are saved as output.
print(os.listdir("../input/seg_train/seg_train/"))
# Here's our 6 categories that we have to classify.

class_names = ['street', 'buildings', 'mountain', 'sea', 'forest', 'glacier']

class_names_label = {'street': 0,

                    'buildings' : 1,

                    'mountain' : 2,

                    'sea' : 3,

                    'forest' : 4,

                    'glacier' : 5

                    }

nb_classes = 6
def load_data():

    datasets = ['seg_train/seg_train', 'seg_test/seg_test']

    size = (150,150)

    output = []

    for dataset in datasets:

        directory = "../input/" + dataset

        images = []

        labels = []

        for folder in os.listdir(directory):

            curr_label = class_names_label[folder]

            for file in os.listdir(directory + "/" + folder):

                img_path = directory + "/" + folder + "/" + file

                curr_img = cv2.imread(img_path)

                curr_img = cv2.resize(curr_img, size)

                images.append(curr_img)

                labels.append(curr_label)

        images, labels = shuffle(images, labels)     ### Shuffle the data

        images = np.array(images, dtype = 'float32') ### Our images

        labels = np.array(labels, dtype = 'int32')   ### From 0 to num_classes-1

        

        output.append((images, labels))



    return output
(train_images, train_labels), (test_images, test_labels) = load_data()
print ("Number of training examples: " + str(train_labels.shape[0]))

print ("Number of testing examples: " + str(test_labels.shape[0]))

print ("Each image is of size: " + str(train_images.shape[1:]))


sizes = np.bincount(train_labels)

explode = (0, 0, 0, 0, 0, 0)  

plt.pie(sizes, explode=explode, labels=class_names,

autopct='%1.1f%%', shadow=True, startangle=150)

plt.axis('equal')

plt.title('Proportion of each observed category')



plt.show()
train_images = train_images / 255.0 

test_images = test_images / 255.0
index = np.random.randint(train_images.shape[0])

plt.figure()

plt.imshow(train_images[index])

plt.grid(False)

plt.title('Image #{} : '.format(index) + class_names[train_labels[index]])

plt.show()
index = np.random.randint(train_images.shape[0])

plt.figure()

plt.imshow(train_images[index])

plt.grid(False)

plt.title('Image #{} : '.format(index) + class_names[train_labels[index]])

plt.show()
fig = plt.figure(figsize=(10,10))

fig.suptitle("Some examples of images of the dataset", fontsize=16)

for i in range(10):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap=plt.cm.binary)

    plt.xlabel(class_names[train_labels[i]])

plt.show()
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)), # the nn will learn the good filter to use

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(6, activation=tf.nn.softmax)

])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=128, epochs=5, validation_split = 0.2)
test_loss = model.evaluate(test_images, test_labels)
index = np.random.randint(test_images.shape[0]) # We choose a random index



img = (np.expand_dims(test_images[index], 0))

predictions = model.predict(img)     # Vector of probabilities

pred_img = np.argmax(predictions[0]) # We take the highest probability

pred_label = class_names[pred_img]

true_label = class_names[test_labels[index]] 



title = 'Guess : {} VS Pred : {}  '.format(pred_label , true_label )



plt.figure()

plt.imshow(test_images[index])

plt.grid(False)

plt.title(title)

plt.show()
def print_mislabeled_images(class_names, test_images, test_labels, pred_labels):

    BOO = (test_labels == pred_labels)

    mislabeled_indices = np.where(BOO == 0)

    mislabeled_images = test_images[mislabeled_indices]

    mislabeled_labels = pred_labels[mislabeled_indices]

    fig = plt.figure(figsize=(10,10))

    fig.suptitle("Some examples of mislabeled images by the classifier:", fontsize=16)

    for i in range(10):

        plt.subplot(5,5,i+1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(mislabeled_images[i], cmap=plt.cm.binary)

        plt.xlabel(class_names[mislabeled_labels[i]])

    plt.show()
predictions = model.predict(test_images)

pred_labels = np.argmax(predictions, axis = 1)

print_mislabeled_images(class_names, test_images, test_labels, pred_labels)