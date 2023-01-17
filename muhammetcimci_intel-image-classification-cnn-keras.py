import numpy as np

import os

from sklearn.metrics import confusion_matrix

import seaborn as sn; sn.set(font_scale=1.4)

from sklearn.utils import shuffle           

import matplotlib.pyplot as plt             

import cv2                                 

import tensorflow as tf                

from tqdm import tqdm
class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']

class_names_label = {class_name:i for i, class_name in enumerate(class_names)}



nb_classes = len(class_names)



IMAGE_SIZE = (150, 150)
def load_data():

    """

        Load the data:

            - 14,034 images to train the network.

            - 3,000 images to evaluate how accurately the network learned to classify images.

    """

    

    datasets = ['../input/seg_train/seg_train', '../input/seg_test/seg_test']

    output = []

    

    # Iterate through training and test sets

    for dataset in datasets:

        

        images = []

        labels = []

        

        print("Loading {}".format(dataset))

        

        # Iterate through each folder corresponding to a category

        for folder in os.listdir(dataset):

            label = class_names_label[folder]

            

            # Iterate through each image in our folder

            for file in tqdm(os.listdir(os.path.join(dataset, folder))):

                

                # Get the path name of the image

                img_path = os.path.join(os.path.join(dataset, folder), file)

                

                # Open and resize the img

                image = cv2.imread(img_path)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = cv2.resize(image, IMAGE_SIZE) 

                

                # Append the image and its corresponding label to the output

                images.append(image)

                labels.append(label)

                

        images = np.array(images, dtype = 'float32')

        labels = np.array(labels, dtype = 'int32')   

        

        output.append((images, labels))



    return output
(train_images, train_labels), (test_images, test_labels) = load_data()
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
n_train = train_labels.shape[0]

n_test = test_labels.shape[0]



print ("Number of training examples: {}".format(n_train))

print ("Number of testing examples: {}".format(n_test))

print ("Each image is of size: {}".format(IMAGE_SIZE))
import pandas as pd



_, train_counts = np.unique(train_labels, return_counts=True)

_, test_counts = np.unique(test_labels, return_counts=True)

pd.DataFrame({'train': train_counts,

                    'test': test_counts}, 

             index=class_names

            ).plot.bar()

plt.show()
plt.pie(train_counts,

        explode=(0, 0, 0, 0, 0, 0) , 

        labels=class_names,

        autopct='%1.1f%%')

plt.axis('equal')

plt.title('Proportion of each observed category')

plt.show()
train_images = train_images / 255.0 

test_images = test_images / 255.0
def display_random_image(class_names, images, labels):

    """

        Display a random image from the images array and its correspond label from the labels array.

    """

    

    index = np.random.randint(images.shape[0])

    plt.figure()

    plt.imshow(images[index])

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.title('Image #{} : '.format(index) + class_names[labels[index]])

    plt.show()
display_random_image(class_names, train_images, train_labels)
def display_examples(class_names, images, labels):

    """

        Display 25 images from the images array with its corresponding labels

    """

    

    fig = plt.figure(figsize=(10,10))

    fig.suptitle("Some examples of images of the dataset", fontsize=16)

    for i in range(25):

        plt.subplot(5,5,i+1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(images[i], cmap=plt.cm.binary)

        plt.xlabel(class_names[labels[i]])

    plt.show()
display_examples(class_names, train_images, train_labels)
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(6, activation=tf.nn.softmax)

])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=128, epochs=20, validation_split = 0.2)
def plot_accuracy_loss(history):

    """

        Plot the accuracy and the loss during the training of the nn.

    """

    fig = plt.figure(figsize=(10,5))



    # Plot accuracy

    plt.subplot(221)

    plt.plot(history.history['acc'],'bo--', label = "acc")

    plt.plot(history.history['val_acc'], 'ro--', label = "val_acc")

    plt.title("train_acc vs val_acc")

    plt.ylabel("accuracy")

    plt.xlabel("epochs")

    plt.legend()



    # Plot loss function

    plt.subplot(222)

    plt.plot(history.history['loss'],'bo--', label = "loss")

    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")

    plt.title("train_loss vs val_loss")

    plt.ylabel("loss")

    plt.xlabel("epochs")



    plt.legend()

    plt.show()
plot_accuracy_loss(history)
test_loss = model.evaluate(test_images, test_labels)
predictions = model.predict(test_images)     # Vector of probabilities

pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability



display_random_image(class_names, test_images, pred_labels)
def print_mislabeled_images(class_names, test_images, test_labels, pred_labels):

    """

        Print 25 examples of mislabeled images by the classifier, e.g when test_labels != pred_labels

    """

    BOO = (test_labels == pred_labels)

    mislabeled_indices = np.where(BOO == 0)

    mislabeled_images = test_images[mislabeled_indices]

    mislabeled_labels = pred_labels[mislabeled_indices]



    title = "Some examples of mislabeled images by the classifier:"

    display_examples(class_names,  mislabeled_images, mislabeled_labels)

print_mislabeled_images(class_names, test_images, test_labels, pred_labels)
CM = confusion_matrix(test_labels, pred_labels)

ax = plt.axes()

sn.heatmap(CM, annot=True, 

           annot_kws={"size": 10}, 

           xticklabels=class_names, 

           yticklabels=class_names, ax = ax)

ax.set_title('Confusion matrix')

plt.show()
from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input



model = VGG16(weights='imagenet', include_top=False)
train_features = model.predict(train_images)

test_features = model.predict(test_images)
n_train, x, y, z = train_features.shape

n_test, x, y, z = test_features.shape

numFeatures = x * y * z
from sklearn import decomposition



pca = decomposition.PCA(n_components = 2)



X = train_features.reshape((n_train, x*y*z))

pca.fit(X)



C = pca.transform(X) # Représentation des individus dans les nouveaux axe

C1 = C[:,0]

C2 = C[:,1]
### Figures



plt.subplots(figsize=(10,10))



for i, class_name in enumerate(class_names):

    plt.scatter(C1[train_labels == i][:1000], C2[train_labels == i][:1000], label = class_name, alpha=0.4)

plt.legend()

plt.title("PCA Projection")

plt.show()
model2 = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape = (x, y, z)),

    tf.keras.layers.Dense(50, activation=tf.nn.relu),

    tf.keras.layers.Dense(6, activation=tf.nn.softmax)

])



model2.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])



history2 = model2.fit(train_features, train_labels, batch_size=128, epochs=15, validation_split = 0.2)
plot_accuracy_loss(history)
test_loss = model2.evaluate(test_features, test_labels)
np.random.seed(seed=1997)

# Number of estimators

n_estimators = 10

# Proporition of samples to use to train each training

max_samples = 0.8



max_samples *= n_train

max_samples = int(max_samples)
models = list()

random = np.random.randint(50, 100, size = n_estimators)



for i in range(n_estimators):

    

    # Model

    model = tf.keras.Sequential([ tf.keras.layers.Flatten(input_shape = (x, y, z)),

                                # One layer with random size

                                    tf.keras.layers.Dense(random[i], activation=tf.nn.relu),

                                    tf.keras.layers.Dense(6, activation=tf.nn.softmax)

                                ])

    

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

    

    # Store model

    models.append(model)
histories = []



for i in range(n_estimators):

    # Train each model on a bag of the training data

    train_idx = np.random.choice(len(train_features), size = max_samples)

    histories.append(models[i].fit(train_features[train_idx], train_labels[train_idx], batch_size=128, epochs=10, validation_split = 0.1))
predictions = []

for i in range(n_estimators):

    predictions.append(models[i].predict(test_features))

    

predictions = np.array(predictions)

predictions = predictions.sum(axis = 0)

pred_labels = predictions.argmax(axis=1)
from sklearn.metrics import accuracy_score

print("Accuracy : {}".format(accuracy_score(test_labels, pred_labels)))
from keras.models import Model



model = VGG16(weights='imagenet', include_top=False)

model = Model(inputs=model.inputs, outputs=model.layers[-5].output)
train_features = model.predict(train_images)

test_features = model.predict(test_images)
from keras.layers import Input, Dense, Conv2D, Activation , MaxPooling2D, Flatten



model2 = VGG16(weights='imagenet', include_top=False)



input_shape = model2.layers[-4].get_input_shape_at(0) # get the input shape of desired layer

layer_input = Input(shape = (9, 9, 512)) # a new input tensor to be able to feed the desired layer

# https://stackoverflow.com/questions/52800025/keras-give-input-to-intermediate-layer-and-get-final-output



x = layer_input

for layer in model2.layers[-4::1]:

    x = layer(x)

    

x = Conv2D(64, (3, 3), activation='relu')(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)

x = Dense(100,activation='relu')(x)

x = Dense(6,activation='softmax')(x)



# create the model

new_model = Model(layer_input, x)
new_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
new_model.summary()
history = new_model.fit(train_features, train_labels, batch_size=128, epochs=10, validation_split = 0.2)
plot_accuracy_loss(history)
from sklearn.metrics import accuracy_score



predictions = new_model.predict(test_features)    

pred_labels = np.argmax(predictions, axis = 1)

print("Accuracy : {}".format(accuracy_score(test_labels, pred_labels)))