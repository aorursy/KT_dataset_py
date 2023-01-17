import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



import tensorflow as tf

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 
data_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

data_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
x_train = np.array(data_train.iloc[:, 1:])

y_train = np.array(data_train.iloc[:, 0])



x_test = np.array(data_test.iloc[:, 1:])

y_test = np.array(data_test.iloc[:, 0])



print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)



x_train = x_train.reshape(x_train.shape[0], 28, 28)

x_test = x_test.reshape(x_test.shape[0], 28, 28)



# Print the number of training and test datasets

print(x_train.shape[0], 'train set')

print(x_test.shape[0], 'test set')



# Define the text labels

fashion_mnist_labels = ["T-shirt/top",  # index 0

                        "Trouser",      # index 1

                        "Pullover",     # index 2 

                        "Dress",        # index 3 

                        "Coat",         # index 4

                        "Sandal",       # index 5

                        "Shirt",        # index 6 

                        "Sneaker",      # index 7 

                        "Bag",          # index 8 

                        "Ankle boot"]   # index 9
# Create a dictionary for each type of label 

labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",

          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}



def get_classes_distribution(data):

    # Get the count for each label

    label_counts = data["label"].value_counts()



    # Get total number of samples

    total_samples = len(data)





    # Count the number of items in each class

    for i in range(len(label_counts)):

        label = labels[label_counts.index[i]]

        count = label_counts.values[i]

        percent = (count / total_samples) * 100

        print("{:<20s}:   {} or {}%".format(label, count, percent))



get_classes_distribution(data_train)
def plot_label_per_class(data):

    f, ax = plt.subplots(1,1, figsize=(12,4))

    g = sns.countplot(data.label, order = data["label"].value_counts().index)

    g.set_title("Number of labels for each class")



    for p, label in zip(g.patches, data["label"].value_counts().index):

        g.annotate(labels[label], (p.get_x(), p.get_height()+0.1))

    plt.show()  

    

plot_label_per_class(data_train)
get_classes_distribution(data_test)
plot_label_per_class(data_test)
x_train = x_train.astype('float32') / 255

x_test = x_test.astype('float32') / 255
# Further break training data into train / validation sets

(x_train, x_valid) = x_train[5000:], x_train[:5000] 

(y_train, y_valid) = y_train[5000:], y_train[:5000]



# Reshape input data

x_train = x_train.reshape(x_train.shape[0], 28,28, 1)

x_valid = x_valid.reshape(x_valid.shape[0], 28,28, 1)

x_test = x_test.reshape(x_test.shape[0], 28,28, 1)



# One-hot encode the labels

y_train = tf.keras.utils.to_categorical(y_train, 10)

y_valid = tf.keras.utils.to_categorical(y_valid, 10)

y_test = tf.keras.utils.to_categorical(y_test, 10)



# Print training set shape

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)



# Print the number of training, validation, and test datasets

print(x_train.shape[0], 'train set')

print(x_valid.shape[0], 'validation set')

print(x_test.shape[0], 'test set')
def plot_count_per_class(yd):

    ydf = pd.DataFrame(yd)

    f, ax = plt.subplots(1,1, figsize=(12,4))

    g = sns.countplot(ydf[0], order = np.arange(0,10))

    g.set_title("Number of items for each class")

    g.set_xlabel("Category")

    

    for p, label in zip(g.patches, np.arange(0,10)):

        g.annotate(labels[label], (p.get_x(), p.get_height()+0.1))

        

    plt.show()  



def get_count_per_class(yd):

    ydf = pd.DataFrame(yd)

    # Get the count for each label

    label_counts = ydf[0].value_counts()



    # Get total number of samples

    total_samples = len(yd)





    # Count the number of items in each class

    for i in range(len(label_counts)):

        label = labels[label_counts.index[i]]

        count = label_counts.values[i]

        percent = (count / total_samples) * 100

        print("{:<20s}:   {} or {}%".format(label, count, percent))

    

plot_count_per_class(np.argmax(y_train,axis=1))

get_count_per_class(np.argmax(y_train,axis=1))
plot_count_per_class(np.argmax(y_valid,axis=1))

get_count_per_class(np.argmax(y_valid,axis=1))
model = tf.keras.Sequential()



# Must define the input shape in the first layer of the neural network

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 

model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model.add(tf.keras.layers.Dropout(0.2))



model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model.add(tf.keras.layers.Dropout(0.2))



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(10, activation='softmax'))



# Take a look at the model summary

model.summary()
model.compile(loss='categorical_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])
history = model.fit(x_train,y_train, batch_size=100, epochs=30, validation_data=(x_valid, y_valid))
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.ylim([-1,1])

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylabel('Cross Entropy')

plt.ylim([-1,1.0])

plt.title('Training and Validation Loss')

plt.xlabel('epoch')

plt.show()
# Evaluate the model on test set

score = model.evaluate(x_test, y_test, verbose=0)



# Print test accuracy

print('\n', 'Test accuracy:', score[1])
y_hat = model.predict(x_test)

predicted_classes = model.predict_classes(x_test)

y_true = data_test.iloc[:, 0]
target_names = ["Class {} ({}) :".format(i,labels[i]) for i in range(10)]

print(classification_report(y_true, predicted_classes, target_names=target_names))
figure = plt.figure(figsize=(20, 8))

for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):

    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])

    # Display each image

    ax.imshow(np.squeeze(x_test[index]))

    predict_index = np.argmax(y_hat[index])

    true_index = np.argmax(y_test[index])

    # Set the title for each image

    ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index], 

                                  fashion_mnist_labels[true_index]),

                                  color=("green" if predict_index == true_index else "red"))