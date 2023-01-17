import numpy as np

import pandas as pd

import random

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 
IMG_ROWS = 28

IMG_COLS = 28

NUM_CLASSES = 10

TEST_SIZE = 0.1

RANDOM_STATE = 2018

#Model

NO_EPOCHS = 150

PATIENCE = 20

VERBOSE = 1

BATCH_SIZE = 128



IS_LOCAL = False



import os



if(IS_LOCAL):

    PATH="../input/digit-recognizer/"

else:

    PATH="../input/"

print(os.listdir(PATH))
train_file = PATH+"train.csv"

test_file  = PATH+"test.csv"



train_df = pd.read_csv(train_file)

test_df = pd.read_csv(test_file)
print("MNIST train -  rows:",train_df.shape[0]," columns:", train_df.shape[1])

print("MNIST test -  rows:",test_df.shape[0]," columns:", test_df.shape[1])
def get_classes_distribution(data):

    # Get the count for each label

    label_counts = data["label"].value_counts()



    # Get total number of samples

    total_samples = len(data)



    # Count the number of items in each class

    for i in range(len(label_counts)):

        label = label_counts.index[i]

        count = label_counts.values[i]

        percent = (count / total_samples) * 100

        print("{}:   {} or {}%".format(label, count, percent))



get_classes_distribution(train_df)
f, ax = plt.subplots(1,1, figsize=(8,6))

g = sns.countplot(train_df.label)

g.set_title("Number of labels for each class")

plt.show()    
def sample_images_data(data, hasLabel=True):

    # An empty list to collect some samples

    sample_images = []

    sample_labels = []



    # Iterate over the keys of the labels dictionary defined in the above cell

    if(hasLabel):

        for k in range(0,10):

            # Get four samples for each category

            samples = data[data["label"] == k].head(4)

            # Append the samples to the samples list

            for j, s in enumerate(samples.values):

                # train data: First column contain labels, hence index should start from 1

                img = np.array(samples.iloc[j, 1:]).reshape(IMG_ROWS,IMG_COLS)

                sample_images.append(img)

                sample_labels.append(samples.iloc[j, 0])

    else:

        import random

        samples = data.iloc[random.sample(range(1, 10000), 40),]

        for j, s in enumerate(samples.values):

            # test data: First column contain pixels, hence index should start from 0

            img = np.array(samples.iloc[j, 0:]).reshape(IMG_ROWS,IMG_COLS)

            sample_images.append(img)

            sample_labels.append(-1)

                

    print("Total number of sample images to plot: ", len(sample_images))

    return sample_images, sample_labels



train_sample_images, train_sample_labels = sample_images_data(train_df)
def plot_sample_images(data_sample_images,data_sample_labels,cmap="Blues"):

    # Plot the sample images now

    f, ax = plt.subplots(5,8, figsize=(16,10))



    for i, img in enumerate(data_sample_images):

        ax[i//8, i%8].imshow(img, cmap=cmap)

        ax[i//8, i%8].axis('off')

        ax[i//8, i%8].set_title(data_sample_labels[i])

    plt.show()    

    

plot_sample_images(train_sample_images,train_sample_labels, "Greens")
test_sample_images, test_sample_labels = sample_images_data(test_df,hasLabel=False)

plot_sample_images(test_sample_images,test_sample_labels)
# data preprocessing

def data_preprocessing(raw, hasLabel=True):

    start_pixel = 0

    if(hasLabel):

        start_pixel = 1

    if(hasLabel):

        out_y = keras.utils.to_categorical(raw.label, NUM_CLASSES)

    else:

        out_y = None

    num_images = raw.shape[0]

    x_as_array = raw.values[:,start_pixel:]

    x_shaped_array = x_as_array.reshape(num_images, IMG_ROWS, IMG_COLS, 1)

    out_x = x_shaped_array / 255

    return out_x, out_y
# prepare the data

X, y = data_preprocessing(train_df)

X_test, y_test = data_preprocessing(test_df,hasLabel=False)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print("MNIST train -  rows:",X_train.shape[0]," columns:", X_train.shape[1:4])

print("MNIST valid -  rows:",X_val.shape[0]," columns:", X_val.shape[1:4])

print("MNIST test -  rows:",X_test.shape[0]," columns:", X_test.shape[1:4])
def plot_count_per_class(yd):

    ydf = pd.DataFrame(yd)

    f, ax = plt.subplots(1,1, figsize=(12,4))

    g = sns.countplot(ydf[0], order = np.arange(0,10))

    g.set_title("Number of items for each class")

    g.set_xlabel("Category")

            

    plt.show()  



def get_count_per_class(yd):

    ydf = pd.DataFrame(yd)

    # Get the count for each label

    label_counts = ydf[0].value_counts()



    # Get total number of samples

    total_samples = len(yd)





    # Count the number of items in each class

    for i in range(len(label_counts)):

        label = label_counts.index[i]

        count = label_counts.values[i]

        percent = (count / total_samples) * 100

        print("{}:   {} or {}%".format(label, count, percent))

    

plot_count_per_class(np.argmax(y_train,axis=1))

get_count_per_class(np.argmax(y_train,axis=1))
plot_count_per_class(np.argmax(y_val,axis=1))

get_count_per_class(np.argmax(y_val,axis=1))
# Model

model = Sequential()

# Add convolution 2D

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', padding="same",

        kernel_initializer='he_normal',input_shape=(IMG_ROWS, IMG_COLS, 1)))



model.add(BatchNormalization())



model.add(Conv2D(32,kernel_size=(3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization())

# Add dropouts to the model

model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=(3, 3), strides=2,padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), strides=2,padding='same', activation='relu'))

# Add dropouts to the model

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

# Add dropouts to the model

model.add(Dropout(0.4))

model.add(Dense(NUM_CLASSES, activation='softmax'))
# Compile the model

model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
plot_model(model, to_file='model.png')

SVG(model_to_dot(model).create(prog='dot', format='svg'))
NO_EPOCHS = 10
from keras.callbacks import EarlyStopping, ModelCheckpoint

earlystopper = EarlyStopping(monitor='loss', patience=PATIENCE, verbose=VERBOSE)

checkpointer = ModelCheckpoint('best_model.h5',

                                monitor='val_acc',

                                verbose=VERBOSE,

                                save_best_only=True,

                                save_weights_only=True)



history = model.fit(X_train, y_train,

          batch_size=BATCH_SIZE,

          epochs=NO_EPOCHS,

          verbose=1,

          validation_data=(X_val, y_val),

          callbacks=[earlystopper, checkpointer])
def plot_accuracy_and_loss(train_model):

    hist = train_model.history

    acc = hist['acc']

    val_acc = hist['val_acc']

    loss = hist['loss']

    val_loss = hist['val_loss']

    epochs = range(len(acc))

    f, ax = plt.subplots(1,2, figsize=(14,6))

    ax[0].plot(epochs, acc, 'g', label='Training accuracy')

    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')

    ax[0].set_title('Training and validation accuracy')

    ax[0].legend()

    ax[1].plot(epochs, loss, 'g', label='Training loss')

    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')

    ax[1].set_title('Training and validation loss')

    ax[1].legend()

    plt.show()

plot_accuracy_and_loss(history)
print("run model - predict validation set")

score = model.evaluate(X_val, y_val, verbose=0)

print(f'Last validation loss: {score[0]}, accuracy: {score[1]}')

# load saved optimal model

model_optimal = model

model_optimal.load_weights('best_model.h5')

score = model_optimal.evaluate(X_val, y_val, verbose=0)

print(f'Best validation loss: {score[0]}, accuracy: {score[1]}')
def predict_show_classes(model, X_val, y_val):

    #get the predictions for the test data

    predicted_classes = model.predict_classes(X_val)

    #get the indices to be plotted

    y_true = np.argmax(y_val,axis=1)

    correct = np.nonzero(predicted_classes==y_true)[0]

    incorrect = np.nonzero(predicted_classes!=y_true)[0]

    print("Correct predicted classes:",correct.shape[0])

    print("Incorrect predicted classes:",incorrect.shape[0])

    target_names = ["Class {}:".format(i) for i in range(NUM_CLASSES)]

    print(classification_report(y_true, predicted_classes, target_names=target_names))

    return correct, incorrect
correct, incorrect = predict_show_classes(model, X_val, y_val)
correct, incorrect =  predict_show_classes(model_optimal, X_val, y_val)
def plot_images(data_index,cmap="Blues"):

    # Plot the sample images now

    f, ax = plt.subplots(4,4, figsize=(12,12))

    y_true = np.argmax(y_val,axis=1)

    for i, indx in enumerate(data_index[:16]):

        ax[i//4, i%4].imshow(X_val[indx].reshape(IMG_ROWS,IMG_COLS), cmap=cmap)

        ax[i//4, i%4].axis('off')

        ax[i//4, i%4].set_title("True:{}  Pred:{}".format(y_true[indx],predicted_classes[indx]))

    plt.show()    



plot_images(correct, "Greens")
plot_images(incorrect, "Reds")
y_cat = model.predict(X_test, batch_size=64)
y_pred = np.argmax(y_cat,axis=1)
output_file = "submission.csv"

with open(output_file, 'w') as f :

    f.write('ImageId,Label\n')

    for i in range(len(y_pred)) :

        f.write("".join([str(i+1),',',str(y_pred[i]),'\n']))
y_cat = model_optimal.predict(X_test, batch_size=64)

y_pred = np.argmax(y_cat,axis=1)

output_file = "submission_optimal.csv"

with open(output_file, 'w') as f :

    f.write('ImageId,Label\n')

    for i in range(len(y_pred)) :

        f.write("".join([str(i+1),',',str(y_pred[i]),'\n']))