%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

import time

from keras.models import Model, load_model, save_model

from keras.layers import Conv2D, Dense, Input, Activation, BatchNormalization, Dropout, Flatten

from keras.layers import MaxPooling2D

from keras.optimizers import Adam
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_csv = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test_csv = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
y = train_csv.label.values

x = train_csv.drop(columns=['label']).values

print('x shape: ', x.shape)

print('y shape: ', y.shape)
x = x.reshape(60000,28,28)

plt.figure()

plt.imshow(x[0], cmap=plt.cm.binary)

plt.colorbar()

plt.grid(False)

plt.show()
#let's select 5 random images from each class

np.random.seed(2)

plt.figure(figsize=(15,15))

for label in range(10):

    for i,n in enumerate(train_csv.loc[train_csv.label == label].sample(5).index):

        plt.subplot(10,5,(label*5+i+1))

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(x[n], cmap=plt.cm.binary)

        plt.xlabel(y[n])

plt.show()
# take random 5000 from train_csv for later accuracy validation

# else goes to training

val_df = train_csv.sample(5000)

train_df = train_csv.drop(index=val_df.index)



# prepare data for training

# set input shape for Conv2D layers

x_train = train_df.drop(columns=['label']).values

x_train = x_train.reshape(55000,28,28,1)



y_train = train_df.label.values



x_val = val_df.drop(columns=['label']).values

x_val = x_val.reshape(5000,28,28,1)



y_val = val_df.label.values
def main_model(input_shape):

    """

    Current NN model

    Arguments:

        input_shape -- shape of the images of the dataset

    Returns:

        model -- a Model() instance in Keras

    """

    

    X_input = Input(input_shape)

    X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv2d0', padding="same")(X_input)

    X = BatchNormalization(name = 'bn0')(X)

    X = Activation('relu')(X)

    

    X = MaxPooling2D((2, 2), name='max_pool0')(X)

    

    X = Conv2D(128, (1, 1), strides = (1, 1), name = 'conv2d1', padding="same")(X) 

    X = Activation('relu')(X)

    

    X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv2d2', padding="same")(X)

    X = BatchNormalization(axis = 3, name = 'bn1')(X)

    X = Activation('relu')(X)

    

    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    

    X = Flatten()(X)

    X = Dropout(rate = 0.5)(X)

    

    X = Dense(64, activation='relu', name='fc0')(X)

    

    X = Dense(10, activation='softmax', name='fc1')(X)



    return Model(inputs = X_input, outputs = X, name='142K_Conv_NN')
model1 = main_model(x_train[0].shape)

model1.compile(optimizer = "Adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model1.summary()
result = model1.fit(x = x_train, y = y_train, batch_size=20, epochs=15, validation_split=0.1, verbose=2)
plt.figure(figsize=(15,5))

plt.subplot(121)

plt.plot(result.history['accuracy'], label='acc')

plt.plot(result.history['val_accuracy'], label='val_acc')

plt.title('Validation accuracy')

plt.title('Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(loc='lower right')

plt.subplot(122)

plt.plot(result.history['loss'], label='loss')

plt.plot(result.history['val_loss'], label='val_loss')

plt.title('Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(loc='upper right')

plt.show()
preds = model1.evaluate(x = x_val, y = y_val)

print ("Evaluation Set Accuracy = " + str(preds[1]))
predictions = model1.predict(x_val)

test_labels = y_val

test_images = x_val.squeeze()



test_labels = pd.Series(y_val)

pred_labels = pd.Series(predictions.argmax(axis=1))



# sample 15 random mislabeled images

wrong_class = list( test_labels[test_labels != pred_labels].sample(15).index )

print("Incorrect classification indexes: ", wrong_class)
class_names = [0,1,2,3,4,5,6,7,8,9,10]

def plot_image(i, predictions_array, true_label, img):

    predictions_array, true_label, img = predictions_array, true_label[i], img[i]

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])



    plt.imshow(img, cmap=plt.cm.binary)



    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label:

      color = 'blue'

    else:

      color = 'red'



    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],

                                100*np.max(predictions_array),

                                class_names[true_label]),

                                color=color)



def plot_value_array(i, predictions_array, true_label):

    predictions_array, true_label = predictions_array, true_label[i]

    plt.grid(False)

    plt.xticks(range(10))

    plt.yticks([])

    thisplot = plt.bar(range(10), predictions_array, color="#777777")

    plt.ylim([0, 1])

    predicted_label = np.argmax(predictions_array)



    thisplot[predicted_label].set_color('red')

    thisplot[true_label].set_color('blue')

num_cols = 3

num_rows = 5

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i, idx in enumerate(wrong_class):

    #print(num_rows, num_cols, 2*i+1)

    plt.subplot(num_rows, 2*num_cols, 2*i+1)

    plot_image(idx, predictions[idx], test_labels, test_images)

    #print(num_rows, num_cols, 2*i+2)

    plt.subplot(num_rows, 2*num_cols, 2*i+2)

    plot_value_array(idx, predictions[idx], test_labels)

plt.tight_layout()

plt.show()
test_x = test_csv.drop(columns=['id']).values

print('test_x shape: ', test_x.shape)
test_x = test_x.reshape(test_x.shape[0],28,28,1)

predictions = model1.predict(test_x)



pred_labels = predictions.argmax(axis=1)

submission = pd.DataFrame({'id': test_csv['id'], 'label': pred_labels})

submission.to_csv('submission.csv',index=False)