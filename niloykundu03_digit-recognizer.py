from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import InputLayer, Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape

from keras.optimizers import Adam

from keras.utils import to_categorical #, plot_model

# from keras.utils.vis_utils import model_to_dot

# from keras import backend



import numpy as np

import pandas as pd

import statistics as stat



from sklearn.metrics import confusion_matrix #, classification_report

# from sklearn.tree import DecisionTreeClassifier

# from sklearn.utils import shuffle



from IPython.display import Image, SVG



%matplotlib inline

import matplotlib.pyplot as plt

# Load the data set

# Train data

data_train = np.array(pd.read_csv('../input/train.csv'))

x_train = data_train[:38000,1:] 

y_train = data_train[:38000,0]



# Test data

x_test = data_train[38000:,1:] 

y_test = data_train[38000:,0]



# Submission Data

submission_train = np.array(pd.read_csv('../input/test.csv'))

# Chech the Shape of the Data Set

print('Training dataset shape:')

print(x_train.shape)

print(y_train.shape)

print('Testing dataset shape:')

print(x_test.shape)

print(y_test.shape)

print('Submission dataset shape:')

print(submission_train.shape)

# Reshape Training and Testing data

train_data = x_train.reshape((-1, 28, 28, 1))

test_data = x_test.reshape((-1, 28, 28, 1))

submission_data = submission_train.reshape((-1, 28, 28, 1))

# Convert >192 pixel value to 255 and <64 to 0

# In this images, total 256 greyscale levels are there.

# Here, I am considering that, if the pixel value is more than or equal to 75% of 256, i.e., 192; then I am 

# changing the pixel value to 255 (heighest greyscale level) and if the pixel value is less than equal to 25% of 256,

# i.e., 64; then I am changing the pixel value to 0 (lowest greyscale level).

train_data[train_data >= 192] = 255

train_data[train_data <= 64] = 0

# To Plot the the images

def plot_of_images(img, labels_actual, labels_pred=None):

    # fig with 3x3 subplots

    fig, axes = plt.subplots(3, 3)

    fig.subplots_adjust(hspace=0.4, wspace=0.2)



    for i, ax in enumerate(axes.flat):

        try:

            ax.imshow(img[i].reshape((28, 28)), cmap='binary')

            if labels_pred is None:

                xlabel = "True: {0}".format(labels_actual[i])

            else:

                xlabel = "True: {0}, Pred: {1}".format(labels_actual[i], labels_pred[i])

            ax.set_xlabel(xlabel)

            ax.set_xticks([])

            ax.set_yticks([])

        except IndexError:

            continue

    plt.show()

plot_of_images(x_train[100:109], y_train[100:109])
# Create an empty list to store the accuracy

testing_accuracy_list = []



for itr in range(5):

    model = Sequential()

    

    # Input layer

    model.add(InputLayer(input_shape=(28, 28, 1)))



    model.add(Conv2D(10, kernel_size=(5, 5), padding='same', activation='relu'))

    model.add(Dropout(rate=0.5))



    model.add(Conv2D(25, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'))

    model.add(Dropout(rate=0.25))



    model.add(Conv2D(50, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(rate=0.25))



    model.add(Flatten())

    

    # Dense layer

    model.add(Dense(1000, activation='relu'))

    model.add(Dropout(rate=0.25))

    

    model.add(Dense(10, activation='softmax'))



    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])



    train_data_generator= ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1,

                                             samplewise_center=True, samplewise_std_normalization=True)



    train_generator = train_data_generator.flow(x=train_data, y=to_categorical(y_train), batch_size=100)

    

    # Fit the model

    model.fit_generator(train_generator, steps_per_epoch=3000, epochs=3, use_multiprocessing=True)

    

    # Predict the model on the Test Data set

    labels_pred_matrix = model.predict(x=test_data)

    labels_pred = np.argmax(labels_pred_matrix, axis=1)



    # Calculation of accuracy of the model on the Test Data set

    accuracy_on_testing = sum(y_test == labels_pred)/x_test.shape[0]

    

    # Append the test accuracy in the empty list

    testing_accuracy_list.append(accuracy_on_testing)



print("accuracy List:", testing_accuracy_list)

print('average accuracy: {:.4f}'.format(stat.mean(testing_accuracy_list)))

# Create Confusion matrix

cf_matrix = confusion_matrix(y_true=y_test, y_pred=labels_pred)



# Calculation

actual_degits_count = np.sum(cf_matrix, axis=1)

correct_pred_count = np.diag(cf_matrix)

wrong_pred_count = actual_degits_count - correct_pred_count



# Print the details

print('Accuracy:')

print(accuracy_on_testing)

print('\nActual Digits:')

print(actual_degits_count)

print('\nCorrect Prediction:')

print(correct_pred_count)

print('\nWrong Prediction:')

print(wrong_pred_count)

print('Total: {}'.format(sum(wrong_pred_count)))

print('sd: {:.2f}'.format(stat.stdev(wrong_pred_count)))

print('\n')

print(cf_matrix)

# Get the index of incorrect prediction

boolean_incorrect_pred = ~(y_test == labels_pred)

error_index_list = np.where(boolean_incorrect_pred)[0]
# Plotting of Digits which is incorrctly predicted

def plot_error_digits(error_index, particular_digit=None):

    error_index_9 = error_index[:9]

    if(particular_digit!=None):

        error_index_9 = np.where((y_test == particular_digit) & boolean_incorrect_pred)[0]

        

    plot_of_images(img=x_test[error_index_9],

                   labels_actual=y_test[error_index_9],

                   labels_pred=labels_pred[error_index_9])
for i in range(10):

    plot_error_digits(error_index_list, particular_digit=i)

# Predict the model on the Training Data set

labels_train_pred = np.argmax(model.predict(x=train_data), axis=1)



# Calculation of accuracy

accuracy_on_training = sum(y_train == labels_train_pred)/x_train.shape[0]



# Print the Accuracy

print('Accuracy: {:.4f}'.format(accuracy_on_training))

# Predict the model on the Submission Data set

labels_submission_pred = np.argmax(model.predict(x=submission_data), axis=1)

pd.DataFrame({'ImageId': range(1, 28001), 'Label': labels_submission_pred}).to_csv("cnn_mnist_digit_submit.csv")

VImageId,
