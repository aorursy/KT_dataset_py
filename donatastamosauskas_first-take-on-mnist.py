import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
%matplotlib inline

import os
print(os.listdir("../input"))
print(os.listdir("../working"))
train_data_path = '../input/train.csv'
test_data_path = '../input/test.csv'
weights_file_name = 'mnist_model_weights.h5'
img_height = 28
img_width = 28
training_data = pd.read_csv(train_data_path)
testing_data = pd.read_csv(test_data_path)
def prepare_image_data(data):        
    data_copy = data.copy()
    if 'label' in data_copy.columns:
            data_copy = data_copy.drop('label', axis=1) 
    data_array = data_copy.values.reshape(-1, img_height, img_width, 1) / 255
    return data_array
    
def prepare_labels(data, number_of_classes):
    return keras.utils.to_categorical(data.label.values, number_of_classes)

def show_image(image_array, label):
    plt.title(str(np.argmax(label)))
    plt.imshow(image_array.reshape(img_height, img_width))
    
def show_multiple_images(images_array, labels, columns=6, rows=2, figsize=(12,5)):
    fig = plt.figure(figsize=figsize)
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        show_image(images_array[i,:,:,:], labels[i,:])
    plt.show()
x = prepare_image_data(training_data)
y = prepare_labels(training_data, number_of_classes=10)

show_multiple_images(x, y)
from sklearn.model_selection import train_test_split
validation_percentage = 0.2

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_percentage)
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

def get_fitted_data_generator(data):
    data_generator = ImageDataGenerator(rotation_range=5, 
                                        width_shift_range=0.15,
                                        height_shift_range=0.15, 
                                        zoom_range=0.2)
    data_generator.fit(data)
    return data_generator
data_generator = get_fitted_data_generator(x[:5,:,:,:])
generated_data_iterator = data_generator.flow(x[:5,:,:,:], y[:5,:], batch_size=1)
manipulated_data = [next(generated_data_iterator) for i in range(35)]

manipulated_images = []
manipulated_labels = []

for item in manipulated_data:
    manipulated_images.append(item[0])
    manipulated_labels.append(item[1])

show_multiple_images(np.array(manipulated_images), np.array(manipulated_labels), rows=5)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.python import keras
nb_filters = 15 # at the moment not used
stride_size = 2 # at the moment not used
dense_neurons = 256
nbr_classes = 10 # from 0 to 9

mnist_model = Sequential()

mnist_model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', 
                       input_shape=(img_height, img_width, 1)))
mnist_model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu',
                       strides=2))
mnist_model.add(Dropout(0.25))

mnist_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
mnist_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                      strides=2))
mnist_model.add(Dropout(0.25))
                
mnist_model.add(Flatten())
mnist_model.add(Dense(dense_neurons, activation='relu'))
mnist_model.add(Dropout(0.25))
                
mnist_model.add(Dense(nbr_classes, activation='softmax'))

mnist_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nb_filters = 15 # at the moment not used
stride_size = 2 # at the moment not used
dense_neurons = 256
nbr_classes = 10 # from 0 to 9

mnist_sized_up = Sequential()

mnist_sized_up.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', 
                       input_shape=(img_height, img_width, 1)))
mnist_sized_up.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu',
                       strides=2))
mnist_sized_up.add(Dropout(0.25))

mnist_sized_up.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
mnist_sized_up.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
mnist_sized_up.add(Dropout(0.25))
                
mnist_sized_up.add(Flatten())
mnist_sized_up.add(Dense(dense_neurons, activation='relu'))
mnist_sized_up.add(Dropout(0.25))

mnist_sized_up.add(Dense(dense_neurons // 2, activation='relu'))
mnist_sized_up.add(Dropout(0.25))
                
mnist_sized_up.add(Dense(nbr_classes, activation='softmax'))

mnist_sized_up.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
def fit_model_generator(model, x_train, y_train, epochs=1, batch=36,
                        use_validation_data=False, val_data=None):
    
    image_nbr = np.size(x_train, 0)
    data_generator = get_fitted_data_generator(x_train)
    
    if use_validation_data:
        return model.fit_generator(data_generator.flow(x_train, y_train, batch_size=batch), 
                                   steps_per_epoch=(image_nbr//batch), epochs=epochs, 
                                   validation_data=val_data, verbose=1)
    else:
        return model.fit_generator(data_generator.flow(x_train, y_train, batch_size=batch), 
                                   steps_per_epoch=(image_nbr//batch), epochs=epochs, verbose=1)
compare_epochs = 30

mnist_model_hist = fit_model_generator(mnist_model, x_train, y_train, epochs=compare_epochs, use_validation_data=True, val_data=(x_val, y_val))
mnist_sized_up_hist = fit_model_generator(mnist_sized_up, x_train, y_train, epochs=compare_epochs, use_validation_data=True, val_data=(x_val, y_val))
def plot_model_training_history(training_history):
    plot_space_compare = plt.figure(figsize=(10, 5))

    plot_space_compare.add_subplot(1, 2, 1)
    plt.plot(training_history.history['acc'], label='Training')
    plt.plot(training_history.history['val_acc'], label='Validation')
    plt.legend(loc='best')
    plt.title("Accuracy")
    plt.xlabel("Epochs")

    plot_space_compare.add_subplot(1, 2, 2)
    plt.plot(training_history.history['loss'], label='Training')
    plt.plot(training_history.history['val_loss'], label='Validation')
    plt.legend(loc='best')
    plt.title("Loss")
    plt.xlabel("Epochs")

    plt.show()
plot_model_training_history(mnist_model_hist)
plot_model_training_history(mnist_sized_up_hist)
epochs = 1

model_history = fit_model_generator(mnist_model, x, y, epochs=epochs, use_validation_data=False)
plot_space = plt.figure(figsize=(10, 5))

plot_space.add_subplot(1, 2, 1)
plt.plot(model_history.history['acc'])
plt.title("Accuracy")
plt.xlabel("Epochs")

plot_space.add_subplot(1, 2, 2)
plt.plot(model_history.history['loss'])
plt.title("Loss")
plt.xlabel("Epochs")

plt.show()
validation_x = prepare_image_data(testing_data)
validation_y_one_hot = mnist_model.predict(validation_x)
validation_y = np.array([np.argmax(prediction) for prediction in validation_y_one_hot])
show_multiple_images(validation_x, validation_y_one_hot)
submission_filename = 'submission_v3.csv'

data_to_file = pd.DataFrame({'Label':validation_y})
data_to_file['ImageId'] = data_to_file.index + 1
data_to_file.to_csv(submission_filename, columns=['ImageId', 'Label'], index=False)