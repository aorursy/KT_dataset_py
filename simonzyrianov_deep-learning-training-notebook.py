# import initial libraries we will need

# input data files are available in the "../input/" directory



import numpy as np

from IPython.display import display

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications import ResNet50, resnet50   

from learntools.deep_learning.decode_predictions import decode_predictions

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# load pre-trained resnet50 model

model_resnet50 = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5') # the default input size for this model is 224x224
# define directory to locate images

directory_test = '../input/hot-dog-not-hot-dog/seefood/test'
def test_model(model):

    """

    Takes model as input, reads images from test directory, makes predictions for test images.

    Then checks model accuracy and prints out accuracy statistics.

    Returns numpy arrays of processed and unprocessed images, true and predicted labels for further analysis.

    """

    # create ImageDataGenerator to load images from our directory, batch size is the whole set of images

    datagen_unprocessed = image.ImageDataGenerator(preprocessing_function=None) 

    datagen_processed = image.ImageDataGenerator(preprocessing_function=resnet50.preprocess_input)



    image_generator = datagen_processed.flow_from_directory(directory_test,

                                                            target_size=(224, 224),

                                                            class_mode='binary',

                                                            batch_size=500,

                                                            seed=0)

    images_arr_unprocessed = datagen_unprocessed.flow_from_directory(directory_test,

                                                                     target_size=(224, 224),

                                                                     batch_size=500,

                                                                     seed=0)[0][0]

    # assign y_true and numpy arrays of images

    images_arr_processed, y_true = image_generator[0][0], image_generator[0][1]

    

    if model == model_resnet50:

        # make and decode top 1 predictions with pre-trained model

        preds_decoded = decode_predictions(model.predict_generator(image_generator), top=1)



        # assign y_predicted for model_resnet50 model

        y_predicted = np.asarray([preds[0][1] for preds in preds_decoded]) != 'hotdog'

    else:

        # assign y_predicted for new model with last dense layer with sigmoid activation

        # we assign 1 label to activations greater than 0.5 

        y_predicted = (model.predict_generator(image_generator).flatten() > 0.5)   

    

    # print accuracy and f1 score

    print('Model: {}'.format(model.name))

    print('Class labels: {}'.format(image_generator.class_indices))

    print('Correctly classified: {} images out of {}'.format((y_true == y_predicted).sum(), len(y_true)))

    print('\tAccuracy score {:.2}'.format(accuracy_score(y_true, y_predicted)))

    print('\tF1 score {:.2}'.format(f1_score(y_true, y_predicted)))

    

    return images_arr_unprocessed, images_arr_processed, y_true, y_predicted 
# test pre-trained resnet50 model

images_arr_unprocessed, images_arr_processed, y_true, y_predicted = test_model(model_resnet50)
# create empty dataframe for storing results and write the first row



import pandas as pd



results = pd.DataFrame(columns=['Correctly classified out of 500', 'Accuracy score', 'F1 score'])

results.loc['Pre-trained ResNet50', :] = [(y_true == y_predicted).sum(), round(accuracy_score(y_true, y_predicted), 2), round(f1_score(y_true, y_predicted), 2)]

results
# print confusion matrix



import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt



class_names = ['hot_dog', 'not_hot_dog']



def print_cm():

    df = pd.DataFrame(confusion_matrix(y_true, y_predicted), index=class_names, columns=class_names)

    sns.heatmap(df, cmap=plt.cm.Blues, annot=True, fmt='d', cbar=False)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()



print_cm()
# print first 10 images not properly classified for further analysis



def print_images():

    images = [(image.array_to_img(images_arr_unprocessed[i]), y_true[i], y_predicted[i]) for i in range(len(y_true)) if y_true[i] != y_predicted[i]]



    print("Class labels: {'hot_dog': 0, 'not_hot_dog': 1}")

    for i in range(10):

        display(images[i][0])

        print('True label: {} Predicted label: {}'.format(int(images[i][1]), int(images[i][2])))



print_images()
# import additional libraries



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
# load pre-trained resnet50 model without top dense layer and 

# add our dense layer for 1 class (since we have binary classification problem)



num_classes = 1



new_model = Sequential()

new_model.add(ResNet50(include_top=False,

                       pooling='avg',

                       weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'))

new_model.add(Dense(num_classes, activation='sigmoid'))

new_model.layers[0].trainable = False
new_model.summary()
# check output shapes of the model if pooling parameter=None



new_model_nopool = Sequential()

new_model_nopool.add(ResNet50(include_top=False,

                              input_shape=(224, 224, 3),

                              pooling=None,

                              weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'))

new_model_nopool.add(Dense(num_classes, activation='sigmoid'))

new_model_nopool.layers[0].trainable = False

new_model_nopool.summary()
# compile the model with adam optimiser and binary_crossentropy loss function

new_model.compile(optimizer='sgd',

                  loss='binary_crossentropy',

                  metrics=['accuracy'])
# train the model on train set for 10 epochs



directory_train = '../input/hot-dog-not-hot-dog/seefood/train'



datagen = image.ImageDataGenerator(preprocessing_function=resnet50.preprocess_input)

image_generator = datagen.flow_from_directory(directory_train,

                                              target_size=(224, 224),

                                              class_mode='binary')

new_model.fit_generator(image_generator,

                        epochs=10,

                        verbose=1)
# test new model

images_arr_unprocessed, images_arr_processed, y_true, y_predicted = test_model(new_model)
# update results table

results.loc['Transfer learning with ResNet50', :] = [(y_true == y_predicted).sum(), round(accuracy_score(y_true, y_predicted), 2), round(f1_score(y_true, y_predicted), 2)]
# print confusion matrix

print_cm()
# print first 10 images not properly classified for further analysis

print_images()
# try data augmentation with same number of epochs and check the results



num_classes = 1



new_model_aug = Sequential()

new_model_aug.add(ResNet50(include_top=False,

                           pooling='avg',

                           weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'))

new_model_aug.add(Dense(num_classes, activation='sigmoid'))

new_model_aug.layers[0].trainable = False

new_model_aug.compile(optimizer='sgd',

                      loss='binary_crossentropy',

                      metrics=['accuracy'])



datagen_aug = image.ImageDataGenerator(preprocessing_function=resnet50.preprocess_input,

                                       rotation_range=30,

                                       width_shift_range=0.1,

                                       height_shift_range=0.1,

                                       horizontal_flip=True)

image_generator = datagen_aug.flow_from_directory(directory_train,

                                                  target_size=(224, 224),

                                                  class_mode='binary',

                                                  seed=0)

new_model_aug.fit_generator(image_generator,

                            epochs=10,

                            verbose=1)



images_arr_unprocessed, images_arr_processed, y_true, y_predicted = test_model(new_model_aug)
results.loc['Transfer learning with ResNet50 and Data augmentation', :] = [(y_true == y_predicted).sum(), round(accuracy_score(y_true, y_predicted), 2), round(f1_score(y_true, y_predicted), 2)]
print_cm()
print_images()
results