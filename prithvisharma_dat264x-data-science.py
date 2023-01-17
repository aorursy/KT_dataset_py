# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import logging

import math

from io import BytesIO

from zipfile import ZipFile

import pandas as pd

import numpy as np

import numpy.random as rnd

from sklearn.preprocessing import LabelBinarizer

from PIL import Image, ImageFilter

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dense
#configuration data 

image_height = 128//1

image_width = 2*176//1

training_image_count = 576

testing_image_count = 384

classes_count = 11
data_root_path = '../input/bijli-wala-project/'

sub = pd.read_csv("../input/bijli-wala-project/submission_format.csv")

sub.to_csv('submission_results.csv', index=False)

sub.head(3)
training_image_data_file_path = data_root_path + 'image_train.data'

training_labels_data_file_path = data_root_path + 'image_train_labels.csv'

testing_data_file_path = data_root_path + 'image_test.data'



testing_submission_file_path = data_root_path + 'submission_format.csv'

submission_results_file_path =  '/kaggle/working/' + 'submission_results.csv'
def compose_train_image(p_img1, p_img2) :

    #Stacks images horizontally (i.e. one afer another on width axis)

    img_merge_data = np.hstack([np.asarray(p_img1), np.asarray(p_img2)])

    img_merge = Image.fromarray( img_merge_data )

        

    return img_merge
def get_image_data(p_image) :

    #Generates image data from the received image object

    width, height = p_image.size

    data = np.asarray(p_image).reshape(height*width)

    

    return data
def create_trainining_images_data_file(p_input_data_file_path, p_training_data_file_path):

    training_labels_file_path = 'train_labels.csv'

    labels = None



    with open(p_training_data_file_path, 'w+b') as data_file :

        with ZipFile(p_input_data_file_path) as data_zip:

            with data_zip.open(training_labels_file_path) as train_labels_file:

                content = train_labels_file.read()

                with BytesIO(content) as io_content:

                    train_labels = pd.read_csv(io_content)

                    max_count = train_labels.shape[0]    

                    labels = np.zeros(max_count)

                    count = 0



                    for _, row in train_labels.iterrows() :

                        with data_zip.open('train/' + str(row["id"]) + "_c.png") as c_file :

                            with BytesIO(c_file.read()) as input_buffer:

                                c_image = Image.open(input_buffer).convert("L")



                        with data_zip.open('train/' + str(row["id"]) + "_v.png") as v_file :

                            with BytesIO(v_file.read()) as input_buffer:

                                v_image = Image.open(input_buffer).convert("L")



                        image_data = get_image_data(compose_train_image(c_image, v_image))

                        labels[count] = row["appliance"]

                        data_file.write(image_data)



                        count = count + 1       



    return labels[:count]
def create_training_labels(p_labels, p_labels_data_file_path) :

    classes = pd.DataFrame(p_labels.astype(int))

    classes.to_csv(p_labels_data_file_path, header=None)



    return
def create_testing_images_data_file(p_input_data_file_path, p_testing_data_file_path):

    submission_format_file_path = 'submission_format.csv'



    with open(p_testing_data_file_path, 'w+b') as data_file :

        with ZipFile(p_input_data_file_path) as data_zip:

            with data_zip.open(submission_format_file_path) as submission_format_file:

                content = submission_format_file.read()

                with BytesIO(content) as io_content:

                    submission_indexes = pd.read_csv(io_content)

                    count = 0



                    for _, row in submission_indexes.iterrows() :

                        with data_zip.open('test/' + str(row["id"]) + "_c.png") as c_file :

                            with BytesIO(c_file.read()) as input_buffer:

                                c_image = Image.open(input_buffer).convert("L")



                        with data_zip.open('test/' + str(row["id"]) + "_v.png") as v_file :

                            with BytesIO(v_file.read()) as input_buffer:

                                v_image = Image.open(input_buffer).convert("L")



                        image_data = get_image_data(compose_train_image(c_image, v_image))

                        data_file.write(image_data)



                        count = count + 1       



    return count
def create_testing_submission(p_input_data_file_path, p_testing_submission_file_path) :

    submission_format_file_path = 'submission_format.csv'



    with ZipFile(p_input_data_file_path) as data_zip:

        with data_zip.open(submission_format_file_path) as submission_format_file:

            content = submission_format_file.read()

            with BytesIO(content) as io_content:

                submission_indexes = pd.read_csv(io_content)

                submission_indexes.to_csv(p_testing_submission_file_path, index=False)



    return
def main() :

    logging.basicConfig(level=logging.INFO)

    

    #create training data

    logging.info('Creating training data ...')

    training_labels  = create_trainining_images_data_file(input_data_file_path, training_image_data_file_path)

    create_training_labels(training_labels, training_labels_data_file_path)

    logging.info("Processed training images count: %d" % training_labels.shape[0])

    logging.info('Creating training data DONE')



    logging.info('Creating testing data ...')

    testing_count = create_testing_images_data_file(input_data_file_path, testing_data_file_path)

    create_testing_submission(input_data_file_path, testing_submission_file_path)

    logging.info("Processed testing images count: %d" % testing_count)

    logging.info('Creating testing data DONE')
def read_image(p_image_data_file_path, p_position, p_image_width, p_image_height) :

    with open(p_image_data_file_path, "rb") as image_file :

        image_file.seek(p_position * p_image_height* p_image_width)

        data = image_file.read(p_image_height * p_image_width)

        data_b = np.frombuffer(data, dtype=np.uint8)



    return np.asarray(data_b)
def process_images(p_images, p_image_width, p_image_height) :



    #reshape according to inputs accepted by a Conv2d layer

    processed_images = p_images.reshape(p_images.shape[0], p_image_height, p_image_width, 1)



    #data normalization to max value (0-255 grayscale values)

    processed_images = (processed_images * 1.0) /255

 

    return processed_images
def read_labels(p_labels_file_path) :

  

    labels = pd.read_csv(p_labels_file_path, header= None)

    labels.columns = ["id", "label"]

  

    return labels
def process_labels(p_labels) :

    processed_labels = LabelBinarizer().fit_transform(p_labels)

    

    return processed_labels
def generate_train_set(

    p_image_training_data_file_path, 

    p_labels_file_path, 

    p_train_set_size, 

    p_image_width, 

    p_image_height

) :

    labels = read_labels(p_labels_file_path)

    

    labels_batch = np.zeros(p_train_set_size)

    labels_batch = labels["label"][0:p_train_set_size].values



    images_batch = []

  

    for i in range(0, p_train_set_size) :

        image_data = read_image(p_image_training_data_file_path, i, p_image_width, p_image_height)

        images_batch.append(image_data.reshape(p_image_height, p_image_width))

  

    train_labels_processed = process_labels(labels_batch)

  

    train_images_processed = process_images(np.array(images_batch), p_image_width, p_image_height)

  

    return train_labels_processed, train_images_processed
def generate_test_set(

    p_test_image_data_file_path, 

    p_test_set_size, 

    p_image_width, 

    p_image_height

) :

    images_batch = []



    for i in range(0, p_test_set_size) :

        image_data = read_image(p_test_image_data_file_path, i, p_image_width, p_image_height)

        images_batch.append(image_data.reshape(p_image_height, p_image_width))



    test_images_processed = process_images(np.array(images_batch), p_image_width, p_image_height)



    return test_images_processed 
def create_model(p_image_width, p_image_height, p_num_classes) :

    input_shape = (p_image_height, p_image_width, 1)



    #we will use a sequential model for training 

    model = Sequential()

	

    #CONV 3x3x32 => RELU => NORMALIZATION => MAX POOL 3x3 block

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))

    model.add(Activation("relu"))

    model.add(BatchNormalization(axis=-1))

    model.add(MaxPooling2D(pool_size=(3, 3)))



    #CONV 3x3x64 => RELU => NORMALIZATION => MAX POOL 2x2 block

    model.add(Conv2D(64, (3, 3), padding="same"))

    model.add(Activation("relu"))

    model.add(BatchNormalization(axis=-1))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    #CONV 3x3x128 => RELU => NORMALIZATION => MAX POOL 2x2 block

    model.add(Conv2D(128, (3, 3), padding="same"))

    model.add(Activation("relu"))

    model.add(BatchNormalization(axis=-1))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    #FLATTEN => DENSE 1024 => RELU => NORMALIZATION block

    model.add(Flatten())

    model.add(Dense(1024))

    model.add(Activation("relu"))

    model.add(BatchNormalization())



    #final DENSE => SOFTMAX block for multi-label classification

    model.add(Dense(p_num_classes))

    model.add(Activation("softmax"))



    #using categorical_crossentropy loss function with adam optimizer

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



    return model
def train_model(

    p_model, 

    p_training_image_data, 

    p_trainging_labels, 

    p_batch_size = 64, 

    p_epochs_to_train = 50, 

    p_verbose_level = 2

) : 

    p_model.fit(

        x = p_training_image_data, 

        y = p_trainging_labels, 

        batch_size = p_batch_size, 

        epochs = p_epochs_to_train,

        shuffle = True,

        verbose = p_verbose_level    

    )

    

    return p_model
def predict_labels(p_model, p_test_image_data, p_batch_size = 32) :

    labels = p_model.predict_classes(p_test_image_data, p_batch_size)

  

    return labels

def write_results(

    p_testing_submission_file_path, 

    p_submission_results_file_path, 

    p_results

) :

    submission_structure = pd.read_csv(p_testing_submission_file_path)

    submission_structure['appliance'] = p_results

    submission_structure.to_csv(p_submission_results_file_path, index=False)
def main():

    logging.basicConfig(level=logging.INFO)

    

    #prepare training data

    logging.info('Reading training data ...')

    train_labels, train_images = generate_train_set(

        training_image_data_file_path, 

        training_labels_data_file_path, 

        training_image_count, 

        image_width, 

        image_height

    )

    logging.info('Reading training data DONE')

    

    #create and train model

    logging.info('Creating model ...')

    model = create_model (image_width, image_height, classes_count)

    logging.info('Creating model DONE')



    logging.info('Training model ... ')

    model = train_model(model, train_images, train_labels, p_epochs_to_train = 50)

    logging.info('Training model DONE')

    

    #create test data

    logging.info('Reading testing data ...')

    test_images = generate_test_set(

      testing_data_file_path, 

      testing_image_count, 

      image_width, 

      image_height

    )

    logging.info('Reading testing data DONE')

    

    #predict labels for test data

    logging.info('Predicting test data classes ...')

    result = predict_labels(model, test_images)

    logging.info('Predicting test data classes DONE')

    

    #write results

    logging.info('Writing results ...')

    write_results(

        testing_submission_file_path, 

        submission_results_file_path, 

        result

    )

    logging.info('Writing results DONE')



if __name__ == '__main__':

    main()