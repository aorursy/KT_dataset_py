# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

TRAIN_IMAGE_PATH = '../input/train/'

TEST_IMAGE_PATH = '../input/test/'

CSV_PATH = '../input/'

OUTPUT_DATA_PATH = './'

WEIGHT_DATA_FILE_NAME = 'model_weights_20190216_00.h5'

SUBMISSIONT_DATA_FILE_NAME = 'sample_submission_20190216_00.csv'
from PIL import Image



import matplotlib.pyplot as plt

%matplotlib inline



from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow.keras as K

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Lambda, MaxPooling2D

from tensorflow.keras.regularizers import l2

from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import Adam



import random

import time
train_df = pd.read_csv(CSV_PATH + 'train.csv')

print('Fist Row\n', train_df.head(1))

print('train_df shape : ', train_df.shape)
unique_id_list = train_df['Id'].unique()

unique_id_list_size = unique_id_list.size

print('Unique ID List Size = ', unique_id_list_size)
ID_images_dict = {}

for index, row in train_df.iterrows():

    (image_file_name, label) = (row[0], row[1])

    ID_images_dict.setdefault(label, []).append(image_file_name)



known_ID_images_dict   = {k: v for (k, v) in ID_images_dict.items() if k != 'new_whale'}
ID_count_dict = train_df['Id'].value_counts(ascending = True).to_dict()



known_ID_count_dict = {k: v for (k, v) in ID_count_dict.items() if k != 'new_whale'}

# print(known_ID_count_dict)

known_ID_count_histogram_df = pd.DataFrame(list(known_ID_count_dict.items()), columns=['ID', 'count'])



value_count = known_ID_count_histogram_df['count'].value_counts()

print('Known whales')

print("# of images\t# of ID's")

print(value_count.head(5))
sorted_known_ID_images_list = sorted(known_ID_images_dict.items(), key = lambda x: len(x[1]))

print(sorted_known_ID_images_list[0:3])
def generate_random_augmented_image(img, nb_images = 100):



    data_generator = ImageDataGenerator(rotation_range     = 10.0, # degree

                                        width_shift_range  = 0.2,

                                        height_shift_range = 0.2,

                                        shear_range        = 5.0, # degree

                                        zoom_range         = 0.2,

#                                         horizontal_flip    = True,

                                        vertical_flip      = True,

                                       )



    img_list = []

    for i in range(nb_images):

        generated_image = data_generator.random_transform(img) 

        img_list.append(generated_image)



    return img_list
img_file_name = TRAIN_IMAGE_PATH + sorted_known_ID_images_list[0][1][0]



IMG_W = int(1050/4) # median of image sizes

IMG_H = int(525/4)

IMG_D = 1



from PIL import Image



def get_image_data(img_file_name):



    with Image.open(img_file_name) as jpg_img:

    

        tmp = jpg_img.resize((IMG_W, IMG_H))

        tmp2d = np.asarray(tmp.convert('L'))/255

        tmp3d = tmp2d.reshape(IMG_H, IMG_W, IMG_D)



    return tmp3d

    

plt.imshow(get_image_data(img_file_name).reshape(IMG_H, IMG_W), cmap = 'gray')
def show_in_grid(img_list, img_height, img_width, grid_shape = (4, 5)):



    (r, c) = grid_shape

    fig, axes = plt.subplots(r, c, figsize = (12, 8))

    

    k = 0

    for i in range(c):

        for j in range(r):

            axes[j, i].matshow(img_list[k].reshape(img_height, img_width), cmap = 'gray')

            axes[j, i].get_yaxis().set_visible(False)

            axes[j, i].get_xaxis().set_visible(False)

            k = k + 1



img_list = generate_random_augmented_image(get_image_data(img_file_name), 100)  

show_in_grid(img_list, IMG_H, IMG_W, (10, 10))
def get_siamese_model(input_shape):



    input_A = Input(input_shape)

    input_B = Input(input_shape)



    conv_net = Sequential()



    # First layer (525, 262)

    conv_net.add(Conv2D(filters = 32, kernel_size = (4, 4), padding  = 'same', activation = 'relu',

                        kernel_initializer = RandomNormal(mean = 0, stddev = 0.01)))

    conv_net.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))



    # Second layer (262, 131)

    conv_net.add(Conv2D(filters = 64, kernel_size = (4, 4), padding  = 'same', activation = 'relu',

                        kernel_initializer = RandomNormal(mean = 0, stddev = 0.01)))

    conv_net.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))

    

    # Third layer (131, 66)

    conv_net.add(Conv2D(filters = 128, kernel_size = (4, 4), padding  = 'same', activation = 'relu',

                        kernel_initializer = RandomNormal(mean = 0, stddev = 0.01)))

    conv_net.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))

        

    conv_net.add(Flatten())

    conv_net.add(Dense(units = 1024, activation = "sigmoid",

                       kernel_initializer = RandomNormal(mean = 0, stddev = 0.01),

                       bias_initializer = RandomNormal(mean = 0.5, stddev = 0.01)))



    #call the convnet Sequential model on each of the input tensors so params will be shared

    encoded_A = conv_net(input_A)

    encoded_B = conv_net(input_B)



    #layer to merge two encoded inputs with the l1 distance between them

    L1_layer = Lambda(lambda tensors:K.backend.abs(tensors[0] - tensors[1]))

    L1_distance = L1_layer([encoded_A, encoded_B])



    prediction = Dense(units = 1, activation = 'sigmoid', bias_initializer = RandomNormal(mean = 0.5, stddev = 0.01))(L1_distance)

    siamese_net = Model(inputs = [input_A, input_B], outputs = prediction)

    optimizer = Adam(0.001)



    siamese_net.compile(loss = "binary_crossentropy", optimizer = optimizer)

    siamese_net.count_params()



    return siamese_net
def get_test_data_pair(ID_images_dict):



    nb_label = 10

    ID_image_list = random.sample(list(ID_images_dict.items()), nb_label)

  

    selected_img_A_list = [] # correct one

    selected_img_B_list = []



    index = random.randint(0, nb_label - 1)

    img_file_A = random.sample(ID_image_list[index][1], 1) # if the img list contains more than 1, choose one randomly

    img_A_src = get_image_data(TRAIN_IMAGE_PATH + img_file_A[0])

#     print('shape = ', img_A.shape)

    img_A_list = generate_random_augmented_image(img_A_src, nb_images = 1)

    img_A = img_A_list[0].reshape(IMG_H, IMG_W, IMG_D)

    for i in range(nb_label):

        selected_img_A_list.append(img_A)    

    

    for i in range(nb_label):

        img_file_B = random.sample(ID_image_list[i][1], 1)

        img_B_src = get_image_data(TRAIN_IMAGE_PATH + img_file_B[0])

        img_B_list = generate_random_augmented_image(img_B_src, nb_images = 1)

        img_B = img_B_list[0].reshape(IMG_H, IMG_W, IMG_D)

        selected_img_B_list.append(img_B)



    # diff class : target = 0, same class : target = 1

    target = np.zeros(nb_label)

    target[index] = 1



    

    return (selected_img_A_list, selected_img_B_list), target
def get_train_data_pair(ID_images_dict, sample_size = 100):



    nb_label = 3

    ID_image_list = random.sample(list(ID_images_dict.items()), nb_label)

    

    target_diff = np.zeros(sample_size)

    target_same = np.ones(sample_size)



    target = np.concatenate([target_diff, target_same])



    label_diff_A = ID_image_list[0][0]

    label_diff_B = ID_image_list[1][0]

    label_same_A = ID_image_list[2][0] # make same data set

    label_same_B = ID_image_list[2][0] # make same data set



    img_src_name_diff_A = random.sample(ID_image_list[0][1], 1) # pick up one image from list of images

    img_src_name_diff_B = random.sample(ID_image_list[1][1], 1) # for image augmentation

    img_src_name_same_A = random.sample(ID_image_list[2][1], 1)

    img_src_name_same_B = random.sample(ID_image_list[2][1], 1)



    img_src_diff_A = get_image_data(TRAIN_IMAGE_PATH + img_src_name_diff_A[0])

    img_src_diff_B = get_image_data(TRAIN_IMAGE_PATH + img_src_name_diff_B[0])

    img_src_same_A = get_image_data(TRAIN_IMAGE_PATH + img_src_name_same_A[0])

    img_src_same_B = get_image_data(TRAIN_IMAGE_PATH + img_src_name_same_B[0])





    augmented_img_diff_A_list = generate_random_augmented_image(img_src_diff_A, nb_images = sample_size)

    augmented_img_diff_B_list = generate_random_augmented_image(img_src_diff_B, nb_images = sample_size)

    augmented_img_same_A_list = generate_random_augmented_image(img_src_same_A, nb_images = sample_size)

    augmented_img_same_B_list = generate_random_augmented_image(img_src_same_B, nb_images = sample_size)

    

    A = np.concatenate([augmented_img_diff_A_list, augmented_img_same_A_list])

    B = np.concatenate([augmented_img_diff_B_list, augmented_img_same_B_list])



    return (A, B), target
def test_oneshot(model, ID_images_dict, nb_validation):



    nb_correct = 0

    for i in range(nb_validation):



        (inputs, targets) = get_test_data_pair(ID_images_dict)

        probabilites = model.predict(inputs)

    

        if np.argmax(probabilites) == np.argmax(targets):

            nb_correct += 1



    accuracy = nb_correct / nb_validation



    return accuracy
print('Model Building Started')

input_shape = (IMG_H, IMG_W, IMG_D)

siamese_net = get_siamese_model(input_shape)

siamese_net.summary()

optimizer = Adam(lr = 0.00006)

siamese_net.compile(loss = "binary_crossentropy", optimizer = optimizer)

print('Model Building Finished')
print('Training Loop Started')



start = time.time()



nb_iter = 10000

tmp_accuracy = -1

evaluation_interval = 10



for i in range(nb_iter):

    

    (train_img_pair, target) = get_train_data_pair(ID_images_dict = known_ID_images_dict, sample_size = 50)

    loss = siamese_net.train_on_batch(train_img_pair, target)

 

    if i % (evaluation_interval * 10) == 0:

        print('Loop = ', i, 'time = ', time.time() - start)



    if i % evaluation_interval == 0:

#         print('Loop = ', i, 'time = ', time.time() - start)

        accuracy = test_oneshot(siamese_net, ID_images_dict = known_ID_images_dict, nb_validation = 20)



        if accuracy >= tmp_accuracy:

            print("Current accuracy : {:.2f}, Previous accuracy : {:.2f}".format(accuracy, tmp_accuracy))

            tmp_accuracy = accuracy

    

    siamese_net.save_weights(OUTPUT_DATA_PATH + WEIGHT_DATA_FILE_NAME)



print('Training Loop Finished')

print(time.time() - start)
submission_df = pd.read_csv(CSV_PATH + 'sample_submission.csv')



print(submission_df.head(3))



submission_img_file_list = submission_df['Image'].values.tolist()

print('\nLength of submission_img_file_list =', len(submission_img_file_list))
img_list = []



for fname in submission_img_file_list[0:10]:



    img_file_name = TEST_IMAGE_PATH + fname

    img_list.append(get_image_data(img_file_name))



show_in_grid(img_list, IMG_H, IMG_W, grid_shape = (2, 5))
sub_ID_images_list = [] #Number of unique ID = 5005

len_sub_dict = 50



def split_test_dict(known_ID_images_dict, unique_id_list, len_sub_dict = 50):

    

    tmp_dict = {}

    for i, (key, value) in enumerate(known_ID_images_dict.items()):



        tmp_dict[key] = value

        if (( (i + 1) % len_sub_dict) == 0 or i == (unique_id_list.size - 2)): # mod

            sub_ID_images_list.append(tmp_dict)

            tmp_dict = {}



    list_length = len(sub_ID_images_list)

    sub_ID_images_list[list_length - 2].update(sub_ID_images_list[list_length - 1])

    sub_ID_images_list[list_length - 1].clear()



split_test_dict(known_ID_images_dict, unique_id_list, len_sub_dict)

# print(sub_ID_images_list[list_length - 2])
# Length of submission_img_file_list = 7960

def get_data_pair(sub_known_ID_images_dict, submission_img_file_name):



    img_A_list = [] # test data

    img_B_list = [] # train data



    list_length = len(sub_known_ID_images_dict)

    

    img_A = get_image_data(TEST_IMAGE_PATH + submission_img_file_name)

    for i in range(list_length):

        img_A_list.append(img_A)    



    for i, (key, value) in enumerate(sub_known_ID_images_dict.items()):

        img_file_name = random.sample(value, 1)[0]

        img_B = get_image_data(TRAIN_IMAGE_PATH + img_file_name)

        img_B_list.append(img_B)



    return (img_A_list, img_B_list)
nb_sub_dict = len(sub_ID_images_list) - 2 # dict
import csv



def write_to_submission_file(Image = 'Image', Id = 'Id', mode = 'w'):



    with open(OUTPUT_DATA_PATH + SUBMISSIONT_DATA_FILE_NAME, mode) as f:

        

        writer = csv.writer(f, lineterminator='\n')

        

        csv_list = []

        csv_list.append(Image)

        csv_list.append(Id)

        

        writer.writerow(csv_list)
write_to_submission_file()



for sub_fname in submission_img_file_list:



    max_probability = -1

    max_pos = 0

#     for i in range(nb_sub_dict + 1):

    for i in range(2):



        img_pair_list = get_data_pair(sub_ID_images_list[i], sub_fname)

        tmp_prob = siamese_net.predict(img_pair_list)

        sub_max_pos = np.argmax(tmp_prob)



        if max_probability <= tmp_prob[sub_max_pos][0]:

            max_pos = i * len_sub_dict + sub_max_pos

            print(i, max_pos, sub_max_pos, tmp_prob[sub_max_pos][0])



    print("best match :", sub_fname, " = ", unique_id_list[max_pos])

    write_to_submission_file(sub_fname, unique_id_list[max_pos], 'a')



    print("done!")



print("all done!")