#

# Install required packages if any

#

!pip install swifter
#

# Import required packages

#

import os 

import multiprocessing

from joblib import Parallel, delayed

import urllib.request



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from PIL import Image

import cv2



import keras

from keras.utils import to_categorical

import tensorflow as tf



from subprocess import check_output

from sklearn.model_selection import train_test_split



# Check input folder

print(check_output(["ls", "../input"]).decode("utf8"))
#

# Read train and test datasets

#

data_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

data_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

train_rows_count = len(data_train.index)

test_rows_count = len(data_test.index)

print(f"data train rows count: {train_rows_count} , data test rows count: {test_rows_count}")



data_train.head(2)
#

# Utility classes

#

class ImageUtil:

    #

    # download images function

    #

    @staticmethod 

    def get_image_path(images_dir, url):

        file_name = url.split("/")[-1]

        file_path = f'{images_dir}/{file_name}'

        return file_path

    

    @staticmethod    

    def download(images_dir, url):        

        file_path = ImageUtil.get_image_path(images_dir, url)

        if not os.path.exists(file_path):            

            urllib.request.urlretrieve(url, file_path)

            print(f'{file_path} downloaded.')

        return file_path

    

    @staticmethod

    def open(file_path):

        img = cv2.imread(file_path)

        return img

    

    @staticmethod

    def resize(source_image, width, height):

        img = cv2.resize(np.array(source_image), (width, height))

        return img

    

    @staticmethod

    def resize_save(image_path, width, height):

        img = ImageUtil.open(image_path)    

        img_resized = ImageUtil.resize(img, width, height)

        cv2.imwrite(image_path, img_resized)

    

    @staticmethod

    def cvt_color(image):

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    

    @staticmethod

    def grid_display(list_of_images, list_of_titles=[], no_of_columns=2, figsize=(10,10), fig_title = None):

        fig = plt.figure(figsize=figsize)

        if fig_title is not None:

            fig.suptitle(fig_title, fontsize=16)

        column = 0

        title_error = 0

        for i in range(len(list_of_images)):

            column += 1

            #  check for end of column and create a new figure

            if column == no_of_columns+1:

                fig = plt.figure(figsize=figsize)

                column = 1

            fig.add_subplot(1, no_of_columns, column)

            plt.imshow(ImageUtil.cvt_color(list_of_images[i]) , cmap="gray")

            plt.axis('off')

            if len(list_of_titles) >= len(list_of_images):

                plt.title(list_of_titles[i])

            elif title_error == 0:

                print('number of titles and images are not the same!')

                title_error = 1

print('ImageUtil defined')
#

# Load Images

#

compl_category_dict = {

    0: "T-shirt/top",

    1: "Trouser",

    2: "Pullover",

    3: "Dress",

    4: "Coat",

    5: "Sandal",

    6: "Shirt",

    7: "Sneaker",

    8: "Bag",

    9: "Ankle boot"

}

img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)



# read training data

data_train_X = np.array(data_train.iloc[:, 1:])

data_train_X = data_train_X.reshape(data_train_X.shape[0], img_rows, img_cols, 1)

data_train_X = data_train_X.astype('float32')

data_train_X /= 255

data_train_y = data_train.iloc[:, 0]



print(f"data train X rows: {len(data_train_X)} shape: {data_train_X.shape} ")

print(f"data train y rows {len(data_train_y)} shape : {data_train_y.shape}")



# read test data

data_test_X = np.array(data_test.iloc[:, 1:])

data_test_X = data_test_X.reshape(data_test_X.shape[0], img_rows, img_cols, 1)

data_test_X = data_test_X.astype('float32')

data_test_X /= 255

data_test_y = data_test.iloc[:, 0]



print(f"data test X rows: {len(data_test_X)} shape: {data_test_X.shape} ")

print(f"data test y rows {len(data_test_y)} shape : {data_test_y.shape}")
#

# Images Spot check

#

def spot_check_array(source_array, title = None, n = 10, no_of_columns =5, figsize=(10, 10), data_seed = 123):

    np.random.seed(data_seed)

    random_images = []

    random_labels = []

    for index in  np.random.randint(len(source_array), size = n):    

        random_images.append(data_X[index])

        random_labels.append(compl_category_dict[data_y[index]])        

    ImageUtil.grid_display(random_images, random_labels, no_of_columns=no_of_columns, figsize=figsize, fig_title = title)



def spot_check_dataset(source_dataset, title = None, n = 5, no_of_columns = 2, figsize=(5, 5), data_seed = 123):

    np.random.seed(data_seed)

    random_images = []

    random_labels = []

    for index in  np.random.randint(len(source_dataset.index), size = n):    

        row = source_dataset.iloc[index]

        random_images.append(row["image1"])

        random_images.append(row["image2"])          

        

        cat1 = compl_category_dict[row['category1']]

        cat2 = compl_category_dict[row['category2']]    

        cls = compl_class_dict[row["class"]]            

        random_labels.append(f"{cat1}: {cls}")

        random_labels.append(f"{cat2}: {cls}")        

    ImageUtil.grid_display(random_images, random_labels, no_of_columns=no_of_columns, figsize=figsize, fig_title = title)
spot_check_array(data_train_X, n = 10)

spot_check_array(data_test_X, n = 10)
#

# Generate fasion complementary items

# ---------------------------------------------------------------

# We assume an arbitary categories as followings are complementary

# For example T-shirt and Trouser are complementary, but T-shirt and Pullover are not complementary

#

#    0: "T-shirt/top" --> [1: "Trouser", 5: "Sandal", 7: "Sneaker"]

#    1: "Trouser",    --> [0: "T-shirt/top", 2: "Pullover", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 9: "Ankle Boot"]

#    2: "Pullover",   --> [1: "Trouser", 5: "Sandal", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"]

#    3: "Dress",      --> [5: "Sandal", 8: "Bag", 9: "Ankle boot"]

#    4: "Coat",       --> [1: "Trouser", 8: "Bag", 9: "Ankle boot"]

#    5: "Sandal",     --> [0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 6: "Shirt"]

#    6: "Shirt",      --> [1: "Trouser", 5: "Sandal", 7: "Sneaker", 9: "Ankle boot"]

#    7: "Sneaker",    --> [0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 6: "Shirt"]

#    8: "Bag",        --> [2: "Pullover", 3: "Dress", 4: "Coat", 9: "Ankle Boot"]

#    9: "Ankle boot"  --> [1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 6: "Shirt", 8: "Bag"]

#

#   Now, We randomly pick two items pair form data set, if they are complementary then class is 0: Complementary otherwise 1: Non-Complementary

#

compl_categories = {

    0: [1, 5, 7],

    1: [0, 2, 4, 5, 6, 7, 9],

    2: [1, 5, 7, 8, 9],

    3: [5, 8, 9],

    4: [1, 8, 9],

    5: [0, 1, 2, 3, 6],

    6: [1, 5, 7, 9],

    7: [0, 1, 2, 6],

    8: [2, 3, 4, 9],

    9: [1, 2, 3, 4, 6, 8]    

}

compl_class_dict = {

    0 : "Rand",

    1 : "Comp"

}



def gen_compl_dataset(data_source_X, data_source_y):

    source_dataset = pd.DataFrame()

    source_dataset["image"] = [i for i in data_source_X]

    source_dataset["category"] = data_source_y

    

    random_source_dataset1 = source_dataset.sample(frac = 1, random_state=1234).copy(deep = True)

    random_source_dataset2 = source_dataset.sample(frac = 1, random_state=5678).copy(deep = True)

    counter = 0

    source_data_rows = []

    for i, row1 in random_source_dataset1.iterrows():

        row2 = random_source_dataset2.iloc[i]

        image1 = row1["image"]

        cat1 = row1["category"]    

        image2 = row2["image"]

        cat2 = row2["category"]    

        cat1_compl_cats = compl_categories[cat1]

        compl_class = 0 # 0: Non-Complementary, 1 : Complementary

        if (cat2 in cat1_compl_cats):

            compl_class = 1



        source_data_rows.append([image1, cat1, image2, cat2, compl_class])    

        counter += 1

        if counter % 10000 == 0:                

            print(f"{counter} example :: index={i}  cat1={cat1}:{compl_category_dict[cat1]}, cat2={cat2}:{compl_category_dict[cat2]} -> {compl_class}:{compl_class_dict[compl_class]}")

            print(f"{counter} example :: cat: {cat1} -> compls: {cat1_compl_cats}")

            print()

        

    source_dataset = None

    random_source_dataset1 = None

    random_source_dataset2 = None

    result_dataset = pd.DataFrame(source_data_rows, columns = ["image1", "category1", "image2", "category2", "class"])

    return result_dataset

  

def print_dataset_sum(dataset_name, dataset):    

    class_0_rows = dataset[dataset["class"] == 0]

    class_1_rows = dataset[dataset["class"] == 1]

    class_0_rows_count = len(class_0_rows.index)

    class_1_rows_count = len(class_1_rows.index)

    class_0_rows = None

    class_1_rows = None

    print("----------------------------------")

    print(f"{dataset_name} dataset rows count: {class_0_rows_count + class_1_rows_count} [Complementary: {class_0_rows_count}, Non-Complementary: {class_1_rows_count}]")

    print("----------------------------------")

    print()

    

train_compl_dataset = gen_compl_dataset(data_train_X, data_train_y)

print_dataset_sum("Train Complement", train_compl_dataset)



test_compl_dataset = gen_compl_dataset(data_test_X, data_test_y)

print_dataset_sum("Test Complement", test_compl_dataset)
#

# Train/Test dateset images Spot check

#

spot_check_dataset(train_compl_dataset, n = 4, data_seed = 23)

spot_check_dataset(test_compl_dataset, n = 3, data_seed = 45)
#

# Split train and val data

#

def train_val_split(dataset, val_size = 0.10):

    X_train, X_val, y_train, y_val = train_test_split(dataset[['image1','image2', "category1", "category2"]], \

                                                          dataset['class'], test_size = val_size)

    # train

    X1_train_temp = X_train['image1'].to_numpy()

    X1_train = np.stack(X1_train_temp[0:len(X1_train_temp)])    

    

    Cat1_train_temp = X_train['category1'].to_numpy()

    Cat1_train = np.stack(Cat1_train_temp[0:len(Cat1_train_temp)])    



    X2_train_temp = X_train['image2'].to_numpy()

    X2_train = np.stack(X2_train_temp[0:len(X2_train_temp)])    

    

    Cat2_train_temp = X_train['category2'].to_numpy()

    Cat2_train = np.stack(Cat2_train_temp[0:len(Cat2_train_temp)])  

    

    y_train = np.array(y_train)



    # validation

    X1_val_temp = X_val['image1'].to_numpy()

    X1_val = np.stack(X1_val_temp[0:len(X1_val_temp)])   

    

    Cat1_val_temp = X_val['category1'].to_numpy()

    Cat1_val = np.stack(Cat1_val_temp[0:len(Cat1_val_temp)])   



    X2_val_temp = X_val['image2'].to_numpy()

    X2_val = np.stack(X2_val_temp[0:len(X2_val_temp)])    

    

    Cat2_val_temp = X_val['category2'].to_numpy()

    Cat2_val = np.stack(Cat2_val_temp[0:len(Cat2_val_temp)])

    

    y_val = np.array(y_val)



    return X1_train, X1_val, X2_train, X2_val, Cat1_train, Cat1_val, Cat2_train, Cat2_val, y_train, y_val

print('Train_val_split defined')
X1_train, X1_val, X2_train, X2_val, Cat1_train, Cat1_val, Cat2_train, Cat2_val, y_train, y_val  = train_val_split(train_compl_dataset)



y_train = to_categorical(y_train)

y_val = to_categorical(y_val)



print(X1_train.shape)

print(X1_val.shape)

print(X2_train.shape)

print(X2_val.shape)

print(Cat1_train.shape)

print(Cat1_val.shape)

print(Cat2_train.shape)

print(Cat2_val.shape)

print(y_train.shape)

print(y_val.shape)



print("Data split succeed")

# spot check

for i in range(0, 5):

    print(Cat1_train[i], Cat2_train[i], y_train[i])

    print(Cat1_val[i], Cat2_val[i], y_val[i])
#

# Siamese Network Model

#

from tensorflow.keras.regularizers import l2

import tensorflow.keras.backend as K

from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Lambda,

                                     Dense, Dropout, BatchNormalization, concatenate, LeakyReLU)

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import regularizers, Model

from tensorflow.keras.models import model_from_json



from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.backend import flatten

import keras as k



initialize_weights = k.initializers.RandomNormal(mean=0.0, stddev=0.51, seed=50001)

initialize_bias = k.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=1221)



def get_compiled_model(input_shape):

    """        

    Model architecture

    https://github.com/eroj333/learning-cv-ml/blob/master/SNN/Learning%20Similarity%20Function.ipynb

    """       



    # Define the tensors for the two input images

    left_input = Input(input_shape)

    right_input = Input(input_shape)    



    model = Sequential()

    model.add(Conv2D(32, (5,5), activation='relu', input_shape=input_shape,

                   kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))

    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3,3), activation='relu',

                     kernel_initializer=initialize_weights,

                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))

    model.add(MaxPooling2D())

    model.add(Conv2D(100, (1,1), activation='relu', kernel_initializer=initialize_weights,

                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))

    model.add(Flatten())

    model.add(Dense(1024, activation='sigmoid', 

                   kernel_regularizer=l2(1e-3),

                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))

    model.add(Dropout(rate=.05))    

    print(model.output_shape)

    



    # Generate the encodings (feature vectors) for the two images

    encoded_l = model(left_input)

    encoded_r = model(right_input)



    # Add a customized layer to compute the absolute difference between the encodings

    L1_layer = Lambda(lambda tensors:k.backend.abs(tensors[0] - tensors[1]))

    L1_distance = L1_layer([encoded_l, encoded_r])

    

    # Add a dense layer with a sigmoid unit to generate the similarity score

    prediction = Dense(2, activation='sigmoid', bias_initializer = initialize_bias)(L1_distance)

    

    # Connect the inputs with the outputs

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)



    # compile the model   

    loss_function = 'binary_crossentropy' #'sparse_categorical_crossentropy'     

    optimizer = Adam()

    siamese_net.compile(loss=loss_function,

                        optimizer=optimizer,

                        metrics=['accuracy'])



    # return the model

    return siamese_net



print('get_compiled_model defined')
#

# create model 

#

input_shape = (img_rows, img_cols, 1)

model = get_compiled_model(input_shape)

model.summary()
#

# Train the model

#

summary = model.fit([X1_train, X2_train], y_train,

         batch_size=64,

         epochs=20,

         validation_data=([X1_val, X2_val], y_val),

         shuffle=True)



print('Training Done')
#

# Confusion Matrix

#

#

# confusion matrix

#

from sklearn.metrics import confusion_matrix

import itertools



#

# Look at confusion matrix 

# Note, this code is taken straight from the SKLEARN website, an nice way of viewing confusion matrix.

#

def plot_confusion_matrix(cm, 

                          classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')





yy_val = np.array(y_val)

#yy_val = yy_val.reshape(-1, 1)

#enc.fit([[0], [1]])

#yy_val = enc.transform(yy_val).toarray()



Y_pred = model.predict([X1_val, X2_val])

YY_pred = np.zeros_like(Y_pred)

YY_pred[np.arange(len(Y_pred)), Y_pred.argmax(1)] = 1



Y_true = np.argmax(yy_val, axis = 1)

Y_pred_classes = np.argmax(YY_pred, axis = 1)



confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 



# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(2))
#

# Show what are most important Errors

#

def get_errors(errors_index, img1_errors, img2_errors, \

               cat1_errors, cat2_errors, \

               pred_errors, obs_errors, nrows = 10):

    """ This function shows images with their predicted and real labels"""

    images = []

    titles = []

    n = 0    

    for row in range(nrows):        

        error = errors_index[n]

        pred_label = compl_class_dict[pred_errors[error]]

        true_label = compl_class_dict[obs_errors[error]]        

        img1 = (img1_errors[error]).reshape((img_rows, img_cols, 1))

        img2 = (img2_errors[error]).reshape((img_rows, img_cols, 1))

        cat1 = cat1_errors[error]

        cat2 = cat2_errors[error]

        cat1_lbl = compl_category_dict[cat1]

        cat2_lbl = compl_category_dict[cat2]

        

        title1 = f"{cat1_lbl}\nPred: {pred_label}\nTrue: {true_label}";

        title2 = f"{cat2_lbl}\nPred: {pred_label}\nTrue: {true_label}";

        images.append(img1)

        images.append(img2)

        titles.append(title1)

        titles.append(title2)

        n += 1        

    return (images, titles)





errors = (Y_pred_classes - Y_true != 0)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

           

# Probabilities of the wrong predicted numbers

Y_pred_errors_prob = np.max(Y_pred_errors, axis = 1)

# Predicted probabilities of the true values in the error set

true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors



# Sorted list of the delta prob errors

sorted_delta_errors = np.argsort(delta_pred_true_errors)



# Top 20 errors 

most_important_errors = sorted_delta_errors[-20:]

less_important_errors = sorted_delta_errors[0:20]



print("most_important_errors")

print(delta_pred_true_errors[most_important_errors[-10:]])

print()

print("less_important_errors")

print(delta_pred_true_errors[less_important_errors[0:10]])
#

# Show  top 20 max errors

#

X1_val_errors = X1_val[errors]

X2_val_errors = X2_val[errors]

Cat1_val_errors = Cat1_val[errors]

Cat2_val_errors = Cat2_val[errors]



error_images, error_titles = get_errors(most_important_errors, X1_val_errors, X2_val_errors, \

                                        Cat1_val_errors, Cat2_val_errors, \

                                        Y_pred_classes_errors, Y_true_errors)

ImageUtil.grid_display(error_images, error_titles, figsize=(5,5))
#

# Show top 20 min errors

#

error_images, error_titles = get_errors(less_important_errors, X1_val_errors, X2_val_errors, \

                                        Cat1_val_errors, Cat2_val_errors, \

                                        Y_pred_classes_errors, Y_true_errors)

ImageUtil.grid_display(error_images, error_titles, figsize=(5,5))