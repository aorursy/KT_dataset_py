import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns #visualization

%matplotlib inline



np.random.seed(42)



import tensorflow as tf



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, EarlyStopping



sns.set(style='white', context='notebook', palette='pastel')



IMSIZE = 28

NUM_CLASSES = 10

BATCH_SIZE = 32
import os

from keras.backend.tensorflow_backend import set_session



#os.environ["CUDA_VISIBLE_DEVICES"]="-1" #hide GPUs, to apply changes restart Kernel



# do not change the following lines

config = tf.ConfigProto()

config.gpu_options.allow_growth = True #allows dynamic memory alloc growth on GPUs

sess = tf.Session(config=config)

set_session(sess) #Keras always uses global TF-Session "sess", so this line is not obligatory

# do not change the above lines
from tensorflow.python.client import device_lib



def get_available_gpus():

    local_device_protos = device_lib.list_local_devices()

    print(local_device_protos)

    return [x.name for x in local_device_protos if x.device_type == "GPU"]



num_gpu = len(get_available_gpus())

print("Number of available GPUs: {}".format(num_gpu))
#print(os.listdir("../input"))

print(os.listdir("../input/dat18mnist-deep/"))



#mnist_path = "../input/mnist-fhj/"

#mnist_train_path = "../input/mnist-fhj/train/"

#mnist_test_path = "../input/mnist-fhj/test/"

mnist_path = "../input/dat18mnist-deep/"

mnist_train_path = "../input/dat18mnist-deep/train/train/"

mnist_test_path = "../input/dat18mnist-deep/test/test/"

mnist_extension = ".jpg"



train_ids_all = pd.read_csv(mnist_path+"train.csv")

test_ids_all = pd.read_csv(mnist_path+"test.csv")



def plot_diag_hist(dataframe, title='NoTitle'):

    f, ax = plt.subplots(figsize=(7, 4))

    ax = sns.countplot(x="label", data=dataframe, palette="GnBu_d")

    sns.despine()

    plt.title(title)

    plt.show()



plot_diag_hist(train_ids_all, title="Labels Training Data")



print("Shape of Training Data: {}".format(train_ids_all.shape))

print("Shape of Test Data: {}\n".format(test_ids_all.shape))



def get_full_path_train(idcode):

    return "{}{}{}".format(mnist_train_path,idcode,mnist_extension)



def get_full_path_test(idcode):

    return "{}{}{}".format(mnist_test_path,idcode,mnist_extension)





train_ids_all["path"] = train_ids_all["id_code"].apply(lambda x: get_full_path_train(x))

test_ids_all["path"] = test_ids_all["id_code"].apply(lambda x: get_full_path_test(x))
train_ids_all.head()
test_ids_all.head()
import cv2



def load_image(image_path):

    #print(image_path)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    return img



def load_images_as_tensor(image_path, dtype=np.uint8):

    data = load_image(image_path).reshape((IMSIZE*IMSIZE,1))

    return data.flatten()



def show_image(image_path, figsize=None, title=None):

    image = load_image(image_path)

    if figsize is not None:

        fig = plt.figure(figsize=figsize)

    if image.ndim == 1:

        plt.imshow(np.reshape(image, (IMSIZE,-1)),cmap='gray')

    elif image.ndim == 2:

        plt.imshow(image,cmap='gray')

    elif image.ndim == 3:

        if image.shape[2] == 1:

            image = image[:,:,0]

            plt.imshow(image,cmap='gray')

        elif image.shape[2] == 3:

            plt.imshow(image)

        else:

            print("Invalid image dimension")

    if title is not None:

        plt.title(title)

        

def show_image_tensor(image, figsize=None, title=None):

    if figsize is not None:

        fig = plt.figure(figsize=figsize)

    if image.ndim == 1:

        plt.imshow(np.reshape(image, (IMSIZE,-1)),cmap='gray')

    elif image.ndim == 2:

        plt.imshow(image,cmap='gray')

    elif image.ndim == 3:

        if image.shape[2] == 1:

            image = image[:,:,0]

            plt.imshow(image,cmap='gray')

        elif image.shape[2] == 3:

            plt.imshow(image)

        else:

            print("Invalid image dimension")

    if title is not None:

        plt.title(title)

        

def show_Nimages(image_filenames, classifications, scale=1):

    N=len(image_filenames)

    fig = plt.figure(figsize=(25/scale, 16/scale))

    for i in range(N):

        ax = fig.add_subplot(1, N, i + 1, xticks=[], yticks=[])

        show_image(image_filenames[i], title="C:{}".format(classifications[i]))

        

def show_Nrandomimages(N=10):

    indices = (np.random.rand(N)*train_ids_all.shape[0]).astype(int)

    show_Nimages(train_ids_all["path"][indices].values, train_ids_all["label"][indices].values)

    

def show_Nimages_of_class(classification=0, N=10):

    indices = train_ids_all[train_ids_all["label"] == classification].sample(N).index

    show_Nimages(train_ids_all["path"][indices].values, train_ids_all["label"][indices].values)

    

def show_Nerrorimages(imgs, pred, true, delta_prob=[], scale=1):

    N=len(imgs)

    fig = plt.figure(figsize=(25/scale, 16/scale))

    for i in range(N):

        ax = fig.add_subplot(1, N, i + 1, xticks=[], yticks=[])

        if (delta_prob!=[]):

            show_image_tensor(imgs[i], title="P:{} T:{} d:{:.2f}".format(pred[i], true[i], delta_prob[i]))

        else:

            show_image_tensor(imgs[i], title="P:{} T:{}".format(pred[i], true[i]))
test_index = 2477

show_image(train_ids_all["path"][test_index], title="Class = {}".format(train_ids_all["label"][test_index]))
show_Nrandomimages(10)
show_Nimages_of_class(classification=2)
train_df, validation_df = train_test_split(train_ids_all, test_size=0.05)
from tqdm import tqdm



def load_training_data(image_filenames):

    N = image_filenames.shape[0]

    train_X = np.zeros((N,IMSIZE*IMSIZE), dtype=np.float32)

    for i in tqdm(range(image_filenames.shape[0])):

        img = load_images_as_tensor(image_filenames.iloc[i])

        train_X[i, :] = np.array(img, np.float32)/255

    return train_X
train_X = load_training_data(train_df["path"])

train_y = train_df["label"].values

validation_X = load_training_data(validation_df["path"])

validation_y = validation_df["label"].values

test_X = load_training_data(test_ids_all["path"])
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

print(train_y.shape)

train_y_cat = to_categorical(train_y, num_classes = 10)

print(train_y_cat.shape)



print(validation_y.shape)

validation_y_cat = to_categorical(validation_y, num_classes = 10)

print(validation_y_cat.shape)
def plot_nice_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8,8))

    sns.heatmap(cm, annot=True, fmt='d', linewidths=.5,  cbar=False, ax=ax, cmap=plt.cm.copper)

    plt.ylabel('true label')

    plt.xlabel('predicted label')
def build_multilayer_perceptron_1L():

    model = Sequential()

    layer_1 = Dense(10, activation="softmax", input_shape=(784,))

    model.add(layer_1)

    return model
from keras.utils import plot_model



model = build_multilayer_perceptron_1L()

plot_model(model, to_file="mymodel.png", show_shapes=True, show_layer_names=False)

model.summary()
# Define the optimizer

my_optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model

model.compile(optimizer = my_optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

model.summary()
#Fit the model - Version 1 without generator

history = model.fit(train_X, train_y_cat, batch_size=128, validation_data=(validation_X, validation_y_cat), epochs=30)
def plot_training_history(history):

    history_df = pd.DataFrame(history.history)

    f = plt.figure(figsize=(25,5))

    ax = f.add_subplot(121)

    ax.plot(history_df["loss"], label="loss")

    ax.plot(history_df["val_loss"], label = "val_loss")

    ax.legend()

    ax = f.add_subplot(122)

    #ax.plot(history_df["acc"], label="acc")

    ax.plot(history_df["accuracy"], label="acc")

    #ax.plot(history_df["val_acc"], label="val_acc")

    ax.plot(history_df["val_accuracy"], label="val_acc")

    ax.legend()
plot_training_history(history)

validation_y_pred_proba = model.predict(validation_X)

validation_y_pred = np.argmax(validation_y_pred_proba, axis=1)

plot_nice_confusion_matrix(validation_y, validation_y_pred)
# Errors are difference between predicted labels and true labels

errors = (validation_y_pred - validation_y != 0)

# Select only erroneous samples, the predicted class and the true class

error_y_pred = validation_y_pred[errors]

error_y_true = validation_y[errors]

error_X = validation_X[errors]



# Get probability vectors of misclassified samples

error_y_probabilities = validation_y_pred_proba[errors]

# Probabilities of the wrong predicted numbers

prob_wrong_prediction = np.max(error_y_probabilities,axis = 1)

# Predicted probabilities of the true values in the error set

prob_true_prediction = np.diagonal(np.take(error_y_probabilities, error_y_true, axis=1))

# Difference between the probability of the predicted label and the true label

delta_errors = prob_wrong_prediction - prob_true_prediction



delta_errors_sorted = np.argsort(delta_errors)



most_important_errors = delta_errors_sorted[:15]



show_Nerrorimages(error_X[most_important_errors], 

                 pred=error_y_pred[most_important_errors], 

                 true=error_y_true[most_important_errors],

                 delta_prob=delta_errors[most_important_errors])
test_y_pred_proba = model.predict(test_X)

test_y_pred = np.argmax(test_y_pred_proba, axis=1)



nn_results = pd.Series(test_y_pred,name="label")

submission = pd.concat([test_ids_all["id_code"],nn_results], axis = 1)

submission.to_csv("nn_submission.csv",index=False)