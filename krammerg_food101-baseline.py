import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns #visualization

%matplotlib inline



np.random.seed(42)



import tensorflow as tf



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

import itertools



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, EarlyStopping



sns.set(style='white', context='notebook', palette='pastel')



IMSIZE = 224

BATCH_SIZE = 16
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
print(os.listdir("../input/dat18seefood"))



base_path = "../input/dat18seefood/"

train_path = "../input/dat18seefood/train/"

test_path = "../input/dat18seefood/test/"

extension = ".jpg"



train_ids_all = pd.read_csv(base_path+"train.csv")

test_ids_all = pd.read_csv(base_path+"test.csv")



labelnames = pd.read_csv(base_path+"labelnames.csv")



def plot_diag_hist(dataframe, title='NoTitle'):

    f, ax = plt.subplots(figsize=(15, 4))

    ax = sns.countplot(x="label", data=dataframe, palette="GnBu_d")

    sns.despine()

    plt.title(title)

    plt.show()



plot_diag_hist(train_ids_all, title="Labels Training Data")



print("Shape of Training Data: {}".format(train_ids_all.shape))

print("Shape of Test Data: {}\n".format(test_ids_all.shape))



def get_full_path_train(idcode):

    return "{}{}{}".format(train_path,idcode,extension)



def get_full_path_test(idcode):

    return "{}{}{}".format(test_path,idcode,extension)





train_ids_all["path"] = train_ids_all["id_code"].apply(lambda x: get_full_path_train(x))

test_ids_all["path"] = test_ids_all["id_code"].apply(lambda x: get_full_path_test(x))
labelnames.at[0,"labelname"]
train_ids_all.head()
test_ids_all.head()
import cv2



def load_image(image_path):

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMSIZE, IMSIZE))

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

    print("{} images of class {} = {}".format(N, classification, labelnames.at[classification,"labelname"]))

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
show_Nimages_of_class(classification=66)
train_ids_all[:3]
#train_ids_all_working, train_ids_all_notused = train_test_split(train_ids_all, test_size=0.75, random_state=42, stratify=train_ids_all[['label']])

train_ids_all_working = train_ids_all
train_df, validation_df = train_test_split(train_ids_all_working, test_size=0.1)
train_y = train_df["label"].values

validation_y = validation_df["label"].values



# Encode labels to one hot vectors

print(train_y.shape)

train_y_cat = to_categorical(train_y, num_classes = 101)

print(train_y_cat.shape)



print(validation_y.shape)

validation_y_cat = to_categorical(validation_y, num_classes = 101)

print(validation_y_cat.shape)
def plot_nice_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(18,18))

    sns.heatmap(cm, annot=True, fmt='d', linewidths=.5,  cbar=False, ax=ax, cmap=plt.cm.copper)

    plt.ylabel('true label')

    plt.xlabel('predicted label')
from keras.applications import DenseNet121, InceptionV3, ResNet50, VGG16

from keras.layers import GlobalAveragePooling2D



reduceLR = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.5,

    patience=7,

    min_lr=1e-6,

    verbose=1,

    mode='min'

)



earlyStopping = EarlyStopping(

    monitor='val_loss',

    patience=10,

    verbose=1,

    mode='min',

    restore_best_weights=True

)
# Define the optimizer

my_optimizer = RMSprop(lr=1e-4)

#my_optimizer = Adam(lr=1e-5)
def plot_training_history(history):

    history_df = pd.DataFrame(history.history)

    f = plt.figure(figsize=(25,5))

    ax = f.add_subplot(121)

    ax.plot(history_df["loss"], label="loss")

    ax.plot(history_df["val_loss"], label = "val_loss")

    ax.legend()

    ax = f.add_subplot(122)

    ax.plot(history_df["acc"], label="acc")

    ax.plot(history_df["val_acc"], label="val_acc")

    ax.legend()
#define test generator "test_data_generator"
test_y_pred_proba = model.predict_generator(test_data_generator, 

                                            steps=np.ceil(float(test_ids_all.shape[0]) / float(BATCH_SIZE)),

                                            verbose=1)



test_y_pred = np.argmax(test_y_pred_proba, axis=1)

nn_results = pd.Series(test_y_pred,name="label")

submission = pd.concat([test_ids_all["id_code"],nn_results], axis = 1)



submission.to_csv("food_submission_simple_model.csv",index=False)


