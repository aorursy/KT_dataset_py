#pip download efficientnet -d ./efficientnet

#import os

#from zipfile import ZipFile

#

#dirName = "./"

#zipName = "packages.zip"



## Create a ZipFile Object

#with ZipFile(zipName, 'w') as zipObj:

#    # Iterate over all the files in directory

#    for folderName, subfolders, filenames in os.walk(dirName):

#        for filename in filenames:

#            if (filename != zipName):

#                # create complete filepath of file in directory

#                filePath = os.path.join(folderName, filename)

#                # Add file to zip

#                zipObj.write(filePath)
! pip install efficientnet --no-index --find-links=file:///kaggle/input/vgis9-2020-packages/efficientnet
! [ -f /kaggle/input/vgis2020model/bestmodel.h5 ] && cp /kaggle/input/vgis2020model/bestmodel.h5 /kaggle/working/bestmodel.h5
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import sys

from pathlib import Path

import random



%matplotlib inline

import matplotlib.pyplot as plt

from matplotlib.image import imread

import cv2

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ModelCheckpoint

import efficientnet.tfkeras as efn



input_dir = Path('../input')

dataset_dir = input_dir / 'landmark-recognition-2020'



test_image_dir = dataset_dir / 'test'

train_image_dir = dataset_dir / 'train'

train_label_path = dataset_dir / 'train.csv'

bestmodel_path = Path('/kaggle/working/bestmodel.h5')

    

ERROR = 1

WARN = 2

INFO = 3

DEBUG = 4

SPAM = 5



VERBOSITY = INFO



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
validation_ratio = 0.2

batch_size = 16

max_epochs = 6



top_n = 1000

img_size = (256,256)

seed = 496



force_retrain = True
def get_img_path(df, prepend=""):

    return prepend + df.id.str[0] + "/" + df.id.str[1] + "/" + df.id.str[2] + "/" + df.id + ".jpg" 
def plot_history(history):

    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]

    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]

    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]

    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    

    if len(loss_list) == 0:

        print('Loss is missing in history')

        return 

    

    ## As loss always exists

    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    

    ## Loss

    plt.figure(1)

    for l in loss_list:

        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    for l in val_loss_list:

        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    

    plt.title('Loss')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()

    

    ## Accuracy

    plt.figure(2)

    for l in acc_list:

        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    for l in val_acc_list:    

        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')



    plt.title('Accuracy')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()

    plt.show()
train_labels = pd.read_csv(train_label_path)

train_labels.head(5)
def check_for_test():

    testdf = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')

    test_images  = test_image_dir.glob("**/*.jpg")



    test_img_arr = []

    for img in test_images:

        test_img_arr.append(img.stem)

    

    x = True

    for _id in testdf.id.values:

        if _id not in test_img_arr:

            x = False

            print(f"{_id} missing from folder")



    for img in test_img_arr:

        if img not in testdf.id.values:

            x = False

            print(f"{_img} missing from csv")

    return x



# x = "are" if check_for_test() else "aren't"

# print(f"All test images {x} listed in sample_submission.csv")

## All test images are listed in sample.csv. Will use that
class_count = len(train_labels["landmark_id"].unique())

test_df = pd.read_csv(dataset_dir/"sample_submission.csv")



test_image_count = len(test_df.id.values)

train_image_count = len(train_labels.id.values)



print(f'''Dataset info:

      \tUnique classes: {class_count:}

      \tImages  : {test_image_count + train_image_count :9,d}

      \t  test  : {test_image_count :9,d}

      \t  train : {train_image_count :9,d}

      ''')
# Make a dataframe sorted by amount of images 

df_by_samples = pd.DataFrame(train_labels['landmark_id'].value_counts())

df_by_samples.reset_index(inplace=True)

df_by_samples.columns=['landmark_id','count']





lt_5_cnt = len(df_by_samples.loc[df_by_samples['count'] < 5])

gt_5_lt_10_cnt = len(df_by_samples.loc[(df_by_samples['count'] > 5) & (df_by_samples['count'] < 10)])

lt_100_cnt = len(df_by_samples.loc[df_by_samples['count'] < 100]) 

print(f"""Classes with:

    <5 samples   : {lt_5_cnt}

    >5<10 samples: {gt_5_lt_10_cnt}

    <500 samples : {lt_100_cnt}""")
def plot_bars(data, edges, col=None):



    if col is None:

        col = data

    else:

        col = data[col]



    bins = {}

    for idx in range(len(edges)-1):

        if idx == len(edges)-2:

            key = f">{edges[idx]}"

        else:

            key = f">{edges[idx]} <={edges[idx+1]}"

        bins[key] = len(data.loc[(col > edges[idx]) & (col <= edges[idx+1])])



    

    fig = plt.figure(figsize=(10,3.5))

    

    plt.bar(bins.keys(), bins.values(), width=0.4)



    

    
plot_bars(df_by_samples, [0,5,10,50,100,7000], 'count')
def plot_n_img(dataset, n:int):

    ids = dataset.drop_duplicates(subset=['landmark_id']).sample(n)

    

    paths = get_img_path(ids, str(train_image_dir.resolve())+'/').values

    grid_size = int(np.ceil(np.sqrt(len(paths))))

    

    fig = plt.figure(figsize=(grid_size*3,grid_size*3))

    

    axes = []

    for idx in range(grid_size*grid_size):

        if idx == n:

            break

        axes.append(fig.add_subplot(grid_size, grid_size, idx+1))

        plt.imshow(imread(paths[idx]))

    

    fig.tight_layout()

    plt.show()

        



        
plot_n_img(train_labels, 16)




df_by_samples = df_by_samples.drop(df_by_samples.index[top_n:])

train_labels = train_labels[train_labels.landmark_id.isin(df_by_samples['landmark_id'])]

print(df_by_samples.tail(1))

print(train_labels.shape)
train_labels['path'] = get_img_path(train_labels)

train_labels['label'] = train_labels.landmark_id.astype(str)
def get_genny(data, x_col, y_col, base_dir :str, target_size=(256,256), batch_size=32, validation_ratio=0.0, subset=None, seed=496):

    gen = ImageDataGenerator(validation_split=validation_ratio)

    

    class_mode = "categorical" if validation_ratio > 0 else None

    

    genny = gen.flow_from_dataframe(

        data,

        directory = base_dir,

        x_col=x_col,

        y_col=y_col,

        target_size=target_size,

        batch_size=batch_size,

        subset=subset,

        class_mode=class_mode,

        validate_filenames=False,

        seed=seed

    )

    return genny
gen = ImageDataGenerator(validation_split=validation_ratio)



# The flow_from_dataframe() shuffles the data after splitting it, meaning the training and validation set will contain different classes, so we shuffle the data before

train_labels = train_labels.sample(frac=1).reset_index(drop=True)



train_gen = get_genny(train_labels, "path", "label", str(train_image_dir), img_size, batch_size, validation_ratio, "training")

valid_gen = get_genny(train_labels, "path", "label", str(train_image_dir), img_size, batch_size, validation_ratio, "validation")







print(f"Split training set into a training and validation set")
if not bestmodel_path.exists() or force_retrain:

    model = tf.keras.Sequential([

        efn.EfficientNetB2(

            input_shape=(256, 256, 3),

            weights='imagenet',

            include_top=False

        ),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(top_n, activation='softmax')

    ])
if not bestmodel_path.exists() or force_retrain:

    model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics = ['categorical_accuracy']

    )

    # I'm using the adam optimizer for a few reasons. It's very popular, and that tends to be for a reason, and it attempts to combine the best of both wordlds of momentum and RMSProp

    # I'm using categorical_crossentropy as there's a lot of classes

image_count = len(train_labels)



train_steps = int(image_count * (1-validation_ratio) // batch_size)

valid_steps = int(image_count * validation_ratio // batch_size)



if not bestmodel_path.exists() or force_retrain:

    print(f"Fitting model over {max_epochs} epochs with {train_steps} training steps and {valid_steps} validation steps.")

    

    model_checkpoint = ModelCheckpoint("bestmodel.h5", save_best_only=True, verbose=1)



    hist = model.fit(train_gen,

                    steps_per_epoch=train_steps,

                    epochs=max_epochs,

                    validation_data=valid_gen,

                    validation_steps=valid_steps,

                    callbacks=[model_checkpoint]

    )
plot_history(hist)
sub_df = pd.read_csv(dataset_dir / "sample_submission.csv")

sub_df["path"] = get_img_path(sub_df)

best_model = tf.keras.models.load_model("bestmodel.h5")



test_gen = get_genny(sub_df, "path", None, str(test_image_dir), img_size, 1)

predictions = best_model.predict(test_gen, verbose=1)
predicted_labels = np.argmax(predictions, axis=-1) # Get the index of the one-hot bit in the last axis



classes = np.unique(train_labels.landmark_id.values)

print(classes.shape)

print(predicted_labels.shape)



predicted_labels = [classes[idx] for idx in predicted_labels] 

prediction_prob = np.max(predictions, axis=-1)



print(f"{predicted_labels[0]}: {prediction_prob[0]}")
result = [str(predicted_labels[idx]) + " " + str(prediction_prob[idx]) for idx in range(len(predicted_labels))]
sub_df["landmarks"] = result

sub_df.drop(columns="path")



sub_df.to_csv("submission.csv", index=False)