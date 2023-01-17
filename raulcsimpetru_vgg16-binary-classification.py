from IPython.display import clear_output

!pip install imutils

!pip install tensorflow-gpu==2.0.0-beta1



clear_output()
# Set everything up

import os



import cv2

import imutils as imutils

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf # machine learning

from tqdm import tqdm # make your loops show a smart progress meter 

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sn



RANDOM_SEED = 1

IMG_SIZE = (224, 224) # size of vgg16 input

IMG_PATH = "../input/brain-tumor-images-dataset/brain tumor images dataset/Brain Tumor Images Dataset/"



print(os.listdir(IMG_PATH))
Test_Path = IMG_PATH + "test_set/"

Training_Path = IMG_PATH + "training_set/"

Validation_Path = IMG_PATH + "validation_set/"
def number_of_imgs(path):

    print(path)

    for value in os.listdir(path):

        print(value, "has", len(os.listdir(path + value)), "imgs")

    print('\n')

    

    

number_of_imgs(Test_Path)

number_of_imgs(Training_Path)

number_of_imgs(Validation_Path)
def create_dataframe(path):

    data = []

    for value in os.listdir(path):

        for image in os.listdir(path + value + "/"):

            file_path = path + value + "/" + image

            # if hemmorhage than set if to 1 else 0

            hemmorhage = 1 if value.lower() == "hemmorhage_data" else 0

            data.append({"path": file_path, 'hemmorhage': hemmorhage})

            

    df = pd.DataFrame(data=data).sample(frac=1).reset_index(drop=True)



    return df
%matplotlib inline

def plot_imgs(title, paths):

    fig = plt.figure(figsize=(14, 8), dpi=72)

    fig.suptitle(title, fontsize=24, y=1.05)

    for i, row in paths.iterrows():

        img=mpimg.imread(row['path'])

        plt.subplot(3, 5, i+1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        subtitle = 'YES' if row['hemmorhage'] == 1 else 'NO'

        plt.title(subtitle)

        plt.imshow(img)

    plt.tight_layout()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None)

    plt.show()

    

    

plot_imgs("Test", create_dataframe(Test_Path).sample(15).reset_index(drop=True))

plot_imgs("Train", create_dataframe(Training_Path).sample(15).reset_index(drop=True))

plot_imgs("Validation", create_dataframe(Validation_Path).sample(15).reset_index(drop=True))
def crop_imgs(set_name, add_pixels_value=0):

    """Finds the extreme points on the image and crops the rectangular out of them"""

    set_new = []

    for img in set_name:

        img = cv2.imread(img)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        gray = cv2.GaussianBlur(gray, (5, 5), 0)



        # threshold the image, then perform a series of erosions +

        # dilations to remove any small regions of noise

        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.erode(thresh, None, iterations=2)

        thresh = cv2.dilate(thresh, None, iterations=2)



        # find contours in thresholded image, then grab the largest one

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)

        c = max(cnts, key=cv2.contourArea)



        # find the extreme points

        ext_left = tuple(c[c[:, :, 0].argmin()][0])

        ext_right = tuple(c[c[:, :, 0].argmax()][0])

        ext_top = tuple(c[c[:, :, 1].argmin()][0])

        ext_bot = tuple(c[c[:, :, 1].argmax()][0])



        add_pixels = add_pixels_value

        new_img = img[ext_top[1] - add_pixels:ext_bot[1] + add_pixels

        , ext_left[0] - add_pixels:ext_right[0] + add_pixels].copy()

        set_new.append(new_img)

        

    return np.array(set_new)



crop_imgs(create_dataframe(Test_Path)['path'])

crop_imgs(create_dataframe(Training_Path)['path'])

crop_imgs(create_dataframe(Validation_Path)['path'])



clear_output()
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    rotation_range=15,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=0.1,

    brightness_range=[0.5, 1.25],

    horizontal_flip=True,

    vertical_flip=True,

    preprocessing_function=tf.keras.applications.vgg16.preprocess_input

)



test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    preprocessing_function=tf.keras.applications.vgg16.preprocess_input

)





train_generator = train_datagen.flow_from_directory(

    Training_Path,

    color_mode='rgb',

    target_size=IMG_SIZE,

    batch_size=32,

    class_mode='binary',

    seed=RANDOM_SEED

)





validation_generator = test_datagen.flow_from_directory(

    Validation_Path,

    color_mode='rgb',

    target_size=IMG_SIZE,

    batch_size=16,

    class_mode='binary',

    seed=RANDOM_SEED

)
vgg16_weight_path = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

base_model = tf.keras.applications.VGG16(

    weights=vgg16_weight_path,

    include_top=False,

    input_shape=IMG_SIZE + (3,)

)



model = tf.keras.models.Sequential()

model.add(base_model)

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



model.layers[0].trainable = False



model.compile(

    loss='binary_crossentropy',

    optimizer=tf.keras.optimizers.Adam(),

    metrics=['accuracy']

)



model.summary()
EPOCHS = 25

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor='val_accuracy',

    mode='max',

    patience=6

)



history = model.fit_generator(

    train_generator,

    steps_per_epoch=50,

    epochs=EPOCHS,

    validation_data=validation_generator,

    validation_steps=25,

    callbacks=[early_stopping]

)



print("Training Done")

model.save("model.h5")
def preprocess_imgs(path, img_size):

    set_new = []

    for value in os.listdir(path):

        for img in os.listdir(path + value):

            img = cv2.imread(path + value + "/" + img)

            img = cv2.resize(

                img,

                dsize=img_size,

                interpolation=cv2.INTER_CUBIC

            )

            set_new.append(tf.keras.applications.vgg16.preprocess_input(img))

    

    return np.array(set_new)



test_data = preprocess_imgs(Test_Path, img_size=IMG_SIZE)



reality = []

for value in os.listdir(Test_Path):

    for img in os.listdir(Test_Path + value):

        reality.append(1) if value.lower() == "hemmorhage_data" else reality.append(0)

        

predictions = model.predict(test_data)

predictions = [0 if x > 0.5 else 1 for x in predictions]



accuracy = accuracy_score(reality, predictions)

print("Test Accuracy:", accuracy)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(1, len(history.epoch) + 1)





plt.figure(figsize=(10,5))



plt.plot(epochs_range, acc, label='Training')

plt.plot(epochs_range, val_acc, label='Validation')

plt.legend(loc="best")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Model Accuracy')

plt.grid(b=True, which='major', color='#666666', linestyle='-')

plt.tight_layout()

plt.show()



plt.figure(figsize=(10,5))



plt.plot(epochs_range, loss, label='Training')

plt.plot(epochs_range, val_loss, label='Validation')

plt.legend(loc="best")

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Model Loss')

plt.grid(b=True, which='major', color='#666666', linestyle='-')

plt.tight_layout()

plt.show()

confusion_mtx = confusion_matrix(reality, predictions)



ax = plt.axes()

sn.heatmap(confusion_mtx, annot=True,annot_kws={"size": 25}, cmap="Blues", ax = ax)

ax.set_title('Test Accuracy', size=14)

plt.show()