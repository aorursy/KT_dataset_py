import os, os.path, shutil

import zipfile

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow as tf

import h5py

import random



from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from tensorflow.keras import layers

from tensorflow.keras import Model

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.applications import *

from keras.preprocessing.image import *

from sklearn.utils import shuffle

from keras.models import *

from keras.layers import *

from os import walk





import warnings

warnings.filterwarnings('ignore')
def seed_everything(seed=13):

    np.random.seed(seed)

    tf.random.set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    os.environ['TF_KERAS'] = '1'

    random.seed(seed)

    

seed_everything(419)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as z:

    z.extractall("../kaggle/working/train_unzip")

    

print(f"We have total {len(os.listdir('../kaggle/working/train_unzip/train'))} images in our training data.")
with zipfile.ZipFile("../input/dogs-vs-cats/test1.zip","r") as z:

    z.extractall("../kaggle/working/test_unzip")

    

print(f"We have total {len(os.listdir('../kaggle/working/test_unzip/test1'))} images in our training data.")
folder_path = "../kaggle/working/train_unzip/train"



images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]



for image in images:

    folder_name = image.split('.')[0]



    new_path = os.path.join(folder_path, folder_name)

    if not os.path.exists(new_path):

        os.makedirs(new_path)



    old_image_path = os.path.join(folder_path, image)

    new_image_path = os.path.join(new_path, image)

    shutil.move(old_image_path, new_image_path)
for (dirpath, dirnames, filenames) in walk("../kaggle/working/test_unzip/test1"):

    print("Directory path: ", dirpath)

    print("Folder name: ", dirnames)

    print("File name: ", filenames)
%time



BATCH_SIZE = 128

image_size = (128, 128)

EPOCHS = 200



def write_gap(MODEL, image_size, lambda_func=None):

    width = image_size[0]

    height = image_size[1]

    input_tensor = Input((height, width, 3))

    x = input_tensor

    if lambda_func:

        x = Lambda(lambda_func)(x)

    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)

    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))



    

    gen = ImageDataGenerator()

    train_generator = gen.flow_from_directory("../kaggle/working/train_unzip/train", image_size, shuffle=False, 

                                              batch_size=BATCH_SIZE)

    test_generator = gen.flow_from_directory("../kaggle/working/test_unzip", image_size, shuffle=False, 

                                             batch_size=BATCH_SIZE, class_mode=None)



    train = model.predict_generator(train_generator, train_generator.samples)

    test = model.predict_generator(test_generator, test_generator.samples)

    with h5py.File("gap_%s.h5"%MODEL.__name__) as h:

        h.create_dataset("train", data=train)

        h.create_dataset("test", data=test)

        h.create_dataset("label", data=train_generator.classes)



write_gap(ResNet50, (224, 224))

write_gap(Xception, (299, 299), xception.preprocess_input)

write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)

X_train = []

X_test = []



for filename in ["gap_ResNet50.h5", "gap_Xception.h5", "gap_InceptionV3.h5"]:

    with h5py.File(filename, 'r') as h:

        X_train.append(np.array(h['train']))

        X_test.append(np.array(h['test']))

        y_train = np.array(h['label'])



X_train = np.concatenate(X_train, axis=1)

X_test = np.concatenate(X_test, axis=1)



X_train, y_train = shuffle(X_train, y_train)
input_tensor = Input(X_train.shape[1:])

x = input_tensor

x = Dropout(0.5)(x)

x = Dense(1, activation='sigmoid')(x)

model = Model(input_tensor, x)



model.compile(optimizer=Adam(lr=0.0003, decay=1e-3),

              loss='binary_crossentropy',

              metrics=['accuracy'])
def Callbacks():

    erl = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', 

                        restore_best_weights=True)

    rdc = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, mode='min')

    return [erl,rdc]
history = model.fit(X_train, 

                    y_train, 

                    batch_size=BATCH_SIZE, 

                    epochs=EPOCHS, 

                    callbacks=Callbacks(), 

                    validation_split=0.2)
model.save('model.h5')
def plot_model_history(model_history, acc='accuracy', val_acc='val_accuracy'):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(range(1,len(model_history.history[acc])+1),model_history.history[acc])

    axs[0].plot(range(1,len(model_history.history[val_acc])+1),model_history.history[val_acc])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].set_xticks(np.arange(1,len(model_history.history[acc])+1),len(model_history.history[acc])/10)

    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(['train', 'val'], loc='best')

    plt.show()

    

plot_model_history(history)
y_pred = model.predict(X_test, verbose=1)

y_pred = y_pred.clip(min=0.005, max=0.995)

y_pred
test_filenames = os.listdir('../kaggle/working/test_unzip/test1')

df = pd.DataFrame({'filename': test_filenames})

df.head()
gen = ImageDataGenerator()

test_generator = gen.flow_from_directory("../kaggle/working/test_unzip/", image_size, shuffle=False, 

                                         batch_size=BATCH_SIZE, class_mode=None)
df['category'] = y_pred

df.head()
threshold = 0.5

df['category'] = np.where(y_pred > threshold, "Cat","Dog")







df.to_csv('submission.csv', index=False)

df.head()
df['category'].value_counts().plot.bar()
sample_test = df.sample(n=36).reset_index()

sample_test.head()



plt.figure(figsize=(24, 16))

for index, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img("../kaggle/working/test_unzip/test1/"+filename, target_size=image_size)

    plt.subplot(6, 6, index+1)

    plt.imshow(img)

    plt.xlabel('It\'s a ' + "{}".format(category) )

plt.tight_layout()

plt.show()