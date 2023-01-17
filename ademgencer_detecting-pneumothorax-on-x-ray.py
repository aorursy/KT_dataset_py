# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # import matplot library

import seaborn as sns # import seaborn library

import cv2





from glob import glob



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Loads path and data

PATH = os.path.abspath(os.path.join('..', 'input/sample/'))

SOURCE_IMAGES = os.path.join(PATH, "sample", "images")

images = glob(os.path.join(SOURCE_IMAGES, "*.png"))



data_path = "/kaggle/input/sample/sample/"

data = pd.read_csv(data_path + "sample_labels.csv")
# Rename column names for easy use 

data.rename(columns=

            {"Image Index": "image",

            "Finding Labels": "labels",

            "Patient Age": "age",

            "Patient Gender": "gender",

            "View Position": "position",

            "OriginalImageWidth": "width",

            "OriginalImageHeight": "height",

            "OriginalImagePixelSpacing_x": "space_x",

            "OriginalImagePixelSpacing_y": "space_y"}, inplace=True)
# Clean data

data["gender"].replace(["F", "M"],[0, 1],inplace=True)



data["age"].replace("411Y", "041Y", inplace=True)

data["age"] = data["age"].str.replace("Y","365")  # For Years

data["age"] = data["age"].str.replace("M","030")  # For Months

data["age"] = data["age"].str.replace("D","001")  # For Days

data["age"] = data["age"].astype(float)

data["age"] = (data["age"]%1000 * data["age"]//1000)//365  # Convert age to years



data.labels.replace("No Finding", "xNothing", inplace=True) 
# Visualization of data



#plt.figure(figsize=(18,9))

#sns.countplot(data.age)

#plt.xticks(rotation=90)

#plt.show()



#sns.countplot(data.gender)

#plt.show()



#sns.countplot(data.position)

#plt.show()



# sns.pairplot(data)
# Describe data statistics

data.describe()
# Extract features for diagnosis



from sklearn.feature_extraction.text import CountVectorizer



# create a dataframe from a word matrix

def wm2df(wm, feat_names, dindex):

    

    # create an index for each row

    doc_names = [format(idx) for idx, _ in enumerate(wm)]

    df = pd.DataFrame(data=wm.toarray(), index=dindex,

                      columns=feat_names)

    return(df)

  

# set of documents

corpora = data.labels



# instantiate the vectorizer object

cvec = CountVectorizer(lowercase=False)



# convert the documents into a document-term matrix

wm = cvec.fit_transform(corpora)



# retrieve the terms found in the corpora

tokens = cvec.get_feature_names()



# Array for indexing

dindex = data.index.array



# create a dataframe from the matrix

df = wm2df(wm, tokens, dindex)



# add features to dataframe

data = pd.concat([df, data], axis=1, sort=False)


# Visualize diagnosis



y = sum(data.iloc[:,0:14].values)

x = data.columns.values[:14]



plt.figure(figsize=(16,8))

sns.barplot(x=x, y=y)

plt.grid()

plt.xticks(rotation=90)

plt.xlabel("Diagnosis")

plt.ylabel("Number of cases")

plt.title("Positive diagnosis for CNN")

plt.show()
# Show sample images (multiple)



multipleImages = glob('/kaggle/input/sample/sample/images/**')

i_ = 0

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.subplots_adjust(wspace=0, hspace=0)

for l in multipleImages[:25]:

    im = cv2.imread(l)

    im = cv2.resize(im, (128, 128)) 

    plt.subplot(5, 5, i_+1) #.set_title(l)

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

    i_ += 1
# Show sample image (single)

test_index = 345

e = cv2.imread(os.path.join(SOURCE_IMAGES, data.image.iloc[test_index:].values[0]))



plt.imshow(e)

plt.show
x = cv2.resize(e, (124, 124), interpolation=cv2.INTER_CUBIC)

x.shape

def proc_images():

    """

    Returns two arrays: 

        x is an array of resized images

        y is an array of labels

    """

    

    disease="Infiltration"



    x = [] # images as arrays

    y = [] # labels for diagnosis

    WIDTH = 128

    HEIGHT = 128



    for img in images[:50]:

        base = os.path.basename(img)

        y.append(data.Pneumothorax[data.image == base].values[0])



        # Read and resize image

        full_size_image = cv2.imread(img)

        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))



    return x, y
x,y = proc_images()
# Set it up as a dataframe if you like

df = pd.DataFrame()

df["labels"]=y

df["images"]=x
print(len(df), df.images[0].shape)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1, stratify=y)



print(np.array(x_train).shape)

print(np.array(x_test).shape)
from keras.preprocessing.image import ImageDataGenerator

from keras import layers, models, optimizers

from keras import backend as K
img_width, img_height = 128, 128

nb_train_samples = len(x_train)

nb_validation_samples = len(x_test)

epochs = 50

batch_size = 32
model = models.Sequential()



model.add(layers.Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))

model.add(layers.BatchNormalization())

model.add(layers.Activation("relu"))



model.add(layers.MaxPooling2D((2, 2)))



model.add(layers.Conv2D(128, (3, 3)))

model.add(layers.BatchNormalization())

model.add(layers.Activation("relu"))



model.add(layers.MaxPooling2D((2, 2)))



model.add(layers.Conv2D(128, (3, 3)))

model.add(layers.BatchNormalization())

model.add(layers.Activation("relu"))



model.add(layers.MaxPooling2D((2, 2)))



model.add(layers.Flatten())

model.add(layers.Dropout(0.2))

model.add(layers.Dense(64))

model.add(layers.BatchNormalization())

model.add(layers.Activation("relu"))



model.add(layers.Dense(1))

model.add(layers.BatchNormalization())

model.add(layers.Activation("sigmoid"))



model.compile(

    loss='binary_crossentropy',

    optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),

    metrics=['acc'])



model.summary()
train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, rotation_range=10)

valtest_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow(np.array(x_train), y_train, batch_size=batch_size)

test_generator = valtest_datagen.flow(np.array(x_test), y_test, batch_size=batch_size)
history = model.fit_generator(

    train_generator, 

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=epochs,

)



model.save_weights('weights.h5')
print(history.history)
acc = history.history["acc"]

#test_acc = history.history['test_acc']

loss = history.history['loss']

#test_loss = history.history['test_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'blue', label='Training acc')

plt.plot(epochs, loss, 'red', label='Training loss')

#plt.plot(epochs, test_acc, 'red', label='Test acc')

plt.title('Training accuracy and loss')

plt.legend()

#plt.plot(epochs, test_loss, 'red', label='Test loss')

plt.show()