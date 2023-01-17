# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print(os.listdir("../input/dogs-vs-cats/"))

import zipfile

import tensorflow as tf



dataset_train="train"

dataset_test="test1"

with zipfile.ZipFile("../input/dogs-vs-cats/"+dataset_train+".zip","r") as z:

    z.extractall(".")

    

with zipfile.ZipFile("../input/dogs-vs-cats/"+dataset_test+".zip","r") as z:

    z.extractall(".")



#from subprocess import check_output

#print(check_output(["ls", "train"]).decode("utf8"))

filenames = os.listdir("/kaggle/working/train")

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})

df.head()
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 
from sklearn.model_selection import train_test_split



train_df, validate_df = train_test_split(df, 

                                         test_size=0.20,                                         

                                         random_state=0)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)



total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

batch_size=15
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import Activation, Flatten, Dense

from keras.utils import to_categorical

#from keras import backend as K



model = Sequential()

#add model layers

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(128,128,3)))

model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(Flatten())

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



validation_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_dataframe(

    train_df,

    "/kaggle/working/train/", 

    x_col='filename',

    y_col='category',

    target_size=(128,128),

    batch_size=15,

    class_mode='categorical')



validation_generator = validation_datagen.flow_from_dataframe(

    validate_df,

    "/kaggle/working/train/", 

    x_col='filename',

    y_col='category',

    target_size=(128,128),

    batch_size=15,

    class_mode='categorical')



model.fit_generator(

    train_generator,

    validation_data=validation_generator,epochs=3)







model.save_weights("model.h5")



test_filenames = os.listdir("/kaggle/working/test1")

test_df = pd.DataFrame({

    'filename': test_filenames

})



nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "/kaggle/working/test1", 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=(128,128),

    batch_size=15,

    shuffle=False

)

predict = model.predict_generator(test_generator)
test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })

test_df.head()
submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission.csv', index=False)