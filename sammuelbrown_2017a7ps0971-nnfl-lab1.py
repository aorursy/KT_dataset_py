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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
filenames = os.listdir("../input/nnfl-lab-1/training/training/")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    real_category = category.split('_')[0]
    categories.append(real_category)
    
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


df.head()
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
total_overall = df.shape[0]
batch_size=10
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "../input/nnfl-lab-1/training/training/", 
    x_col='filename',
    y_col='category',
    target_size=(128,128),
    class_mode='categorical',
    batch_size=batch_size
)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "../input/nnfl-lab-1/training/training/", 
    x_col='filename',
    y_col='category',
    target_size=(128,128),
    class_mode='categorical',
    batch_size=batch_size
)
overall_datagen = train_datagen.flow_from_dataframe(
    df, 
    "../input/nnfl-lab-1/training/training/", 
    x_col='filename',
    y_col='category',
    target_size=(128,128),
    class_mode='categorical',
    batch_size=batch_size
)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization



model = Sequential()


model.add(Conv2D(filters=96, input_shape=(128,128,3), kernel_size=(11,11), strides=(4,4), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))


model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))


model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same')) #384
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))  #384
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))





model.add(Flatten())

model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.4))


model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))


model.add(Dense(1000))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))


model.add(Dense(4))
model.add(BatchNormalization())
model.add(Activation('softmax'))


model.summary()
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import add
from keras.utils import plot_model



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 50

history = model.fit_generator(
    overall_datagen, 
    epochs=epochs,
    #validation_data=validation_generator,
    #validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    
)
model.save_weights("model.h5")
test_filenames = os.listdir("../input/nnfl-lab-1/testing/testing/")
test_df = pd.DataFrame({
    'id': test_filenames
})
nb_samples = test_df.shape[0]
test_df.head()
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "../input/nnfl-lab-1/testing/testing/", 
    x_col='id',
    y_col=None,
    class_mode=None,
    target_size=(128,128),
    batch_size=batch_size,
    shuffle=False
)
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
predict
test_df['label'] = np.argmax(predict, axis=-1)
test_df.head()
from collections import Counter

Counter(test_df["label"])
sample_test = test_df.sample(10)
sample_test.head()
plt.figure(figsize=(12, 24))
count = 1
for index, row in sample_test.iterrows():
    filename = row['id']
    category = row['label']
    img = load_img("../input/nnfl-lab-1/testing/testing/"+filename, target_size=(128,128))
    plt.subplot(6, 3, count)
    count +=1
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()
test_df.head()
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(test_df)
from IPython.display import FileLink
FileLink(r'./model.h5')
