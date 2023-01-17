# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.listdir("/kaggle/input/dogs-vs-cats-redux-kernels-edition/")
import zipfile
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip", "r") as z:
    z.extractall(".")
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip", "r") as z:
    z.extractall(".")
len(os.listdir("/kaggle/working/train/")), len(os.listdir("/kaggle/working/test"))
path_train_data = "/kaggle/working/train/"
filenames = os.listdir(path_train_data)
categories = []

for i in filenames:
    categ = i.split(".")[0]
    if categ == "dog":
        categories.append(1)
    else:
        categories.append(0)
        
train_df = pd.DataFrame({
    "filename" : filenames,
    "category" : categories
})
train_df.head()
train_df["category"] = train_df["category"].replace({0 : "cat", 1 : "dog"})
train_df.head()
from sklearn.model_selection import train_test_split
df_train, df_validation = train_test_split(train_df, test_size = 0.2, random_state = 42)
df_train.shape, df_validation.shape
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range= 15,
                                                                rescale= 1./255,
                                                                shear_range= 0.1,
                                                                zoom_range= 0.2,
                                                                horizontal_flip= True,
                                                                width_shift_range= 0.1,
                                                                height_shift_range= 0.1)
train_generator = train_datagen.flow_from_dataframe(dataframe= df_train,
                                                   directory= path_train_data,
                                                   x_col= "filename",
                                                   y_col= "category",
                                                   target_size= (128, 128),
                                                   class_mode= "binary",
                                                   batch_size= 15)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1./255)
validation_generator = validation_datagen.flow_from_dataframe(dataframe= df_validation,
                                                             directory= path_train_data,
                                                             x_col= "filename",
                                                             y_col= "category",
                                                             target_size= (128,128),
                                                             class_mode= "binary",
                                                             batch_size= 15)
def build_model():
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation = "relu", input_shape = (128,128,3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation = "relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.Conv2D(128, (3,3), activation = "relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation = "relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(2, activation = "softmax"))
    
    return model
model = build_model()
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics=["accuracy"])
model.summary()
total_train = df_train.shape[0]
total_validate = df_validation.shape[0]
total_train, total_validate
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3,
                                                verbose=1, min_lr= 0.0001)
history = model.fit_generator(train_generator,
                             epochs= 10, validation_data= validation_generator,
                             validation_steps= total_validate // 15,
                             steps_per_epoch= total_train // 15,
                             callbacks=[reduce_lr])
test_file = os.listdir("/kaggle/working/test/")

test_df = pd.DataFrame({
    "filename" : test_file
})

test_size = test_df.shape[0]
test_df.head()
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1./255)
test_generator = test_gen.flow_from_dataframe(dataframe= test_df,
                                             directory= "/kaggle/working/test/",
                                             x_col= "filename",
                                             y_col= None,
                                             class_mode= None,
                                             target_size= (128,128),
                                             batch_size= 15)
pred = model.predict_generator(test_generator)
pred
pred = np.argmax(pred, axis=1)
pred
pred_df = pd.DataFrame(data= pred)
pred_df.head()
pred_df.index += 1
pred_df = pred_df.reset_index()
pred_df.columns = ["id", "label"]
pred_df.head()
pred_df.shape
pred_df.tail()
pred_df.to_csv("Dogs Vs Cats CNN 1.csv", index= False)
