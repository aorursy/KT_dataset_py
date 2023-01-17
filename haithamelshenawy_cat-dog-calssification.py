# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, load_img

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
local_zip = '/kaggle/input/dogs-vs-cats/train.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/kaggle/output/kaggle/working/dogs-vs-cats')
zip_ref.close()

save_dir = "/kaggle/output/kaggle/working/dogs-vs-cats/saved_history"
# os.mkdir(save_dir+"/saved_history")
img_dir = "/kaggle/output/kaggle/working/dogs-vs-cats/train"
filenames = os.listdir("/kaggle/output/kaggle/working/dogs-vs-cats/train")

category = []

for i in filenames:
    s= i.split(".")
    category.append(s[0])

df = pd.DataFrame({"filename": filenames, "Category": category})
df["Category"].unique()
fig, ax = plt.subplots(figsize=(5,5))
ax.hist(df["Category"])
df = df.sample(frac=1).reset_index(drop=True)
df
smple = np.random.choice(df["filename"])
img = load_img(img_dir+"/"+smple, target_size = (150,150))
plt.imshow(img)
df_train, df_test = train_test_split(df, test_size=.2)
df_train.shape
train_datagen = ImageDataGenerator( rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1 )
test_datagen = ImageDataGenerator( rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1 )

train_generator = train_datagen.flow_from_dataframe(df_train, img_dir, target_size=(150,150),
                                                   x_col= "filename", y_col= "Category",
                                                   class_mode= "categorical",
                                                   batch_size=15)

test_generator = test_datagen.flow_from_dataframe(df_test, img_dir, target_size=(150,150),
                                                   x_col= "filename", y_col= "Category",
                                                   class_mode= "categorical",
                                                   batch_size=15)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=1, 
                                            factor=0.5, min_lr=0.00001)

csv_logger = tf.keras.callbacks.CSVLogger(save_dir+"/training_log", separator=",", append=True)
from tensorflow.keras.layers import BatchNormalization, Dropout

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3), input_shape=(150,150,3), activation="relu"),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(32,(3,3), activation="relu"),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, "relu"),
    BatchNormalization(),
    Dropout(0.5),
    
    tf.keras.layers.Dense(265, "relu"),
    BatchNormalization(),
    Dropout(0.5),
    
    tf.keras.layers.Dense(2, "softmax")
])
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()
history= model.fit_generator(train_generator, epochs = 10, validation_data = test_generator,
                            callbacks = [early_stop, learning_rate_reduction, csv_logger])
history.history
epochs = np.arange(1,11,1)
train_acc = history.history["accuracy"]
test_acc = history.history["val_accuracy"]
train_loss = history.history["loss"]
test_loss = history.history["val_loss"]

fig, (ax1, ax2 )= plt.subplots(2,1, figsize= (10,10))
ax1.plot(epochs,train_acc, color="r", label= "Train_Accuracy" )
ax1.plot(epochs, test_acc, color="b", label= "Test_Accuracy")
ax1.legend(shadow=True, fontsize= 13)
ax1.set_xlabel("Epochs", fontsize=16 )
ax1.set_ylabel("Accuracy", fontsize=16)


ax2.plot(epochs,train_loss, color="r", label= "Train_Loss" )
ax2.plot(epochs, test_loss, color="b", label= "Test_Loss")
ax2.legend(shadow=True, fontsize= 13)
ax2.set_xlabel("Epochs", fontsize=16 )
ax2.set_ylabel("Loss", fontsize=16)
plt.show()

print(os.listdir(save_dir))
model.save(save_dir)
new_model = tf.keras.models.load_model(save_dir)
new_model.summary()
df_log = pd.read_csv(save_dir+"/training_log")
df_log
local_zip = '/kaggle/input/dogs-vs-cats/test1.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/kaggle/output/kaggle/working/dogs-vs-cats/test1')
zip_ref.close()
test_dir = "/kaggle/output/kaggle/working/dogs-vs-cats/test1/test1"

test_list = os.listdir(test_dir)
len(test_list)
df_test = pd.DataFrame({"filename": test_list})
df_test.head()
test2_datagen = ImageDataGenerator(rescale=1/255)
test2_generator = test2_datagen.flow_from_dataframe(df_test, test_dir, x_col= "filename",
                                                    target_size=(150,150),class_mode=None)
predict = model.predict_generator(test2_generator)
predict
predict_class = predict.argmax(axis = -1)

predict_class
predict_by_name = ["cat" if x == 0 else "dog" for x in predict_class]

predict_by_name
train_generator.class_indices
df_test.insert(column="category_by_name", value=predict_by_name, loc=1)
df_test.head()
sample = np.random.choice(df_test["filename"])
img = load_img(test_dir+"/"+sample, target_size=(150,150))
label = df_test.loc[(df_test["filename"] == sample)]["category_by_name"].item()
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.xlabel("{} is a {}".format(sample, label))

