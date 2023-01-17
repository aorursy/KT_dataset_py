from tensorflow.keras.preprocessing.image import ImageDataGenerator

from IPython.display import Image, display

import tensorflow as tf

import pandas as pd
csv_path = "../input/landmark-recognition-2020/train.csv"

base_directory = "../input/landmark-recognition-2020/train"
columns = ['id','landmark_id']

annotations = pd.read_csv(csv_path, usecols=columns)



data_frame = {"image_dir":[],"landmark_id":[]}
for image_id, land_id in zip(annotations["id"],annotations["landmark_id"]):

    image_dir = "{}/{}/{}/{}/{}.jpg".format(base_directory,image_id[0],image_id[1],image_id[2],image_id)

    data_frame["image_dir"].append(image_dir)

    data_frame["landmark_id"].append(land_id)
df = pd.DataFrame(data_frame)

df.to_csv("train_data.csv",index=False)
data_csv_path = "./train_data.csv"



columns = ['image_dir','landmark_id']

data_csv = pd.read_csv(data_csv_path, usecols=columns,dtype=str)
# let us check is everthing is assigned properly

# This line just tell pandas to print whole string 

pd.options.display.max_colwidth = 100

print("First 7 entries of original train.csv")

print(annotations.head(7))

print("First 7 entries of our processed train_data.csv")

print(data_csv.head(7))

print("Last 7 entries of original train.csv")

print(annotations.tail(7))

print("Last 7 entries of our processed train_data.csv")

print(data_csv.tail(7))
# let's visualize some images from our csv

tail_part = data_csv.tail(5)

for image,label in zip(tail_part["image_dir"],tail_part["landmark_id"]):

    display(Image(image))
no_of_classes=len(annotations["landmark_id"].unique())

no_of_images=len(annotations["id"].unique())

print("There are total {} images belonging to {} classes".format(no_of_images,no_of_classes))
datagen = ImageDataGenerator(

    rescale=1./255,

    shear_range=0.2,

    rotation_range=30,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    validation_split=0.1,

    fill_mode='nearest')
batch_size = 32

target_shape=(64,64)



train_generator = datagen.flow_from_dataframe(

        dataframe = data_csv,

        x_col = "image_dir",

        y_col = "landmark_id",

        target_size = target_shape,

        color_mode = "rgb",

        class_mode = "categorical",

        batch_size = batch_size,

        subset = 'training'

)

validation_generator = datagen.flow_from_dataframe(

        dataframe = data_csv,

        x_col = "image_dir",

        y_col = "landmark_id",

        target_size = target_shape,

        color_mode = "rgb",

        class_mode = "categorical",

        batch_size = batch_size,

        subset = 'validation'

)
mymodel = tf.keras.Sequential([

    tf.keras.layers.InputLayer(input_shape=(64,64,3)),

    tf.keras.layers.MaxPooling2D(2),

    tf.keras.layers.Conv2D(filters=4, kernel_size=3, activation="relu"),

    tf.keras.layers.MaxPooling2D(2),

    tf.keras.layers.Conv2D(filters=4, kernel_size=3, activation="relu"),

    tf.keras.layers.MaxPooling2D(2),

    tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation="relu"),

    tf.keras.layers.MaxPooling2D(2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation="sigmoid"),

    tf.keras.layers.Dense(no_of_classes, activation="softmax"),

])

mymodel.summary()
mymodel.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['categorical_accuracy'])