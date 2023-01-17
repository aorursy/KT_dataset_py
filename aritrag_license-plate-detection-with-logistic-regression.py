import numpy as np

import pandas as pd



import tensorflow as tf

from tensorflow import keras



import matplotlib.pyplot as plt

%matplotlib inline



import os

import cv2
df = pd.read_json("/kaggle/input/vehicle-number-plate-detection/Indian_Number_plates.json", lines=True)

df.head()
os.mkdir("Indian Number Plates")
%%capture

dataset = dict()

dataset["image_name"] = list()

dataset["top_x"] = list()

dataset["top_y"] = list()

dataset["bottom_x"] = list()

dataset["bottom_y"] = list()



counter = 0

for index, row in df.iterrows():

    tf.keras.utils.get_file('/kaggle/working/Indian Number Plates/licensed_car{}.jpeg'.format(counter), row["content"])

    

    dataset["image_name"].append('licensed_car{}.jpeg'.format(counter))

    

    data_points = row["annotation"]

    

    dataset["top_x"].append(data_points[0]["points"][0]["x"])

    dataset["top_y"].append(data_points[0]["points"][0]["y"])

    dataset["bottom_x"].append(data_points[0]["points"][1]["x"])

    dataset["bottom_y"].append(data_points[0]["points"][1]["y"])

    

    counter += 1

print("Downloaded {} car images.".format(counter))
df = pd.DataFrame(dataset)

df.head()
from skimage import io

def verify_image(img_file):

    try:

        img = io.imread(img_file)

    except:

        return False

    return True
for files in os.listdir('/kaggle/working/Indian Number Plates/'):

    if not verify_image('/kaggle/working/Indian Number Plates/{}'.format(files)):

        print(files)
df = df.drop([107])
df.to_csv("indian_license_plates.csv", index=False)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.1)



train_generator = datagen.flow_from_dataframe(

    df,

    directory="Indian Number Plates/",

    x_col="image_name",

    y_col=["top_x", "top_y", "bottom_x", "bottom_y"],

    target_size=(128, 128),

    batch_size=32, 

    class_mode="other",

    subset="training")



validation_generator = datagen.flow_from_dataframe(

    df,

    directory="Indian Number Plates/",

    x_col="image_name",

    y_col=["top_x", "top_y", "bottom_x", "bottom_y"],

    target_size=(128, 128),

    batch_size=32, 

    class_mode="other",

    subset="validation")
tf.keras.backend.clear_session()

model = tf.keras.models.Sequential(

[

    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128, 128, 3)),

    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),

    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(264,(3,3),activation='relu'),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(256,activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dense(64,activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(32,activation='relu'),

    tf.keras.layers.Dense(4,activation='sigmoid')

]

)





model.summary()
STEP_SIZE_TRAIN = int(np.ceil(train_generator.n / train_generator.batch_size))

STEP_SIZE_VAL = int(np.ceil(validation_generator.n / validation_generator.batch_size))



print("Train step size:", STEP_SIZE_TRAIN)

print("Validation step size:", STEP_SIZE_VAL)



train_generator.reset()

validation_generator.reset()
model.compile(optimizer='adam', loss="mse")
history = model.fit_generator(train_generator,

    steps_per_epoch=STEP_SIZE_TRAIN,

    validation_data=validation_generator,

    validation_steps=STEP_SIZE_VAL,

    epochs=20)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
model.evaluate_generator(validation_generator, steps=STEP_SIZE_VAL)
for idx, row in df.iloc[0:5,:].iterrows():    

    img = cv2.resize(cv2.imread("Indian Number Plates/" + row[0]) / 255.0, dsize=(128, 128))

    y_hat = model.predict(img.reshape(1, 128, 128, 3)).reshape(-1) * 128

    

    xt, yt = y_hat[0], y_hat[1]

    xb, yb = y_hat[2], y_hat[3]

    

    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)

    image = cv2.rectangle(img, (xt, yt), (xb, yb), (0, 0, 255), 1)

    plt.imshow(image)

    plt.show()