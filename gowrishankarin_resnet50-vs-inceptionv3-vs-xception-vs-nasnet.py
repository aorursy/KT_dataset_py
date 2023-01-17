import os

print(os.listdir("../input"))

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")

print("Shape of train data: {0}".format(train_df.shape))

test_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")

print("Shape of test data: {0}".format(test_df.shape))



diagnosis_df = pd.DataFrame({

    'diagnosis': [0, 1, 2, 3, 4],

    'diagnosis_label': ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

})



train_df = train_df.merge(diagnosis_df, how="left", on="diagnosis")



train_image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("../input/aptos2019-blindness-detection/train_images")) for f in fn]

train_images_df = pd.DataFrame({

    'files': train_image_files,

    'id_code': [file.split('/')[4].split('.')[0] for file in train_image_files],

})

train_df = train_df.merge(train_images_df, how="left", on="id_code")

del train_images_df

print("Shape of train data: {0}".format(train_df.shape))



test_image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("../input/aptos2019-blindness-detection/test_images")) for f in fn]

test_images_df = pd.DataFrame({

    'files': test_image_files,

    'id_code': [file.split('/')[4].split('.')[0] for file in test_image_files],

})





test_df = test_df.merge(test_images_df, how="left", on="id_code")

del test_images_df

print("Shape of test data: {0}".format(test_df.shape))



# Any results you write to the current directory are saved as output.
train_df.head()
test_df.head()
IMG_SIZE = 150

N_CLASSES = train_df.diagnosis.nunique()

CLASSES = list(map(str, range(N_CLASSES)))

BATCH_SIZE = 32

EPOCH_STEPS = 10

EPOCHS = 25

NB_FILTERS = 32

KERNEL_SIZE = 4

CHANNELS = 3
import tensorflow as tf

print(tf.__version__)



from keras.preprocessing.image import ImageDataGenerator



train_df["diagnosis"] = train_df["diagnosis"].astype(str)



train_data_gen = ImageDataGenerator(

    # featurewise_center = True, # Set input mean to 0 over the dataset

    samplewise_center = True, # set each sample mean to 0

    featurewise_std_normalization = True, # Divide inputs by std of the dataset

    samplewise_std_normalization = True, # Divide each input by its std

    # zca_whitening = True, # Apply ZCA whitening

    zca_epsilon = 1e-06, # Epsilon for ZCA whitening,

    rotation_range = 30, # randomly rotate imges in the range (degrees, 0 to 189)

    width_shift_range = 0.1, # randomly shift images horizontally (fraction of total width)

    height_shift_range = 0.1, # Randomly shift images vertically (fraction of total height)

    shear_range = 0, # set range for random shear

    zoom_range = [0.75, 1.25], # set range for random zoom

    channel_shift_range = 0.05, # set range for random channel shifts

    fill_mode = 'constant', # set mode for filling points outside the input boundaries

    cval = 0, # value used for fill_mode

    horizontal_flip = True,

    vertical_flip = True,

    rescale = 1/255.,

    preprocessing_function = None,

    validation_split=0.1

)

train_data = train_data_gen.flow_from_dataframe(

    dataframe=train_df, 

    x_col="files",

    y_col="diagnosis",

    batch_size=BATCH_SIZE,

    shuffle=True,

    classes=CLASSES,

    class_mode="categorical",

    target_size=(IMG_SIZE, IMG_SIZE),

    subset="training"

)



validation_data = train_data_gen.flow_from_dataframe(

    dataframe=train_df, 

    x_col="files",

    y_col="diagnosis",

    batch_size=BATCH_SIZE,

    shuffle=True,

    classes=CLASSES,

    class_mode="categorical",

    target_size=(IMG_SIZE, IMG_SIZE),

    subset="validation"

)



test_data_gen = ImageDataGenerator(rescale=1./255)

test_data = test_data_gen.flow_from_dataframe(

    dataframe=test_df,

    x_col="files",

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size = 1,

    shuffle=False,

    class_mode=None

)
from tensorflow.python.keras.applications import ResNet50, InceptionV3, Xception, NASNetLarge

print(os.listdir(("../input/keras-pretrained-models/")))

print(os.listdir(("../input/nasnetlarge/")))



model_resnet50 = ResNet50(

    weights="../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", 

    include_top=False, 

    input_tensor=tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

)





model_inception_v3 = InceptionV3(

    weights="../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5", 

    include_top=False, 

    input_tensor=tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

)



model_xception = Xception(

    weights="../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5", 

    include_top=False, 

    input_tensor=tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

)



model_nasnet_large = NASNetLarge(

    weights="../input/nasnetlarge/NASNet-large-no-top.h5", 

    include_top=False, 

    input_tensor=tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

)





def create_model():

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(NB_FILTERS, (KERNEL_SIZE, KERNEL_SIZE), padding="valid", strides=1, input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS), activation="relu"),

        tf.keras.layers.Conv2D(NB_FILTERS, (KERNEL_SIZE, KERNEL_SIZE), activation="relu"),

        tf.keras.layers.Conv2D(NB_FILTERS, (KERNEL_SIZE, KERNEL_SIZE), activation="relu"),

        tf.keras.layers.MaxPooling2D(pool_size=(8, 8)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(2048, activation="relu"),

        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(2048, activation="relu"),

        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(N_CLASSES, activation="softmax")

        

    ])

    return model
# Resnet50: 0.396

# create_model: 0.152

# InceptionV3: 0.559

# Xception: 0.509



def get_model(model):

    X = model.output



    X = tf.keras.layers.GlobalAveragePooling2D()(X)

    X = tf.keras.layers.Dense(2048, activation='relu')(X)

    X = tf.keras.layers.Dropout(0.25)(X)

    X = tf.keras.layers.Dense(1024, activation='relu')(X)

    X = tf.keras.layers.Dropout(0.25)(X)

    X = tf.keras.layers.Dense(512, activation='relu')(X)

    X = tf.keras.layers.Dropout(0.25)(X)

    X = tf.keras.layers.Dense(256, activation='relu')(X)

    X = tf.keras.layers.Dropout(0.25)(X)

    X = tf.keras.layers.Dense(128, activation='relu')(X)

    predictions = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(X)

    model = tf.keras.Model(inputs=model.input, outputs=predictions)

    

#     for layer in model.layers:

#         layer.trainable = True

        

#     for layer in model.layers[15:]:

#         layer.trainable = False

    

    return model



optimizer=tf.keras.optimizers.Nadam(lr=2*1e-3, schedule_decay=1e-5)
algo = "inception_v3"

klass = "basics"



model = get_model(model_inception_v3)



opt = tf.keras.optimizers.Adam(lr=0.001, epsilon=1e-6)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()



from sklearn.metrics import cohen_kappa_score

class QWKCallback(tf.keras.callbacks.Callback):

    def __init__(self, validation_data):

        super(tf.keras.callbacks.Callback, self).__init__()

        self.X = validation_data[0]

        self.Y = validation_data[1]

        self.history = []

        

    def on_epoch_end(self, epoch, logs={}):

        pred = self.model.predict(self.X)

        score = cohen_kappa_score(

            np.argmax(self.Y, axis=1), np.argmax(pred, axis=1), labels=[0, 1, 2, 3, 4], weights="quadratic"

        )

        print(("Epoch {0} : QWK : {1}".format(epoch, score)))

        self.history.append(score)

        if(score >= max(self.history)):

            print("Saving Checkpoint: {0}".format(score))

            self.model.save("../Resnet50_bestQWK.h5")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", 

    min_delta=0.0001, patience=3, verbose=1, mode="auto")

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", 

    min_delta=0.0004, patience=2, factor=0.1, min_lr=1e-6, mode="auto", verbose=1)
qwk = QWKCallback(validation_data)

model.fit_generator(

    generator=train_data,

    #steps_per_epochs=EPOCH_STEPS,

    #batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    validation_data=validation_data,

    validation_steps=30#,

    #callbacks=[early_stopping, reduce_lr]

)
filenames = test_data.filenames

classifications = model.predict_generator(test_data, steps=len(filenames))
results = pd.DataFrame({

    "id_code": filenames,

    "diagnosis": np.argmax(classifications, axis=1)

})

results["id_code"] = results["id_code"].map(lambda x: str(x)[:-4].split("/")[4])

results.head()
file_name = "{0}_{1}.csv".format(algo, klass)

results.to_csv("submission.csv", index=False)
results.diagnosis.value_counts()



len(model.layers)