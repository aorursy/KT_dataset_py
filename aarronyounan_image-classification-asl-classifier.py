# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
import os
import string
import glob
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
trainingGenerator = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1/255.,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.25,
    dtype=None)
trainingSet = trainingGenerator.flow_from_directory("/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/",
    target_size=(256, 256),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=128,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="png",
    follow_links=False,
    subset="training",
    interpolation="nearest")
trainingSet.class_indices
validationSet= trainingGenerator.flow_from_directory("/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/",
    target_size=(256, 256),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=128,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="png",
    follow_links=False,
    subset="validation",
    interpolation="nearest")
batch = trainingSet.next()
plt.imshow(batch[0][0])
print(np.argmax(batch[1][0],axis=-1))
trainingSet.reset()
labels = dict(zip(range(0,26),string.ascii_uppercase))
labels[26]="del"
labels[27]="nothing"
labels[28]="space"
labels
fileList=[fileName for fileName in glob.glob('/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/*.jpg')]
fileList.append('/kaggle/input/asltesting/del_test.jpg')
fileList
labelsFromFile = [file.split("/")[-1].split("_")[0] for file in fileList]
labelsFromFile
zippedList = zip(fileList,labelsFromFile)
df = pd.DataFrame(zip(fileList,labelsFromFile),columns=["filename","label"])
df
testingGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=(1.0/255))
testingSet = testingGenerator.flow_from_dataframe(df,
    x_col="filename",
    y_col="label",
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=False,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="png",
    subset=None,
    interpolation="nearest",
    validate_filenames=True,
)
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)
model = make_model(input_shape=trainingSet.target_size+(3,),num_classes=trainingSet.num_classes)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
callbacks = [
    keras.callbacks.ModelCheckpoint("/kaggle/working/save_at_{epoch}.h5",save_best_only=True,verbose=1,),
]
model.fit(trainingSet, epochs=4, callbacks=callbacks, validation_data=validationSet,)
#model.load_weights('/kaggle/working/save_at_3.h5')
model.load_weights('/kaggle/working/save_at_4.h5')
evaluation = model.predict_generator(testingSet)
df["pred"]=[labels[prediction] for prediction in np.argmax(evaluation,axis=-1)]
df
precision_score(y_true=df.label,y_pred=df.pred,average='micro')
recall_score(y_true=df.label,y_pred=df.pred,average='micro')
demoList_1=["/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/H_test.jpg","/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/O_test.jpg","/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/L_test.jpg","/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/A_test.jpg","/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/space_test.jpg","/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/V_test.jpg","/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/I_test.jpg","/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/S_test.jpg","/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/T_test.jpg","/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/A_test.jpg","/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/space_test.jpg","/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/M_test.jpg","/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/X_test.jpg"]
for file in demoList_1:
    img = keras.preprocessing.image.load_img(file, target_size=(256,256))
    img_array = keras.preprocessing.image.img_to_array(img)/255
    img_data = tf.expand_dims(img_array, 0)  # Create batch axis   
    predictions = model.predict(img_data)
    plt.title(labels[np.argmax(predictions)])
    plt.imshow(img_array)
    plt.show()
    
demoList_2=["/kaggle/input/asltesting/C.jpg","/kaggle/input/asltesting/O.jpg","/kaggle/input/asltesting/M.jpg","/kaggle/input/asltesting/O3.jpg","/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/space_test.jpg","/kaggle/input/asltesting/E.jpg","/kaggle/input/asltesting/S.jpg","/kaggle/input/asltesting/T2.jpg","/kaggle/input/asltesting/A.jpg","/kaggle/input/asltesting/S1.jpg","/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/space_test.jpg","/kaggle/input/asltesting/T.jpg","/kaggle/input/asltesting/O2.jpg","/kaggle/input/asltesting/D.jpg","/kaggle/input/asltesting/O2.jpg","/kaggle/input/asltesting/S2.jpg"]
for file in demoList_2:
    img = keras.preprocessing.image.load_img(file, target_size=(256,256))
    img_array = keras.preprocessing.image.img_to_array(img)/255
    img_data = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_data)
    plt.title(labels[np.argmax(predictions)])
    plt.imshow(img_array)
    plt.show()