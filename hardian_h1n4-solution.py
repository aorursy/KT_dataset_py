# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline
root_dir = "../input/train"
categoryAmount = {}
categoryAmount["category"] = []
categoryAmount["no_of_images"] = []

sorted_food_dirs = sorted(os.listdir(root_dir))

rows = len(sorted_food_dirs)
cols = 6
fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(15, 25))
fig.suptitle('Random Image from Each Category', fontsize=20)

for i in range(rows):
    food_dir = sorted_food_dirs[i]
    all_files = os.listdir(os.path.join(root_dir, food_dir))
    categoryAmount["category"].append(food_dir)
    categoryAmount["no_of_images"].append(len(all_files))
    for j in range(cols):
        rand_img = np.random.choice(all_files)
        img = plt.imread(os.path.join(root_dir, food_dir, rand_img))
        ax[i][j].imshow(img)
        ec = (0, .6, .1)
        fc = (0, .7, .2)
        ax[i][j].text(0, -20, food_dir, size=13, rotation=0,
                      ha="left", va="top", bbox=dict(boxstyle="round", ec=ec, fc=fc))

plt.setp(ax, xticks=[], yticks=[])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
categoryAmount = pd.DataFrame(categoryAmount)
print("Number of images:", categoryAmount.no_of_images.sum())

categoryAmount.plot(x="category", y="no_of_images", kind="bar", figsize=(15, 6), legend=False)
shapes = []
for i in range(rows):
    food_dir = sorted_food_dirs[i]
    for file in glob.glob(os.path.join(root_dir, food_dir, "*")):
        try:
            img = plt.imread(file)
            shapes.append("x".join([str(x) for x in img.shape]))
        except:
            print(file, "is broken.")

shapesDf = Counter(shapes)
df = pd.DataFrame(list(zip(shapesDf.keys(), shapesDf.values())), columns=["shapes", "no_of_images"])
print(df.shape)

df1 = pd.DataFrame(df.shapes.str.split("x").tolist(), columns=["h", "w", "c"])
df1["no_of_images"] = df.no_of_images
df1["h"] = df1.h.astype(int)
df1["w"] = df1.w.astype(int)
df1["c"] = df1.c.astype(int)

fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(df1.pivot("h", "w", "no_of_images").fillna(0), ax=ax, cmap="YlGnBu")
root_dir = "../input/test"

categoryAmount = {}
categoryAmount["category"] = []
categoryAmount["no_of_images"] = []

shapes = []

for file in glob.glob(os.path.join(root_dir, "*")):
    try:
        img = plt.imread(file)
        shapes.append("x".join([str(x) for x in img.shape]))
    except:
        print(file, "is broken.")
        
print("Number of images:", len(shapes))
            
shapesDf = Counter(shapes)
df = pd.DataFrame(list(zip(shapesDf.keys(), shapesDf.values())), columns=["shapes", "no_of_images"])
print(df.shape)

df1 = pd.DataFrame(df.shapes.str.split("x").tolist(), columns=["h", "w", "c"])
df1["no_of_images"] = df.no_of_images
df1["h"] = df1.h.astype(int)
df1["w"] = df1.w.astype(int)
df1["c"] = df1.c.astype(int)

fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(df1.pivot("h", "w", "no_of_images").fillna(0), ax=ax, cmap="YlGnBu")
import os
from shutil import copyfile
from sklearn.model_selection import StratifiedKFold


train_dir = "../input/train"
output_dir = "../input/train_val_v1"
n_splits = 7
def load_images(train_dir):
    print('split train dir', train_dir)
    all_images = []
    images_names = []
    categories = []

    for category in os.listdir(train_dir):
        for filename in os.listdir(os.path.join(train_dir, category)):
            try:
                all_images.append(os.path.join(train_dir, category, filename))
                images_names.append(filename)
                categories.append(category)
            except:
                print(filename, "is broken.")

    return all_images, images_names, categories


def save_images(dir_path, all_images, images_names, categories, iteration, dtype):
    print(save_images, dir_path, len(all_images), iteration)
    for img, filename, category in zip(all_images, images_names, categories):

        if dtype == "train":

            if not os.path.exists(os.path.join(dir_path, "train_val_%d" % iteration, "train", category)):
                os.makedirs(os.path.join(dir_path, "train_val_%d" % iteration, "train", category))
            copyfile(img, os.path.join(dir_path, "train_val_%d" % iteration, "train", category, filename))

        else:

            if not os.path.exists(os.path.join(dir_path, "train_val_%d" % iteration, "val", category)):
                os.makedirs(os.path.join(dir_path, "train_val_%d" % iteration, "val", category))
            copyfile(img, os.path.join(dir_path, "train_val_%d" % iteration, "val", category, filename))


all_images_dir, images_names, categories = load_images(train_dir)
print('loaded')
# Define a splitter
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2018)
print('defined')
for ii, (train, val) in enumerate(skf.split(all_images_dir, categories)):

    print('iteration', ii, train, val)

    tmp1 = []
    tmp2 = []
    tmp3 = []
    for i in train:
        tmp1.append(all_images_dir[i])
        tmp2.append(images_names[i])
        tmp3.append(categories[i])

    tmp11 = []
    tmp22 = []
    tmp33 = []
    for i in val:
        tmp11.append(all_images_dir[i])
        tmp22.append(images_names[i])
        tmp33.append(categories[i])

    save_images(
        dir_path=output_dir,
        all_images=tmp1,
        images_names=tmp2,
        categories=tmp3,
        iteration=ii,
        dtype="train"
    )

    save_images(
        dir_path=output_dir,
        all_images=tmp11,
        images_names=tmp22,
        categories=tmp33,
        iteration=ii,
        dtype="val"
    )
import gc
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K

from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, LearningRateScheduler, Callback
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, AveragePooling2D
from keras.regularizers import l2
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import multi_gpu_model
from keras.applications import (
    vgg16,
    vgg19,
    inception_v3,
    resnet50,
    inception_resnet_v2,
    xception,
    densenet,
    nasnet
)
class ModelMGPU(Model):
    
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)
def load_default_input_shape(model_type):
    if model_type == "VGG19" or model_type == "VGG16" or model_type == "ResNet50" or "NASNetMobile" in model_type or "DenseNet" in model_type:
        input_shape = (224, 224)
    elif "NASNetLarge" == model_type:
        input_shape = (331, 331)
    else:
        input_shape = (299, 299)
    return input_shape
def load_preprocess_input(model_type):
    if model_type == "VGG19":
        preprocess_input = vgg19.preprocess_input
    elif model_type == "VGG16":
        preprocess_input = vgg16.preprocess_input
    elif model_type == "InceptionResNetV2":
        preprocess_input = inception_resnet_v2.preprocess_input
    elif model_type == "ResNet50":
        preprocess_input = resnet50.preprocess_input
    elif model_type == "Xception":
        preprocess_input = xception.preprocess_input
    elif "DenseNet" in model_type:
        preprocess_input = densenet.preprocess_input
    elif "NASNet" in model_type:
        preprocess_input = nasnet.preprocess_input
    else:
        preprocess_input = inception_v3.preprocess_input
    return preprocess_input
def load_model(model_type, input_shape, n_classes=None, include_top=True, stack_new_layers=True, flatten_fn=Flatten(), dropout_rate=0.5):

    if model_type == "VGG19":
        base_model = vgg19.VGG19(weights="imagenet", include_top=include_top, input_shape=input_shape)
    elif model_type == "VGG16":
        base_model = vgg16.VGG16(weights="imagenet", include_top=include_top, input_shape=input_shape)
    elif model_type == "InceptionResNetV2":
        base_model = inception_resnet_v2.InceptionResNetV2(weights="imagenet", include_top=include_top, input_shape=input_shape)
    elif model_type == "ResNet50":
        base_model = resnet50.ResNet50(weights="imagenet", include_top=include_top, input_shape=input_shape)
    elif model_type == "Xception":
        base_model = xception.Xception(weights="imagenet", include_top=include_top, input_shape=input_shape)
    elif model_type == "DenseNet121":
        base_model = densenet.DenseNet121(weights="imagenet", include_top=include_top, input_shape=input_shape)
    elif model_type == "DenseNet169":
        base_model = densenet.DenseNet169(weights="imagenet", include_top=include_top, input_shape=input_shape)
    elif model_type == "DenseNet201":
        base_model = densenet.DenseNet201(weights="imagenet", include_top=include_top, input_shape=input_shape)
    elif model_type == "NASNetLarge":
        base_model = nasnet.NASNetLarge(weights="imagenet", include_top=include_top, input_shape=input_shape)
    elif model_type == "NASNetMobile":
        base_model = nasnet.NASNetMobile(weights="imagenet", include_top=include_top, input_shape=input_shape)
    else:
        base_model = inception_v3.InceptionV3(weights="imagenet", include_top=include_top, input_shape=input_shape)

    if include_top:

        # Pop last layer to fit class
        base_model.layers.pop()

        # Create last layer
        x = Dense(n_classes, activation="softmax", W_regularizer=l2(.0005), name="predictions")(base_model.layers[-1].output)

    else:

        x = base_model.output

        if stack_new_layers:

            x = AveragePooling2D(pool_size=(8, 8))(x)
            x = Dropout(.5)(x)
            x = Flatten()(x)
            x = Dense(n_classes, activation="softmax", W_regularizer=l2(.0005), name="predictions")(x)

    # Redefine model
    model = Model(inputs=base_model.input, outputs=x)

    return model
class LearningRateTracker(Callback):

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        # If you want to apply decay.
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("Learning rate is", K.eval(lr_with_decay))
def scheduler(epoch):
    if epoch < 10:
        return 0.001
    elif epoch < 20:
        return 0.0005
    elif epoch < 30:
        return 0.0001
    elif epoch < 40:
        return 0.00008
    else:
        return 0.000009
train_val_dir = "../input/train_val_v1"
output_dir = "../outputs"
logs_dir = "../logs"
model_dir = os.path.join(output_dir, 'saved_models')
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
if not os.path.isdir(model_dir) or not os.path.exists(model_dir):
    os.makedirs(model_dir)
def train(model_type):
    
    # Load default input shape and preprocess_input
    input_shape = load_default_input_shape(model_type)
    preprocess_input = load_preprocess_input(model_type)

    # Configs
    include_top = True
    stack_new_layers = True
    dropout_rate = 0.5
    n_classes = 18
    no_of_gpus = 1

    # Training Config
    n_splits = 7  # No of split for skfold cross validation
    batch_size = 32  # No of samples fit every step
    epochs = 50  # No of epochs
    lr = 0.005  # Optimizer learning rate

    # Training
    print("Start cross-validation training...")
    histories = []
    for iteration in range(n_splits):

        # Define model name
        model_name = '%s_model_{epoch:03d}_{val_acc:.2f}_iter%d.h5' % (model_type, iteration)
        filepath = os.path.join(model_dir, model_name)

        # Load model
        if no_of_gpus > 1:

            with tf.device("/cpu:0"):
                model = load_model(
                    model_type,
                    input_shape=input_shape + (3,),
                    n_classes=n_classes,
                    include_top=include_top,
                    stack_new_layers=stack_new_layers,
                    dropout_rate=dropout_rate
                )
                
            model = ModelMGPU(model, no_of_gpus)

        else:
            
            model = load_model(
                model_type,
                input_shape=input_shape + (3,),
                n_classes=n_classes,
                include_top=include_top,
                stack_new_layers=stack_new_layers,
                dropout_rate=dropout_rate
            )

        # Define optimizer
        sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9)

        # compile the model
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

        # Prepare callbacks
        checkpoint = ModelCheckpoint(
            filepath=filepath,
            monitor='val_acc',
            verbose=1,
            save_best_only=True
        )

        lr_reducer = ReduceLROnPlateau(
            monitor="val_acc",
            factor=np.sqrt(0.1),
            cooldown=0,
            patience=1,
            min_lr=0.5e-6
        )

        csv_logger = CSVLogger(os.path.join(logs_dir, "%s_training_iter%d.csv" % (model_type, iteration)))

        early_stopping = EarlyStopping(
            monitor='val_acc',
            min_delta=0,
            patience=4,
            verbose=0,
            mode='auto'
        )

        lr_scheduler = LearningRateScheduler(scheduler)

        lr_tracker = LearningRateTracker()

        callbacks = [checkpoint, lr_reducer, csv_logger, early_stopping, lr_scheduler, lr_tracker]

        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-6,
            rotation_range=0,
            width_shift_range=0.,
            height_shift_range=0.,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=preprocess_input,
            data_format=K.image_data_format()
        )

        train_generator = datagen.flow_from_directory(
            directory=os.path.join(train_val_dir, "train_val_%d" % iteration, "train"),
            target_size=input_shape,
            class_mode="categorical",
            batch_size=batch_size,
            seed=2018
        )

        train_val_generator = datagen.flow_from_directory(
            directory=os.path.join(train_val_dir, "train_val_%d" % iteration, "val"),
            target_size=input_shape,
            class_mode="categorical",
            batch_size=batch_size,
            seed=2018
        )

        history = model.fit_generator(
            generator=train_generator,
            validation_data=train_val_generator,
            epochs=epochs,
            verbose=1,
            workers=8,
            callbacks=callbacks,
        )

        histories.append(history)

        K.clear_session()
for model_type in ["Xception", "InceptionV3", "InceptionResNetV2", "ResNet50", "DenseNet121", "DenseNet169", "DenseNet201", "NASNetLarge", "NASNetMobile"]:
    train(model_type)
import gc
import os
import glob
import pandas as pd
import numpy as np

import keras.backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

train_val_dir = "../input/train_val_v1"
testDataset = "../meta/mapTest.csv"

# Training Config
batch_size = 32  # No of samples fit every step
n_classes = 18

# Read pre-generated dataset comprising of 3 columns (file, category, category_id)
tDf = pd.read_csv(testDataset)

input_prev = None
for model_type in ["DenseNet121", 'DenseNet169', "DenseNet201", "NASNetMobile", "ResNet50", "InceptionResNetV2", "InceptionV3", "Xception", "NASNetLarge"]:

    if model_type == "NASNetLarge":
        del Xtest, broken_test_imgs
        gc.collect()
        batch_size = 16

    # Load preprocessor based on model_type
    preprocess_input = load_preprocess_input(model_type)
    print(preprocess_input)

    input_shape = load_default_input_shape(model_type)
    print("Input_shape:", input_shape)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=0,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        cval=0.,
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=preprocess_input,
        data_format=K.image_data_format()
    )

    # Load test images
    if input_shape != input_prev:
        Xtest, broken_test_imgs = load_images(tDf.file, input_shape=input_shape)
        input_prev = input_shape

        # Convert and process test images
        Xtest = np.asarray(Xtest)
        Xtest = preprocess_input(Xtest)

    valDf = pd.DataFrame()

    val_dir = os.path.join("../outputs/val/", model_type)
    test_dir = os.path.join("../outputs/test/", model_type)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for f in sorted(glob.glob(os.path.join("../outputs/saved_models", model_type, "*")), key=lambda x: x[-4]):

        print("Model:", f)
        iteration = f[-4]

        model = load_model(f)

        print("Generating validation data...")
        val_generator = datagen.flow_from_directory(
            directory=os.path.join(train_val_dir, "train_val_%s" % iteration, "val"),
            target_size=input_shape,
            class_mode="categorical",
            batch_size=100000,
            seed=2018
        )

        for Xval, label in val_generator:
            val_predictions = model.predict(Xval)
            valDf = pd.concat([
                valDf,
                pd.DataFrame(
                    np.hstack([
                        val_predictions,
                        np.argmax(label, axis=-1)[:, np.newaxis]
                    ]),
                    columns=["f"+str(x) for x in range(n_classes)] + ["category_id"])
            ])
            break

        del Xval, label, val_predictions
        gc.collect()

        print("Generating prediction on test data...")
        test_predictions = model.predict(Xtest)
        testDf = pd.DataFrame({"id": tDf["id"]})
        testDf = pd.concat([testDf, pd.DataFrame(test_predictions, columns=[model_type+"_f"+str(x) for x in range(n_classes)])], axis=1)
        testDf.to_csv(
            os.path.join(test_dir, "%s_test_iter%s.csv" % (model_type, iteration)),
            index=False
        )

        del test_predictions, testDf
        gc.collect()

    valDf.to_csv(os.path.join(val_dir, "%s_val.csv" % model_type), index=False)
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score

%matplotlib inline
InceptionV3 = pd.read_csv("./outputs/val/v3/InceptionV3_val.csv")
InceptionResNetV2 = pd.read_csv("./outputs/val/v3/InceptionResNetV2_val.csv")
Xception = pd.read_csv("./outputs/val/v3/Xception_val.csv")
ResNet50 = pd.read_csv("./outputs/val/v3/ResNet50_val.csv")
DenseNet201 = pd.read_csv("./outputs/val/v3/DenseNet201_val.csv")
DenseNet169 = pd.read_csv("./outputs/val/v3/DenseNet169_val.csv")
DenseNet121 = pd.read_csv("./outputs/val/v3/DenseNet121_val.csv")
NASNetLarge = pd.read_csv("./outputs/val/v3/NASNetLarge_val.csv")
NASNetMobile = pd.read_csv("./outputs/val/v3/NASNetMobile_val.csv")
df = {}
# InceptionV3

fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(16, 24))

y_true = InceptionV3.category_id
y_pred = np.argmax(InceptionV3.drop("category_id", axis=1).as_matrix(), axis=-1)
print("InceptionV3 acc:", accuracy_score(y_true, y_pred))

sns.heatmap(confusion_matrix(y_true, y_pred), cmap="YlGnBu", ax=ax[0][0], annot=True, fmt="d")
ax[0][0].set_title("InceptionV3")

df["InceptionV3"] = y_pred

# InceptionResNetV2

y_true = InceptionResNetV2.category_id
y_pred = np.argmax(InceptionResNetV2.drop("category_id", axis=1).as_matrix(), axis=-1)
print("InceptionResNetV2 acc:", accuracy_score(y_true, y_pred))

sns.heatmap(confusion_matrix(y_true, y_pred), cmap="YlGnBu", ax=ax[0][1], annot=True, fmt="d")
ax[0][1].set_title("InceptionResNetV2")

df["InceptionResNetV2"] = y_pred

# Xception

y_true = Xception.category_id
y_pred = np.argmax(Xception.drop("category_id", axis=1).as_matrix(), axis=-1)
print("Xception acc:", accuracy_score(y_true, y_pred))

sns.heatmap(confusion_matrix(y_true, y_pred), cmap="YlGnBu", ax=ax[1][0], annot=True, fmt="d")
ax[1][0].set_title("Xception")

df["Xception"] = y_pred

# ResNet50

y_true = ResNet50.category_id
y_pred = np.argmax(ResNet50.drop("category_id", axis=1).as_matrix(), axis=-1)
print("ResNet50 acc:", accuracy_score(y_true, y_pred))

sns.heatmap(confusion_matrix(y_true, y_pred), cmap="YlGnBu", ax=ax[1][1], annot=True, fmt="d")
ax[1][1].set_title("ResNet50")

df["ResNet50"] = y_pred

# DenseNet201

y_true = DenseNet201.category_id
y_pred = np.argmax(DenseNet201.drop("category_id", axis=1).as_matrix(), axis=-1)
print("DenseNet201 acc:", accuracy_score(y_true, y_pred))

sns.heatmap(confusion_matrix(y_true, y_pred), cmap="YlGnBu", ax=ax[2][0], annot=True, fmt="d")
ax[2][0].set_title("DenseNet201")

df["DenseNet201"] = y_pred

# DenseNet169

y_true = DenseNet169.category_id
y_pred = np.argmax(DenseNet169.drop("category_id", axis=1).as_matrix(), axis=-1)
print("DenseNet169 acc:", accuracy_score(y_true, y_pred))

sns.heatmap(confusion_matrix(y_true, y_pred), cmap="YlGnBu", ax=ax[2][1], annot=True, fmt="d")
ax[2][1].set_title("DenseNet169")

df["DenseNet169"] = y_pred

# DenseNet121

y_true = DenseNet121.category_id
y_pred = np.argmax(DenseNet121.drop("category_id", axis=1).as_matrix(), axis=-1)
print("DenseNet121 acc:", accuracy_score(y_true, y_pred))

sns.heatmap(confusion_matrix(y_true, y_pred), cmap="YlGnBu", ax=ax[3][0], annot=True, fmt="d")
ax[3][0].set_title("DenseNet121")

df["DenseNet121"] = y_pred

# NASNetLarge

y_true = NASNetLarge.category_id
y_pred = np.argmax(NASNetLarge.drop("category_id", axis=1).as_matrix(), axis=-1)
print("NASNetLarge acc:", accuracy_score(y_true, y_pred))

sns.heatmap(confusion_matrix(y_true, y_pred), cmap="YlGnBu", ax=ax[3][1], annot=True, fmt="d")
ax[3][1].set_title("NASNetLarge")

df["NASNetLarge"] = y_pred

# NASNetMobile

fig, ax = plt.subplots(figsize=(6.5, 5))

y_true = NASNetMobile.category_id
y_pred = np.argmax(NASNetMobile.drop("category_id", axis=1).as_matrix(), axis=-1)
print("NASNetMobile acc:", accuracy_score(y_true, y_pred))

sns.heatmap(confusion_matrix(y_true, y_pred), cmap="YlGnBu", ax=ax, annot=True, fmt="d")
ax.set_title("NASNetMobile")

df["NASNetMobile"] = y_pred

plt.tight_layout()
df = pd.DataFrame(df)
df.head()
fig, ax = plt.subplots(figsize=(18, 14))
sns.heatmap(df.corr(), ax=ax)
df = {}
for path in os.listdir("./outputs/test"):
    for f in glob.glob(os.path.join("./outputs/test", path, "*")):
        df[path + "_" + f[-5]] = np.argmax(pd.read_csv(f).drop("id", axis=1).as_matrix(), axis=1)
df = pd.DataFrame(df)
df.head()
fig, ax = plt.subplots(figsize=(18, 14))
sns.heatmap(df.corr(), ax=ax)
import numpy
import csv

category_count = 18
test_rows = 16111
OUTPUT_FILE = '../outputs/submissions/result.csv'

test_outputs = {
    'ResNet50': {
        'type': 'TEST_ITER',
        'weight': 1,
        'file_paths': [
            # "../outputs/test/ResNet50/ResNet50_test_iter0.csv",
            "../outputs/test/ResNet50/ResNet50_test_iter1.csv",
            "../outputs/test/ResNet50/ResNet50_test_iter2.csv",
            "../outputs/test/ResNet50/ResNet50_test_iter3.csv",
            # "../outputs/test/ResNet50/ResNet50_test_iter4.csv",
            # "../outputs/test/ResNet50/ResNet50_test_iter5.csv",
        ],
    },
    'InceptionResNetV2': {
        'type': 'TEST_ITER',
        'weight': 1,
        'file_paths': [
            "../outputs/test/InceptionResNetV2/InceptionResNetV2_test_iter0.csv",
            "../outputs/test/InceptionResNetV2/InceptionResNetV2_test_iter1.csv",
            "../outputs/test/InceptionResNetV2/InceptionResNetV2_test_iter2.csv",
            "../outputs/test/InceptionResNetV2/InceptionResNetV2_test_iter3.csv",
            "../outputs/test/InceptionResNetV2/InceptionResNetV2_test_iter5.csv",
            "../outputs/test/InceptionResNetV2/InceptionResNetV2_test_iter6.csv",
        ],
    },
    'InceptionV3': {
        'type': 'TEST_ITER',
        'weight': 1,
        'file_paths': [
            "../outputs/test/InceptionV3/InceptionV3_test_iter0.csv",
            "../outputs/test/InceptionV3/InceptionV3_test_iter1.csv",
            # "../outputs/test/InceptionV3/InceptionV3_test_iter2.csv",
            "../outputs/test/InceptionV3/InceptionV3_test_iter3.csv",
            # "../outputs/test/InceptionV3/InceptionV3_test_iter4.csv",
            "../outputs/test/InceptionV3/InceptionV3_test_iter5.csv",
            "../outputs/test/InceptionV3/InceptionV3_test_iter6.csv",
        ],
    },
    'DenseNet201': {
        'type': 'TEST_ITER',
        'weight': 1,
        'file_paths': [
            "../outputs/test/DenseNet201/DenseNet201_test_iter0.csv",
            "../outputs/test/DenseNet201/DenseNet201_test_iter1.csv",
            "../outputs/test/DenseNet201/DenseNet201_test_iter2.csv",
            "../outputs/test/DenseNet201/DenseNet201_test_iter3.csv",
            "../outputs/test/DenseNet201/DenseNet201_test_iter4.csv",
            "../outputs/test/DenseNet201/DenseNet201_test_iter5.csv",
            "../outputs/test/DenseNet201/DenseNet201_test_iter6.csv",
        ],
    },
    'DenseNet121': {
        'type': 'TEST_ITER',
        'weight': 1,
        'file_paths': [
            # "../outputs/test/DenseNet121/DenseNet121_test_iter0.csv",
            # "../outputs/test/DenseNet121/DenseNet121_test_iter1.csv",
            "../outputs/test/DenseNet121/DenseNet121_test_iter2.csv",
            "../outputs/test/DenseNet121/DenseNet121_test_iter3.csv",
            # "../outputs/test/DenseNet121/DenseNet121_test_iter4.csv",
            # "../outputs/test/DenseNet121/DenseNet121_test_iter5.csv",
            # "../outputs/test/DenseNet121/DenseNet121_test_iter6.csv",
        ],
    },
    'DenseNet169': {
        'type': 'TEST_ITER',
        'weight': 0.5,
        'file_paths': [
            "../outputs/test/DenseNet169/DenseNet169_test_iter0.csv",
            "../outputs/test/DenseNet169/DenseNet169_test_iter1.csv",
            "../outputs/test/DenseNet169/DenseNet169_test_iter2.csv",
            # "../outputs/test/DenseNet169/DenseNet169_test_iter3.csv",
            # "../outputs/test/DenseNet169/DenseNet169_test_iter4.csv",
            "../outputs/test/DenseNet169/DenseNet169_test_iter5.csv",
            # "../outputs/test/DenseNet169/DenseNet169_test_iter6.csv",
        ],
    },
    'Xception': {
        'type': 'TEST_ITER',
        'weight': 2,
        'file_paths': [
            "../outputs/test/Xception/Xception_test_iter0.csv",
            "../outputs/test/Xception/Xception_test_iter1.csv",
            "../outputs/test/Xception/Xception_test_iter2.csv",
            "../outputs/test/Xception/Xception_test_iter3.csv",
            "../outputs/test/Xception/Xception_test_iter4.csv",
            "../outputs/test/Xception/Xception_test_iter5.csv",
            "../outputs/test/Xception/Xception_test_iter6.csv",
        ],
    },
    'NASNetLarge': {
        'type': 'TEST_ITER',
        'weight': 1,
        'file_paths': [
            "../outputs/test/NASNetLarge/NASNetLarge_test_iter1.csv",
            "../outputs/test/NASNetLarge/NASNetLarge_test_iter2.csv",
            "../outputs/test/NASNetLarge/NASNetLarge_test_iter3.csv",
            # "../outputs/test/NASNetLarge/NASNetLarge_test_iter6.csv",
            "../outputs/test/NASNetLarge/NASNetLarge_test_iter0.csv",
            "../outputs/test/NASNetLarge/NASNetLarge_test_iter4.csv",
        ],
    },
}


def voting_iter(rows, maps, weight=1):
    for row_num, row in enumerate(rows):
        ans = numpy.array([float(x) for x in row]).argmax(axis=0)
        maps[row_num][ans] += 1.0 * weight


def voting_result(rows, maps, weight=1):
    for row_num, row in enumerate(rows):
        maps[row_num][row[1]] += 1.0 * weight


answer_maps = [[0] * category_count for _ in range(test_rows)]

for key, value in test_outputs.items():

    output_type = value['type']

    if output_type == 'TEST_ITER':
        weight, file_paths = float(value['weight']), value['file_paths']
        single_answer_maps = [[0] * category_count for _ in range(test_rows)]
        for file_path in file_paths:
            file = open(file_path)
            reader = csv.reader(file)
            rows = [row[1:] for row in reader][1:]
            voting_iter(rows, single_answer_maps)
        voting_iter(single_answer_maps, answer_maps, weight)

    elif output_type == 'TEST_RESULT':
        weight, file_path = float(value['weight']), value['file_path']
        file = open(file_path)
        reader = csv.reader(file)
        rows = [row for row in reader][1:]
        rows = [map(int, row) for row in rows]
        voting_result(rows, answer_maps, weight)

result = []
for row_num, row in enumerate(answer_maps):
    row = [float(x) for x in row]
    ans = numpy.array(row).argmax(axis=0)
    maxs = row[ans]
    same_ans = [idx for idx, x in enumerate(row) if x == maxs]
    if len(same_ans) > 1:
        print('answer for question number {} has {} candidates: {}, choosing {}'.format(row_num + 1, len(same_ans), same_ans, ans))
    result.append([row_num + 1, ans])

file = open(OUTPUT_FILE, 'w')
writer = csv.writer(file)
writer.writerows([['id', 'category']] + result)