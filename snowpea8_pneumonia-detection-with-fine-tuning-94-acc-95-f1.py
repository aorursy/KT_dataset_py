!pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index

!pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index



import efficientnet.tfkeras as efn
import os

import random

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from mlxtend.plotting import plot_confusion_matrix



from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import tensorflow as tf

from keras.utils.vis_utils import plot_model

import tensorflow_addons as tfa



def seed_everything(seed=2020):

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



SEED = 42

seed_everything(SEED)



sns.set_style("whitegrid")

mpl.rc("axes", labelsize=14)

mpl.rc("xtick", labelsize=12)

mpl.rc("ytick", labelsize=12)

palette_ro = ["#ee2f35", "#fa7211", "#fbd600", "#75c731", "#1fb86e", "#0488cf", "#7b44ab"]



ROOT = "../input/chest-xray-pneumonia/chest_xray/"

print(os.listdir(ROOT))
train_dir = ROOT + "train/"

val_dir = ROOT + "val/"

test_dir = ROOT + "test/"



IMG_SIZE = 224

# b0: 224, b1: 240, b2: 260, b3: 300, b4: 380, b5: 456, b6: 528, b7: 600

BATCH_SIZE = 64



d1 = 64

d2 = 32

EPOCHS_1 = 7



fine_tune_at = 45

EPOCHS_2 = 10
print("Train dataset:")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    train_dir,

    # validation_split=0.2,

    # subset="training",

    seed=SEED,

    image_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE)



print("Validation dataset:")

val_ds = tf.keras.preprocessing.image_dataset_from_directory(

    val_dir,

    seed=SEED,

    image_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE)
class_count = pd.DataFrame(index=["label", "count"])



for _class in train_ds.class_names:

    print("{:<14}{} images".format(_class, len(os.listdir(train_dir + _class))))

    class_count[_class] = [_class, len(os.listdir(train_dir + _class))]



# NUM_CLASSES = len(train_ds.class_names)



fig, ax = plt.subplots(1, 1, figsize=(8, 5))

sns.barplot(x="label", y="count", data=class_count.T, ax=ax, palette=palette_ro[1::5])

fig.suptitle("Train dataset distribution", fontsize=18);
plt.figure(figsize=(12, 12))

for images, labels in train_ds.take(1):

    for i in range(16):

        ax = plt.subplot(4, 4, i+1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(train_ds.class_names[labels[i]])

        plt.axis("off")
sample = next(iter(train_ds))[0][0].numpy().astype("uint8")

plt.imshow(sample)

plt.axis("off");
data_augmentation = tf.keras.Sequential([

    tf.keras.layers.experimental.preprocessing.RandomRotation(0.02, seed=SEED),

    tf.keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1, seed=SEED),

    tf.keras.layers.experimental.preprocessing.RandomZoom(0.33, seed=SEED),

    tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal", seed=SEED),

    # tf.keras.layers.experimental.preprocessing.RandomHeight(0.1, interpolation="nearest", seed=SEED),

    # tf.keras.layers.experimental.preprocessing.RandomWidth(0.1, interpolation="nearest", seed=SEED),

    # tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE, interpolation="nearest"),

    # tf.keras.layers.experimental.preprocessing.RandomContrast(0.1, 1.9)

])



sample_b = tf.expand_dims(sample, 0)



plt.figure(figsize=(10, 10))

for i in range(9):

    result = data_augmentation(sample_b)

    ax = plt.subplot(3, 3, i + 1)

    plt.imshow(result[0])

    plt.axis("off")
def build_model():

    inp = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3))

    x =  tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)(inp)

    x = data_augmentation(x)

    

    ef = efn.EfficientNetB0(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights="imagenet", input_tensor=x)

    ef.trainable = False

    

    x = tf.keras.layers.GlobalAveragePooling2D()(ef.output)

    #x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.32)(x)

    x = tf.keras.layers.Dense(d1, activation="elu")(x)

    # x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Dense(d2, activation="elu")(x)

    # x = tf.keras.layers.BatchNormalization()(x)

    # x = tf.keras.layers.Dropout(0.2)(x)

    

    y = tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer="he_normal")(x)

    

    model = tf.keras.Model(inputs=inp, outputs=y)

    

    return model



model = build_model()



print("Number of layers in the model: ", len(model.layers))

model.summary()
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
model.compile(optimizer=tfa.optimizers.RectifiedAdam(),

              loss="binary_crossentropy",

              metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])



history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS_1

)
acc = history.history["accuracy"]

val_acc = history.history["val_accuracy"]



loss = history.history["loss"]

val_loss = history.history["val_loss"]



prec = history.history["precision"]

val_prec = history.history["val_precision"]



rec = history.history["recall"]

val_rec = history.history["val_recall"]



epochs_range = range(EPOCHS_1)



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))



ax1.plot(epochs_range, acc, label="Training Accuracy")

ax1.plot(epochs_range, val_acc, label="Validation Accuracy")

ax1.legend(loc="lower right")

ax1.set_title("Training and Validation Accuracy", fontsize=16)



ax2.plot(epochs_range, loss, label="Training Loss")

ax2.plot(epochs_range, val_loss, label="Validation Loss")

ax2.legend(loc="upper right")

ax2.set_title("Training and Validation Loss", fontsize=16)



ax3.plot(epochs_range, prec, label="Training Precision")

ax3.plot(epochs_range, val_prec, label="Validation Precision")

ax3.legend(loc="lower right")

ax3.set_title("Training and Validation Precision", fontsize=16)



ax4.plot(epochs_range, rec, label="Training Recall")

ax4.plot(epochs_range, val_rec, label="Validation Recall")

ax4.legend(loc="lower right")

ax4.set_title("Training and Validation Recall", fontsize=16)



plt.tight_layout()

plt.show()
def unfreeze_model(model):

    for layer in model.layers[-fine_tune_at:]:

        if not isinstance(layer, tf.keras.layers.BatchNormalization):

            layer.trainable = True



    model.compile(optimizer=tfa.optimizers.RectifiedAdam(),

                  loss="binary_crossentropy",

                  metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])



unfreeze_model(model)
history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS_2

)
acc = history.history["accuracy"]

val_acc = history.history["val_accuracy"]



loss = history.history["loss"]

val_loss = history.history["val_loss"]



prec = history.history["precision_1"]

val_prec = history.history["val_precision_1"]



rec = history.history["recall_1"]

val_rec = history.history["val_recall_1"]



epochs_range = range(EPOCHS_2)



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))



ax1.plot(epochs_range, acc, label="Training Accuracy")

ax1.plot(epochs_range, val_acc, label="Validation Accuracy")

ax1.legend(loc="lower right")

ax1.set_title("Training and Validation Accuracy", fontsize=16)



ax2.plot(epochs_range, loss, label="Training Loss")

ax2.plot(epochs_range, val_loss, label="Validation Loss")

ax2.legend(loc="upper right")

ax2.set_title("Training and Validation Loss", fontsize=16)



ax3.plot(epochs_range, prec, label="Training Precision")

ax3.plot(epochs_range, val_prec, label="Validation Precision")

ax3.legend(loc="lower right")

ax3.set_title("Training and Validation Precision", fontsize=16)



ax4.plot(epochs_range, rec, label="Training Recall")

ax4.plot(epochs_range, val_rec, label="Validation Recall")

ax4.legend(loc="lower right")

ax4.set_title("Training and Validation Recall", fontsize=16)



plt.tight_layout()

plt.show()
print("Test dataset:")

test_ds = tf.keras.preprocessing.image_dataset_from_directory(

    test_dir,

    seed=SEED,

    image_size=(IMG_SIZE, IMG_SIZE),

)



test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
y_test = []



for i in range(len(list(test_ds))):

    y_test += list(list(test_ds)[i][1].numpy())
preds = model.predict(test_ds)

y_pred = preds > 0.5



fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["NORMAL", "PNEUMONIA"])



plt.title("Confusion Matrix", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1-score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), ["NORMAL", "PNEUMONIA"], fontsize=16)

plt.yticks(np.arange(2), ["NORMAL", "PNEUMONIA"], fontsize=16);



print("Classification report on test data")

print(classification_report(y_test, y_pred))