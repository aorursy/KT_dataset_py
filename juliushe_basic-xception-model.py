import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

data_dir = "/kaggle/input/sarscov2-ctscan-dataset/"
classes = ["COVID", "non-COVID"]

# create a dataframe containing image and meta data
data = pd.DataFrame()
for c in classes:
    for file in os.listdir(os.path.join(data_dir, c)):
        img = cv2.imread(os.path.join(data_dir, c, file))
        h, w, _ = img.shape
        data = data.append({"path": os.path.join(c, file),
                            "name": file,
                            "class": c,
                            "label": 1 if c == "COVID" else 0,
                            "shape": img.shape,
                            "height": h,
                            "width": w,
                            "image_raw": img}, ignore_index=True)

print(data.head())
print(data["class"].value_counts())
print(data["height"].nlargest(3))
print(data["width"].nlargest(3))
print(data["height"].nsmallest(3))
print(data["width"].nsmallest(3))
fig, (ax1, ax2) = plt.subplots(1, 2)

sample1 = data[data["label"]==1].sample()
img1 = sample1["image_raw"].iloc[0]
ax1.imshow(img1)
ax1.set_title(sample1["name"].iloc[0])

sample2 = data[data["label"]==0].sample()
img2 = sample2["image_raw"].iloc[0]
ax2.imshow(img2)
ax2.set_title(sample2["name"].iloc[0])

fig.show()
# resize images
dest_shape = (128, 128)
def resize(img):
    return cv2.resize(img, dest_shape)
data["image_resized"] = data["image_raw"].apply(resize)
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf

X, y = data["image_resized"], data["label"]
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in splitter.split(X, y):
    X_train, y_train = X.iloc[train_idx].values, y.iloc[train_idx].values
    X_val, y_val = X.iloc[val_idx].values, y.iloc[val_idx].values

X_train, y_train = tf.convert_to_tensor(np.stack(X_train), dtype=tf.float32), tf.convert_to_tensor(y_train, dtype=tf.int32)
X_val, y_val = tf.convert_to_tensor(np.stack(X_val), dtype=tf.float32), tf.convert_to_tensor(y_val, dtype=tf.int32)

print("Training images: {} ({} COVID)".format(len(X_train), np.sum(y_train)))
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

batch_size = 32

# explore different options of keras image data generator
train_datagen = ImageDataGenerator(
        rescale=1/255.,
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1/255.)

train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size, seed=12)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size, seed=12)
from keras.regularizers import l2

dropout_prob = 0.2

# load Xception model with pre-trained weights
base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(128, 128, 3),
    include_top=False)

base_model.trainable = False

# define small classification module on top
inputs = keras.Input(shape=(128, 128, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = keras.layers.Dropout(dropout_prob)(x)
x = keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
x = keras.layers.Dropout(dropout_prob)(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(),
              metrics=[keras.metrics.BinaryAccuracy()])

print(model.summary())
# train model
hist = model.fit_generator(
            train_generator,
            steps_per_epoch=2000 // batch_size,
            epochs=50,
            validation_data=val_generator,
            validation_steps=800 // batch_size)
# loss plot
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# accuracy plot 
plt.plot(hist.history['binary_accuracy'])
plt.plot(hist.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
from sklearn.metrics import roc_curve, roc_auc_score

final_loss, final_accuracy = model.evaluate(X_val / 255., y_val)
print("Final Validation Set Accuracy: {}".format(final_accuracy))

# compute model output
y_pred = model.predict(X_val / 255.)
fpr, tpr, thresholds = roc_curve(y_val, y_pred, pos_label=1)
auc = roc_auc_score(y_val, y_pred)

# stolen from
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()