import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook as tqdm

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import *

from glob import glob
!ls /kaggle/input/land-cover-class/
!ls /kaggle/input/land-cover-class/train
test_images = sorted([fn.split('/')[-1] for fn in glob('/kaggle/input/land-cover-class/test/*.jpg')])
test_images[:5]
len(test_images)
train_df = pd.read_csv('/kaggle/input/land-cover-class/train.csv')
train_df.head()
train_dir = '/kaggle/input/land-cover-class/train/'

test_dir = '/kaggle/input/land-cover-class/test/'

img = plt.imread(f'{train_dir}1.jpg')
img.shape
img
plt.imshow(img); train_df['class'][0]
classes = sorted(train_df['class'].unique())

classes
fig, axs = plt.subplots(3, 4, figsize=(15, 12))

rand_idxs = np.random.randint(1, len(train_df)+1, 9)

for c, ax in zip(classes, axs.flat):

    fn = train_df[train_df['class'] == c].sample().iloc[0]['fn']

    img = plt.imread(f'{train_dir}/{fn}')

    ax.imshow(img)

    ax.set_title(c)

for ax in axs.flat:

    ax.set_axis_off()

plt.tight_layout()
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2, rescale=1./255)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    directory=train_dir,

    batch_size=32,

    x_col='fn',

    y_col='class',

    target_size=(64, 64), 

    subset='training'

)



valid_generator = train_datagen.flow_from_dataframe(

    train_df, 

    directory=train_dir,

    batch_size=32,

    x_col='fn',

    y_col='class',

    target_size=(64, 64), 

    subset='validation'

)



test_generator = test_datagen.flow_from_directory(

    directory='/kaggle/input/land-cover-class/',

    classes=['test'],

    batch_size=32,

    target_size=(64, 64), 

    shuffle=False

)
X, y = train_generator[0]
X.shape, y.shape
train_generator.class_indices
y[:3]
p = np.random.rand(10) * 10

plt.bar(np.arange(10), p);
p.sum()
pp = p / p.sum()
pp
pp.sum()
pexp = np.exp(p)

plt.bar(np.arange(10), pexp);
def softmax(a):

    return np.exp(a) / np.exp(a).sum()
p = softmax(p)

p.sum()
plt.bar(np.arange(10), p);
y[0], p
plt.bar(np.arange(10), p)

plt.bar(np.arange(10), y[0], zorder=0.5);
a = np.linspace(0, 1, 100)

true = 1
loss = - true * np.log(a)
plt.plot(a, loss);
(- y[0] * np.log(p)).sum()
model = tf.keras.Sequential([

    Input(shape=(64, 64, 3)),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(256, activation='relu'),

    Dense(128, activation='relu'),

    Dense(10, activation='softmax'),

])
model.summary()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=3, validation_data=valid_generator)
model.fit(train_generator, epochs=3, validation_data=valid_generator)
preds = model.predict(X)
i=2

plt.bar(np.arange(10), preds[i], alpha=0.5)

plt.bar(np.arange(10), y[i], alpha=0.5);
(X, _), _ = tf.keras.datasets.mnist.load_data()
X.shape
a = X[0]
import matplotlib.patches as patches

plt.figure(figsize=(10, 10))

plt.imshow(a, cmap='Greys')

rect = patches.Rectangle((7-0.5, 7-0.5),3, 3,linewidth=2,edgecolor='r',facecolor='none')

plt.gca().add_patch(rect);
a[7:10, 7:10]
k = np.array([

    [-1, 0, 1],

    [-2, 0, 2],

    [-1, 0, 1]

])

k = np.array([

    [-1, -2, -1],

    [0, 0, 0],

    [1, 2, 1]

])

k
a[7:10, 7:10] * k
(a[7:10, 7:10] * k).sum()
b = np.zeros((28-2, 28-2))

for i in range(0, 28-2):

    for j in range(0, 28-2):

        patch = a[i: i+3, j:j+3]

        b[i, j] = (patch * k).sum()
plt.imshow(a, cmap='Greys');
plt.imshow(b, cmap='Greys')

plt.colorbar();
model = tf.keras.Sequential([

    Input(shape=(64, 64, 3)),

    Conv2D(32, kernel_size=(3, 3), activation="relu"),

    MaxPooling2D(),

    Conv2D(32, kernel_size=(3, 3), activation="relu"),

    MaxPooling2D(),

    Conv2D(32, kernel_size=(3, 3), activation="relu"),

    MaxPooling2D(),

    Conv2D(32, kernel_size=(3, 3), activation="relu"),

    MaxPooling2D(),

    Flatten(),

    Dense(10, activation="softmax"),

])
model.summary()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=3, validation_data=valid_generator)
model = tf.keras.Sequential([

    Input(shape=(64, 64, 3)),

    Conv2D(128, kernel_size=(3, 3), activation="relu"),

    Conv2D(128, kernel_size=(3, 3), activation="relu"),

    MaxPooling2D(),

    Conv2D(128, kernel_size=(3, 3), activation="relu"),

    Conv2D(128, kernel_size=(3, 3), activation="relu"),

    MaxPooling2D(),

    Conv2D(128, kernel_size=(3, 3), activation="relu"),

    Conv2D(128, kernel_size=(3, 3), activation="relu"),

    MaxPooling2D(),



    Flatten(),

    Dense(64, activation='relu'),

    Dense(10, activation="softmax"),

])
model.summary()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
h = model.fit(train_generator, epochs=10, validation_data=valid_generator)
plt.plot(h.history['accuracy'][1:])

plt.plot(h.history['val_accuracy'][1:])
model.optimizer.lr = 1e-4
h = model.fit(train_generator, epochs=10, validation_data=valid_generator)
plt.plot(h.history['accuracy'][1:])

plt.plot(h.history['val_accuracy'][1:])
l2 = 2e-5

model = tf.keras.Sequential([

    Input(shape=(64, 64, 3)),

    Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2)),

    Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2)),

    MaxPooling2D(),

    Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2)),

    Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2)),

    MaxPooling2D(),

    Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2)),

    Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2)),

    MaxPooling2D(),



    Flatten(),

    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),

    Dense(10, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(1e-5)),

])
model.summary()
model.compile(tf.keras.optimizers.Adam(1e-3), 'categorical_crossentropy', metrics=['accuracy'])
h = model.fit(train_generator, epochs=10, validation_data=valid_generator)
plt.plot(h.history['accuracy'][1:])

plt.plot(h.history['val_accuracy'][1:])
model.optimizer.lr = 1e-4
h = model.fit(train_generator, epochs=10, validation_data=valid_generator)
plt.plot(h.history['accuracy'][1:])

plt.plot(h.history['val_accuracy'][1:])
model = tf.keras.Sequential([

    Input(shape=(64, 64, 3)),

    Conv2D(128, kernel_size=(3, 3), activation="relu"),

    Conv2D(128, kernel_size=(3, 3), activation="relu"),

    MaxPooling2D(),

    Dropout(0.25),

    Conv2D(128, kernel_size=(3, 3), activation="relu"),

    Conv2D(128, kernel_size=(3, 3), activation="relu"),

    MaxPooling2D(),

    Dropout(0.25),

    Conv2D(128, kernel_size=(3, 3), activation="relu"),

    Conv2D(128, kernel_size=(3, 3), activation="relu"),

    MaxPooling2D(),

    Flatten(),

    Dropout(0.25),

    Dense(64, activation='relu'),

    Dense(10, activation="softmax"),

])
model.summary()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
h = model.fit(train_generator, epochs=10, validation_data=valid_generator)
plt.plot(h.history['accuracy'][1:])

plt.plot(h.history['val_accuracy'][1:])
model.optimizer.lr=1e-4
h = model.fit(train_generator, epochs=10, validation_data=valid_generator)
plt.plot(h.history['accuracy'][1:])

plt.plot(h.history['val_accuracy'][1:])
probs = model.predict(test_generator)
probs.shape
probs[0]
preds = np.argmax(probs, 1)

preds.shape
preds[:3]
pred_classes = [classes[i] for i in preds]
pred_classes[:3]
len(test_images)
def create_submission(model, test_generator, classes, test_images):

    probs = model.predict(test_generator)

    preds = np.argmax(probs, 1)

    pred_classes = [classes[i] for i in preds]

    sub =  pd.DataFrame({'fn': test_images, 'class': pred_classes})

    return sub
sub = create_submission(model, test_generator, classes, test_images)

sub.to_csv('submission1.csv', index=False)
sub.head()