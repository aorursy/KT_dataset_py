import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
train_df = pd.read_csv('/kaggle/input/land-cover-class/train.csv')
train_dir = '/kaggle/input/land-cover-class/train/'
test_images = sorted([fn.split('/')[-1] for fn in glob('/kaggle/input/land-cover-class/test/*.jpg')])
classes = sorted(train_df['class'].unique())
classes
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
size=224
train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    directory=train_dir,
    x_col='fn',
    y_col='class',
    target_size=(size, size), 
    subset='training'
)

valid_generator = train_datagen.flow_from_dataframe(
    train_df, 
    directory=train_dir,
    x_col='fn',
    y_col='class',
    target_size=(size, size), 
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    directory='/kaggle/input/land-cover-class/',
    classes=['test'],
    batch_size=32,
    target_size=(size, size), 
    shuffle=False
)
from tensorflow.keras.applications import EfficientNetB0
X, y = train_generator[0]
X.shape, y.shape
inp = Input(shape=(224, 224, 3))
model = EfficientNetB0(include_top=False, input_tensor=inp, weights='imagenet', classes=len(classes))
model.trainable = False

# Rebuild top
x = GlobalAveragePooling2D()(model.output)
x = BatchNormalization()(x)

top_dropout_rate = 0.2
x = Dropout(top_dropout_rate)(x)
out = Dense(len(classes), activation="softmax")(x)

model = tf.keras.Model(inp, out)
model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"]
    )
model.summary()
#tf.keras.utils.plot_model(model, show_shapes=True, to_file='model.png')
model.fit(train_generator, epochs=1, validation_data=valid_generator)
model.fit(train_generator, epochs=5, validation_data=valid_generator)
def _create_generator(**kwargs):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **kwargs
    )
    size=224
    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        directory=train_dir,
        x_col='fn',
        y_col='class',
        target_size=(size, size), 
        subset='training',
        shuffle=False
    )

    return train_generator
np.random.seed(123)
def plot_augmentation(batch, sample, **kwargs):
    train_generator = _create_generator(**kwargs)

    imgs = []
    for i in range(8):
        imgs.append(train_generator[batch][0][sample].astype(int))
        
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))
    for img, ax in zip(imgs, axs.flat):
        ax.imshow(img)
    plt.tight_layout()
plot_augmentation(0, 3)
plot_augmentation(0, 3, horizontal_flip=True, vertical_flip=True)
plot_augmentation(0, 3, rotation_range=45)
plot_augmentation(0, 3, brightness_range=(0.5, 1.5))
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.2,
    brightness_range=(0.8, 1.2),
#     rotation_range=10,
    horizontal_flip=True, vertical_flip=True,
)
size=224
train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    directory=train_dir,
    x_col='fn',
    y_col='class',
    target_size=(size, size), 
    subset='training'
)

valid_generator = train_datagen.flow_from_dataframe(
    train_df, 
    directory=train_dir,
    x_col='fn',
    y_col='class',
    target_size=(size, size), 
    subset='validation'
)
from tensorflow.keras.layers.experimental import preprocessing
X, y = train_generator[0]
X.shape
inp = Input(shape=(224, 224, 3))
x = inp
# x = preprocessing.RandomRotation(factor=0.05)(x)
model = EfficientNetB0(include_top=False, input_tensor=x, weights='imagenet', classes=len(classes))
model.trainable = False

# Rebuild top
x = GlobalAveragePooling2D()(model.output)
x = BatchNormalization()(x)

top_dropout_rate = 0.2
x = Dropout(top_dropout_rate)(x)
out = Dense(len(classes), activation="softmax")(x)

model = tf.keras.Model(inp, out)
model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"]
    )
model.fit(train_generator, epochs=5, validation_data=valid_generator, workers=4, use_multiprocessing=True)
def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
unfreeze_model(model)
model.fit(train_generator, epochs=5, validation_data=valid_generator, workers=4, use_multiprocessing=True)
def create_submission(model, test_generator, classes, test_images):
    probs = model.predict(test_generator)
    preds = np.argmax(probs, 1)
    pred_classes = [classes[i] for i in preds]
    sub =  pd.DataFrame({'fn': test_images, 'class': pred_classes})
    return sub
sub = create_submission(model, test_generator, classes, test_images)
sub.to_csv('submission1.csv', index=False)


