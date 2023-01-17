import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
%matplotlib inline

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import plot_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import SVG, Image
import tensorflow as tf
print("Tensorflow version:", tf.__version__)
def plot_example_images(plt):
    img_size = 255
    plt.figure(0, figsize=(12,20))
    ctr = 0
    for expression in [f.name for f in os.scandir("../input/covid19-radiography-database/COVID-19 Radiography Database") if f.is_dir() ]:
        for i in range(1,6):
            ctr += 1
            plt.subplot(7,5,ctr)
            img = load_img("../input/covid19-radiography-database/COVID-19 Radiography Database/" + expression + "/" +os.listdir("../input/covid19-radiography-database/COVID-19 Radiography Database/" + expression)[i], target_size=(img_size, img_size))
            plt.title(expression)
            plt.imshow(img)

    plt.tight_layout()
    return plt.show()
plot_example_images(plt)
batch_size = 32
img_height = 80
img_width = 80
data_dir = "../input/covid19-radiography-database/COVID-19 Radiography Database"
tf.random.set_seed(73)
train_datagen = ImageDataGenerator(rescale=1.0/255.0, horizontal_flip=True, validation_split = 0.2)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split = 0.2)


train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

test_generator = validation_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                           input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (5, 5), padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    
    tf.keras.layers.Conv2D(512, (3, 3), padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(512, (3, 3), padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(1024),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              loss="categorical_crossentropy",
              metrics=['categorical_accuracy'])
epochs = 25
steps_per_epoch = train_generator.n // train_generator.batch_size
validation_steps = test_generator.n // test_generator.batch_size

checkpoint = tf.keras.callbacks.ModelCheckpoint("os./model_weights.h5", monitor='val_accuracy',
                            save_weights_only = True,
                            mode = 'max',
                            verbose = 1)
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=3, verbose=1)
checkpointer = tf.keras.callbacks.ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True)

callbacks = [checkpoint, lr_reducer, checkpointer]

history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=test_generator,
                   callbacks=callbacks)
epochs = epochs
metric = "loss"
metric2 = "categorical_accuracy"
train_m1 = history.history[metric]
train_m2 = history.history[metric2]
val_m1 = history.history[f'val_{metric}']
val_m2 = history.history[f'val_{metric2}']

fig_model = make_subplots(1,
                          2,
                          subplot_titles=(f"{metric.capitalize()} over Time",
                                          f"{metric2.capitalize()} over Time"))
fig_model.add_trace(go.Scatter(x=np.arange(epochs),
                               y=train_m1,
                               mode="lines+markers",
                               name=f"Training {metric.capitalize()}"),
                    row=1,
                    col=1)
fig_model.add_trace(go.Scatter(x=np.arange(epochs),
                               y=val_m1,
                               mode="lines+markers",
                               name=f"Validation {metric.capitalize()}"),
                    row=1,
                    col=1)

fig_model.add_trace(go.Scatter(x=np.arange(epochs),
                               y=train_m2,
                               mode="lines+markers",
                               name=f"Training {metric2.capitalize()}"),
                    row=1,
                    col=2)
fig_model.add_trace(go.Scatter(x=np.arange(epochs),
                               y=val_m2,
                               mode="lines+markers",
                               name=f"Validation {metric2.capitalize()}"),
                    row=1,
                    col=2)
fig_model.show()
print("Training Accuracy: "+str(np.round(history.history["categorical_accuracy"][-1]*100,2))+"%")
print("Validation Accuracy: "+str(np.round(history.history["val_categorical_accuracy"][-1]*100,2))+"%")
model_json = model.to_json()
with open("model.json", "w") as f:
    f.write(model_json)