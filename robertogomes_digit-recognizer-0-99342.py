import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

print("Importing Train Data")
train_data = pd.read_csv('../input/digit-recognizer/train.csv')
num_labels = len(np.unique(train_data['label'].values))
train_x = train_data[train_data.columns[1:]].values
train_y = train_data['label'].values
tr_ex, tr_pixels = train_x.shape

sqrt_dim = int(np.sqrt(tr_pixels))
channel = 1 # Grayscale

print("Importing Test Data")
test_data = pd.read_csv('../input/digit-recognizer/test.csv')
test_x = test_data.values
te_ex, te_pixels = test_x.shape
assert tr_pixels == te_pixels
def plot_images(pixels, indexes, labels=None, image_size=(28, 28)):
    """Given a dataset, its labels (if they exist)
    and the indexes to plot, plots the image(s)."""
    
    plots_num = len(indexes)
    current_plot = 1
    for idx in indexes:
        image = pixels[idx].reshape(image_size)
        ax = plt.subplot(1, plots_num, current_plot)
        ax.imshow(image, cmap=plt.get_cmap('gray'))
        label = -1 if labels is None else labels[idx]
        ax.set_title("label:%i" %label)
        current_plot +=1
    plt.show()
fig = plt.gcf()
fig.set_size_inches(20, 20)
plot_images(train_x, np.random.randint(0, train_x.shape[0], 15), train_y)
plt.gcf().set_size_inches(20, 20)
plot_images(test_x, np.random.randint(0, test_x.shape[0], 15))
validation_split = 0.3
train_datagen = ImageDataGenerator(
    validation_split=validation_split, 
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.02,
)

train_batch_size = 150
reshaped_train_x = train_x.reshape((tr_ex, sqrt_dim, sqrt_dim, channel))

train_iterator= train_datagen.flow(
    x=reshaped_train_x,
    y=train_y,
    batch_size=train_batch_size,
    subset='training'
)

validation_iterator= train_datagen.flow(
    x=reshaped_train_x,
    y=train_y,
    batch_size=train_batch_size,
    subset='validation'
)

# Quick look at the some resulting images
plt.gcf().set_size_inches(10, 10)
for x_batch, y_batch in train_iterator:
    for i in range(0, 25):
        plt.subplot(5, 5, i+1)
        plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    break
test_data_normalized = (test_x * 1./255).reshape((te_ex, sqrt_dim, sqrt_dim, channel))
model = keras.Sequential([
    keras.layers.Conv2D(32, 5, padding="same", activation=tf.nn.relu, input_shape=(28, 28, 1)),
    keras.layers.Conv2D(32, 5, padding="same", activation=tf.nn.relu),
    keras.layers.SpatialDropout2D(0.3),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(64, 5, padding="same", activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(128, 5, activation='relu', padding='same'),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(128, 5, activation='relu', padding='same'),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(800, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(400, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['sparse_categorical_accuracy'])
    
model.summary()
EPOCHS = 50
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    verbose=1,
    restore_best_weights=True,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5)

valid_input_length = int(len(train_x)*validation_split)
train_input_length = int(len(train_x)-valid_input_length)
model.fit_generator(
    generator=train_iterator,
    steps_per_epoch= train_input_length // train_batch_size,
    validation_data=validation_iterator,
    callbacks = [earlystop, reduce_lr],
    validation_steps= 1,
    epochs=EPOCHS
)
print("Predicting Test labels")
predictions = model.predict(test_data_normalized)


sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

probs = np.argmax(predictions, axis=1)
sub.Label = probs
sub.to_csv('submission_ld.csv', index=False)
sub.head()
pred_df = test_data.copy()
pred_df['Label'] = np.argmax(predictions, axis=1)

fig = plt.gcf()
fig.set_size_inches(20, 20)
pred_x = pred_df[[p for p in pred_df.columns if 'pixel' in p]].values
pred_y = pred_df['Label'].values
plot_images(pred_x, np.random.randint(0, pred_x.shape[0], 15), pred_y)