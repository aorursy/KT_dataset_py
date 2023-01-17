import numpy as np 

import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense, Dropout

%matplotlib inline
print("Importing Train Data")

train_data = pd.read_csv('../input/train.csv')

num_labels = len(np.unique(train_data['label'].values))

train_x = train_data[train_data.columns[1:]].values

train_y = train_data['label'].values

tr_ex, tr_pixels = train_x.shape



sqrt_dim = int(np.sqrt(tr_pixels))

channel = 1 # Grayscale



print("Importing Test Data")

test_data = pd.read_csv('../input/test.csv')

test_x = test_data.values

te_ex, te_pixels = test_x.shape

assert tr_pixels == te_pixels
train_data.head(5)
train_data.describe()
sns.distplot(train_y, bins=num_labels, kde=False)

plt.show()
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
test_data.head(5)
test_data.describe()
plt.gcf().set_size_inches(20, 20)

plot_images(test_x, np.random.randint(0, test_x.shape[0], 15))
validation_split = 0.2

train_datagen = ImageDataGenerator(

    validation_split=0.2, 

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
# Callback on validation accuracy

class AccuracyCallback(tf.keras.callbacks.Callback):

    """Callback that stops training on requested accuracy."""

    def __init__(self, target_accuracy=0.997):

        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs):

        if logs.get('val_acc') >= self.target_accuracy:

            print("Target validation accuracy (%f) reached !" % self.target_accuracy )

            self.model.stop_training = True
print("\nSetting up the model")

model = tf.keras.models.Sequential([    

    Conv2D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(sqrt_dim, sqrt_dim, 1)),

    BatchNormalization(),

    MaxPool2D(2),

    

    Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'),

    BatchNormalization(),

    MaxPool2D(2),

  

    Conv2D(filters=90, kernel_size=5, padding='same', activation='relu'),

    BatchNormalization(),

    MaxPool2D(2),

    

    Flatten(),

    Dense(units=1024, activation='relu'),

    Dropout(0.35),

    Dense(units=num_labels, activation='softmax'),

])

model.summary()



model.compile(

    optimizer=tf.keras.optimizers.Adagrad(),

    loss=tf.keras.losses.sparse_categorical_crossentropy,

    metrics=['accuracy']

) 
target_accuracy = 0.999

max_epochs = 40

print("\nTraining untill accuracy reaches %s or for %i epochs" %(target_accuracy, max_epochs))

valid_input_length = int(len(train_x)*validation_split)

train_input_length = int(len(train_x)-valid_input_length)

model.fit_generator(

    generator=train_iterator,

    steps_per_epoch= train_input_length // train_batch_size,

    validation_data=validation_iterator,

    validation_steps= valid_input_length // train_batch_size,

    epochs=max_epochs, 

    callbacks=[AccuracyCallback(target_accuracy)]

)
print("Predicting Test labels")

predictions = model.predict(test_data_normalized)
pred_df = test_data.copy()

pred_df['Label'] = np.argmax(predictions, axis=1)

pred_df['ImageId'] = pred_df.index +1

fig = plt.gcf()

fig.set_size_inches(20, 20)

pred_x = pred_df[[p for p in pred_df.columns if 'pixel' in p]].values

pred_y = pred_df['Label'].values

plot_images(pred_x, np.random.randint(0, pred_x.shape[0], 15), pred_y)
print("Saving Test labels")

#pred_df.to_csv('3by3Same70filters_paxpool2_twice_Dense150.csv', index=True)

res = pred_df[['ImageId', 'Label']]

res.to_csv('test3.csv', header=True, index=False)