import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = '../input/emnist/'

edf = pd.read_csv(os.path.join(BASE_DIR, 'emnist-balanced-train.csv'))
mdf = pd.read_csv(os.path.join(BASE_DIR, 'emnist-mnist-train.csv'))
mdf.columns = range(len(mdf.columns))
edf.columns = range(len(edf.columns))


edf_categories = edf.groupby([edf.columns[0]])[edf.columns[1]].count()
print('EDF categories: ', len(edf_categories))

mdf.iloc[:, 0] += len(edf_categories)

mdf_categories = mdf.groupby([mdf.columns[0]])[mdf.columns[1]].count()
print('MDF categories: ', len(mdf_categories))


data = pd.concat([mdf, edf])
merged_categories = data.groupby([data.columns[0]])[data.columns[1]].count()
print('Merged categories: ', len(merged_categories))

data = data.sample(frac=1).reset_index(drop=True)

print('Total images: ', len(data)) 
def normalize_image(image):
    image = image.reshape((28, 28))
    image = np.rot90(image[::-1], 3)
    return image


clean_data = []
clean_labels = []

for image in data.values:
    img = normalize_image(image[1:])
    img = np.asarray(img).astype('float32') / 255
    
    label = np.asarray(image[0]).astype('float32')
    
    clean_data.append(img)
    clean_labels.append(label)

full_data = np.asarray(clean_data).reshape(len(clean_data), 28, 28, 1)
full_labels = np.asarray(clean_labels)
from tensorflow.keras.utils import to_categorical


train_data = full_data[:100000]
train_labels = to_categorical(full_labels[:100000])

validation_data = full_data[100000:150000]
validation_labels = to_categorical(full_labels[100000:150000])

test_data = full_data[150000:]
test_labels = to_categorical(full_labels[150000:])
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(test_data[i].reshape((28, 28)))
    print(np.argmax(test_labels[i]), end=' ')
plt.show()
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers


class Model(tf.keras.models.Model):
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv_1 = layers.SeparableConv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1))
        self.batch_norm_1 = layers.BatchNormalization(axis=1)
        self.pooling_1 = layers.MaxPooling2D((2, 2))
        
        
        self.conv_2 = layers.SeparableConv2D(64, (3, 3), activation=tf.nn.relu)
        self.pooling_2 = layers.MaxPooling2D((2, 2))

        self.flatten = layers.Flatten()
        self.dropout_1 = layers.Dropout(0.2)
        self.fc = layers.Dense(1024, activation=tf.nn.relu)
        self.fc1 = layers.Dense(512, activation=tf.nn.relu)
        self.fc2 = layers.Dense(256, activation=tf.nn.relu)
        self.fc3 = layers.Dense(len(merged_categories), activation=tf.nn.softmax)
        
        
    def call(self, x):
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.pooling_1(x)
        
        x = self.conv_2(x)
        x = self.pooling_2(x)
        
        x = self.flatten(x)
        x = self.dropout_1(x)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return self.fc3(x)

    
model = Model()


callbacks_list = [
    tf.keras.callbacks.EarlyStopping(
        monitor = 'val_acc',
        patience = 1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath = 'emnist_model.h5',
        monitor = 'val_loss',
        save_best_only=True,
    )
]

model.compile(
    loss = losses.categorical_crossentropy,
    optimizer = optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'],
)
history = model.fit(
    train_data, train_labels,
    validation_data = [validation_data, validation_labels],
    batch_size = 64,
    epochs = 10,
    callbacks = callbacks_list,
)
model.load_weights('emnist_model.h5')
_, accuracy = model.evaluate(test_data, test_labels, verbose=0)
print('Test accuracy: ', round(accuracy * 100, 2), '%')

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'b-', label='Train accuracy')
plt.plot(history.history['val_accuracy'], 'r-', label='Validation accuracy')

plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'b-', label='Train loss')
plt.plot(history.history['val_loss'], 'r-', label='Validation loss')

plt.legend()
plt.grid()
plt.show()
prediction = model.predict(test_data)
index = 1
print('Prediction: ', np.argmax(prediction[index]), '\t', 'Label: ', np.argmax(test_labels[index]))
plt.imshow(test_data[index].reshape((28, 28)))
