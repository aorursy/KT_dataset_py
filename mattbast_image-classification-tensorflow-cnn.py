import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

print(data.info())
train_data = data.head(37800)

val_data = data.tail(4200)



train_labels = train_data.pop('label')

val_labels = val_data.pop('label')
tf_train_data = tf.data.Dataset.from_tensor_slices((train_data.values, train_labels.values))

tf_val_data = tf.data.Dataset.from_tensor_slices((val_data.values, val_labels.values))



print(tf_train_data)

print(tf_val_data)
plt.figure(figsize=(10,10))

i = 0



for image, label in tf_train_data.take(5):

    plt.subplot(1,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)



    plt.imshow(image.numpy().reshape((28, 28)), cmap='gray')

    plt.xlabel(label.numpy())



    i = i + 1
def preprocess_image(image, label):

    image = tf.reshape(image, [28, 28, 1])

    image = tf.cast(image, tf.float32) / 255.

    

    return image, label



tf_train_data = tf_train_data.map(

    preprocess_image, 

    num_parallel_calls=tf.data.experimental.AUTOTUNE

)



tf_val_data = tf_val_data.map(

    preprocess_image, 

    num_parallel_calls=tf.data.experimental.AUTOTUNE

)



print(tf_train_data)

print(tf_val_data)
def pipeline(tf_data):

    tf_data = tf_data.shuffle(100)

    tf_data = tf_data.batch(32)

    tf_data = tf_data.prefetch(tf.data.experimental.AUTOTUNE)

    

    return tf_data



tf_train_data = pipeline(tf_train_data)

tf_val_data = pipeline(tf_val_data)



print(tf_train_data)

print(tf_val_data)
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(6, (5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D((2, 2)),

    

    tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='valid'),

    tf.keras.layers.MaxPooling2D((2, 2)),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(120, activation='relu'),

    tf.keras.layers.Dense(84, activation='relu'),

    

    tf.keras.layers.Dense(10, activation='softmax'),

])
optimiser = tf.keras.optimizers.Adam(learning_rate=0.001)



model.compile(

    optimizer=optimiser, 

    loss='sparse_categorical_crossentropy', 

    metrics=['accuracy'])



model.summary()
callbacks = [

    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),

    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1),

]
train_log = model.fit(

    tf_train_data,

    validation_data=tf_val_data,

    epochs=30,

    callbacks=callbacks

)
plt.plot(train_log.history['accuracy'], label='accuracy')

plt.plot(train_log.history['val_accuracy'], label = 'val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()



print('Training accuracy: %f' % train_log.history['accuracy'][-1])

print('Validation accuracy: %f' % train_log.history['val_accuracy'][-1])
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

tf_test_data = tf.data.Dataset.from_tensor_slices(([test_data.to_numpy().reshape(len(test_data), 28, 28, 1)]))
predictions = model.predict(tf_test_data)

predictions = np.argmax(predictions, axis=1)
plt.figure(figsize=(10,10))



for i, row in test_data.head(15).iterrows():

    plt.subplot(3,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)



    plt.imshow(row.values.reshape((28, 28)), cmap='gray')

    plt.xlabel(predictions[i])
predictions_df = pd.DataFrame(data={'Label': predictions}, index=pd.RangeIndex(start=1, stop=28001))

predictions_df.index = predictions_df.index.rename('ImageId')



predictions_df.to_csv('submission_file.csv')