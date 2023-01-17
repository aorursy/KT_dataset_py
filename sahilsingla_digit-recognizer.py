!pip install git+https://github.com/qubvel/efficientnet
import numpy as np
import pandas as pd
import tensorflow as tf
import efficientnet.tfkeras as efn
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
%matplotlib inline
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train_df.head()
X = train_df.iloc[:, 1:].values
Y = train_df['label'].values
X[0]
X = X.reshape(-1, 28, 28)
X = np.stack((X,)*3, axis=-1)
Y = tf.keras.utils.to_categorical(Y, num_classes = 10)
final_X = np.zeros((X.shape[0], 32, 32, 3))
final_X[:, 2:30, 2:30, :] = X
X = final_X/255.
final_X = None
X[0][20]
plt.imshow(X[0])
X = efn.preprocess_input(X)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.25, random_state=31196)
X_train.shape
gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=5,
                                                      width_shift_range=5, 
                                                      height_shift_range=5,
                                                      shear_range=5)

gen.fit(X_train)
no_of_classes = 10
DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}
model = efn.EfficientNetB0(include_top=False, input_shape=(32, 32, 3))

x = model.layers[-1].output
x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = tf.keras.layers.Dropout(0.2, name='top_dropout')(x)
x = tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=DENSE_KERNEL_INITIALIZER, name='probs')(x)

model = tf.keras.models.Model(inputs=model.input, outputs=x)

for layer in model.layers:
    layer.trainable = True
model.summary()
model.compile(tf.keras.optimizers.Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, mode='min', verbose=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint('/kaggle/working/digit_recognizer_{epoch:02d}.hdf5', 
                                             monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', 
                                              baseline=None, restore_best_weights=True)
callbacks = [reduce, checkpoint, earlystopping]
history = model.fit_generator(gen.flow(X_train, Y_train, batch_size=16), epochs = 1000, validation_data = (X_val, Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0]//16, callbacks=callbacks)
history
