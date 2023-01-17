import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization

import tensorflow as tf
from tensorflow import keras
import os

%load_ext tensorboard
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train.shape
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test.shape
def preprocessing(data, img_row, img_cols):
    
    num_images = data.shape[0]
    x_as_arr = data.values
    x_as_arr_reshape = x_as_arr.reshape(num_images, img_row, img_cols)
    out_x = x_as_arr_reshape / 255.
    return out_x
X_train = train.iloc[:,1:]
X_train_prep = preprocessing(X_train, 28, 28)
y_train = train.label
n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train_prep[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(y_train[index], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
run_index = 1 # increment this at every run
run_logdir = os.path.join(os.curdir, "digit_rec_logs", "run_{:03d}".format(run_index))
run_logdir
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=2e-1),
              metrics=["accuracy"])
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
checkpoint_cb = keras.callbacks.ModelCheckpoint("digit_rec_model.h5", save_best_only=True)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

history = model.fit(X_train_prep, y_train, epochs=100,
                    validation_split=.2,
                    callbacks=[early_stopping_cb, checkpoint_cb, tensorboard_cb])
X_test_prep = preprocessing(test, 28, 28)
model = keras.models.load_model("digit_rec_model.h5") # rollback to best model
y_pred = np.argmax(model.predict(X_test_prep), axis=-1)
%tensorboard --logdir=./digit_rec_logs --port=6006
ssub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
ssub.head()
#submisiion
sub_file = pd.DataFrame({'ImageId':ssub['ImageId'], 'Label':y_pred})

sub_file.to_csv('sub_file.csv', index=False)
