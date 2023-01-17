import numpy as np 

import pandas as pd





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
from matplotlib import pyplot as plt
y_train = train_df.iloc[:,0]

X_train = train_df.iloc[:,1:]



X_test = test_df.copy()
n=10

plt.imshow(X_train.iloc[n].values.reshape(28,28))

plt.show()

print(y_train[n])

X_train = X_train / 255.0

X_test = X_test/ 255.0



X_train_im = X_train.values.reshape(-1,28,28,1)

X_test_im = X_test.values.reshape(-1,28,28,1)
X_test_im.shape,X_train_im.shape
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
model = Sequential()

model.add(Conv2D(64 , 7 ,activation='relu', padding = "same" ,input_shape=[28,28,1]))

model.add(MaxPooling2D(2))

model.add(Conv2D(128 , 3,activation='relu', padding = "same" ))

model.add(Conv2D(128 , 3,activation='relu', padding = "same" ))

model.add(MaxPooling2D(2))

model.add(Dropout(0.25))

model.add(Conv2D(256 , 3,activation='relu', padding = "same" ))

model.add(Conv2D(256 , 3,activation='relu', padding = "same" ))

model.add(MaxPooling2D(2))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128,activation='elu',kernel_initializer="he_normal"))

model.add(Dropout(0.5))

model.add(Dense(64,activation='elu',kernel_initializer="he_normal",kernel_regularizer=keras.regularizers.l2(0.01)))

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))

model.summary()
# s = 20 * len(X_train) // 32

# learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)

def exponential_decay(lr0, s):

    def exponential_decay_fn(epoch):

        return lr0 * 0.1**(epoch / s)

    return exponential_decay_fn



exponential_decay_fn = exponential_decay(lr0=0.01, s=20)

lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)

optimizer=keras.optimizers.SGD(lr=0.01,momentum=0.9, nesterov=True)

model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5",save_best_only=True)

history = model.fit(X_train_im, y_train, epochs=100,batch_size=32,validation_split=0.1,callbacks=[EarlyStopping(patience=8),lr_scheduler,checkpoint_cb])
pd.DataFrame(history.history).plot(figsize=(16, 10))

plt.grid(True)

plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]

plt.show()

model = keras.models.load_model("my_keras_model.h5")

y_test = model.predict(X_test_im)
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
results = np.argmax(y_test,axis = 1)

results = pd.Series(results,name="Label")



submission['Label'] =results



submission.head()
submission.to_csv("cnn_mnist_datagen.csv",index=False)
