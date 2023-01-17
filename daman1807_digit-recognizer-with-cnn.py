!pip install --upgrade git+https://github.com/aleju/imgaug.git
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
# read data
df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
predictions = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

# create mask for train-test split
msk = np.random.rand(len(df)) < 0.7

# train-test split [ 30% ]
train = df[msk]
test = test = df[~msk]

print("Train Samples: ", len(train))
print("Test Samples: ", len(test))
model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.15))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.15))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.summary()
seq = iaa.Sequential([
    iaa.SomeOf((0, 1), [
        iaa.PiecewiseAffine(scale=(0.01, 0.04)),
        iaa.PerspectiveTransform(scale=(0.01, 0.04))
    ], random_order=True),
    iaa.SomeOf((0, 1), [
        iaa.geometric.Affine(rotate=(-7, 7)),
        iaa.geometric.Affine(shear=(-5, 5)),
    ], random_order=True)
])
def data_generator(mode, batch_size):
    while True:
        if mode == "train" :
            batch = train.sample(n=batch_size)
        else :
            batch = test.sample(n=batch_size)
        batch_x, batch_y = [], []
        for (index, row) in batch.iterrows():
            img = np.array(row[1:])
            img = img.reshape((28, 28, 1)).astype(np.uint8)
            batch_x.append(img)
            batch_y.append(row[0])
        batch_x = seq(images=batch_x)
        batch_x = np.array(batch_x).astype(np.float32)
        batch_y = np.array(batch_y)
        # batch_y = to_categorical(batch_y, num_classes=10)
        batch_x /= 255.0
        yield batch_x, batch_y
model.compile(
    optimizer=Adam(lr=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

batch_size = 64

history = model.fit_generator(
    data_generator("train", batch_size),
    epochs=25,
    steps_per_epoch= len(train) // batch_size,
    validation_data=data_generator("test", batch_size),
    validation_steps=int(0.5 * len(train)) // batch_size,
    use_multiprocessing=True
)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# predict for submission
submission = []
for (index, row) in predictions.iterrows():
    img = np.array(row)
    x = img.reshape((1, 28, 28, 1)).astype(np.float32)
    x /= 255.0
    y = np.argmax(model.predict(x)[0])
    submission.append([index+1, y])

df = pd.DataFrame(submission, columns=["ImageId", "label"])
df.to_csv("/kaggle/working/submission.csv", index=False)
df.head(20)