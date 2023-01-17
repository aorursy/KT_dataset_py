import warnings

import pandas as pd



# Ignore warnings

warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None
import os



# Show input data files

for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Train set



df_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv", index_col=False)

df_train.shape
df_train.head()
# Test set



df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv", index_col=False)

df_test.shape
df_test.head()
# Separate features & labels



df_y = df_train["label"]

df_X = df_train.drop(labels=["label"], axis=1)
import seaborn as sns



# Visualize distribution of given data



def plot_count(df, title, clr="darkgrid"):

    sns.set(style=clr)

    ax = sns.countplot(df)

    ax.set_title(title)

    

plot_count(df_y, "Train Labels")
import numpy as np



# Reshape pixels into 3D matrices



def reshape_3d(df, h, w, channel):

    if not isinstance(df, np.ndarray):

        df = df.values

    return df.reshape(-1, h, w, channel)



df_X = reshape_3d(df_X, 28, 28, 1)
from keras.utils.np_utils import to_categorical



# Encode labels to one-hot vectors



df_y = to_categorical(df_y)

df_y.shape
from sklearn.model_selection import train_test_split



# Create train and validation sets



X_train, X_val, y_train, y_val = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

X_train.shape
# Calculate mean (𝜇) and standard deviation (𝜎) of train set



X_train_mean = X_train.mean().astype(np.float32)

X_train_std = X_train.std().astype(np.float32)



# Rescale pixels to have 𝜇 of 0 and 𝜎 of 1



def standardize(df, mean=X_train_mean, std=X_train_std):

    return (df - mean) / std
from keras.models import Sequential



# Use Keras Sequential to create stack of layers



model = Sequential()
from keras.layers import Lambda 



# Add Lambda layer to perform standardization



model.add(Lambda(standardize, input_shape=(28, 28, 1)))
from keras.layers import Conv2D, MaxPool2D



# Add Conv2D and MaxPooling2D layers to create convolution kernel and downsample



model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))

model.add(MaxPool2D(pool_size=(2, 2)))
from keras.layers.core import Flatten



# Add Flatten layer to transform input into 1D array



model.add(Flatten())
from keras.layers.core import Dense



# Add Dense layers to connect neurons in previous layers



model.add(Dense(256, activation="relu"))

model.add(Dense(10, activation="softmax"))
from keras.optimizers import RMSprop



# Add loss, optimizer, metrics functions



model.compile(optimizer=RMSprop(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
from keras.preprocessing.image import ImageDataGenerator



# Generate batches of image data to avoid overfitting



genr = ImageDataGenerator()

genr.fit(X_train)



batches_train = genr.flow(X_train, y_train, batch_size=64)

batches_val = genr.flow(X_val, y_val, batch_size=64)
history = model.fit_generator(epochs=10, verbose=1,

                              generator=batches_train, steps_per_epoch=batches_train.n / batches_train.batch_size, 

                              validation_data=batches_val, validation_steps=batches_val.n / batches_val.batch_size)
%matplotlib inline



import matplotlib.pyplot as plt



i = range(1, len(history.history["loss"]) + 1, 1)
plt.plot(i, history.history["loss"], "go--", markersize=10, label="Train")

plt.plot(i, history.history["val_loss"], "bx--", markersize=10, label="Validation")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.xticks(i)

plt.legend()

plt.show()
plt.plot(i, history.history["accuracy"], "go--", markersize=10, label="Train")

plt.plot(i, history.history["val_accuracy"], "bx--", markersize=10, label="Validation")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.xticks(i)

plt.legend()

plt.show()
model.summary()
# Preprocess test set



df_test = reshape_3d(df_test, 28, 28, 1)
# Predict labels



preds = model.predict_classes(df_test, verbose=1)
plot_count(preds, "Predictions")
import csv



# Export predictions to csv file

with open("result.csv", "w") as f:

    writer = csv.DictWriter(f, fieldnames=["ImageId", "Label"])

    writer.writeheader()

    for i in range(len(preds)):

        writer.writerow({"ImageId": i + 1, "Label": preds[i]})
# Render pixels as image and show label along with prediction



def show_sample(i, X=X_train, y=y_train, model=model, h=28, w=28, channel=1, clr="gray"):

    pred = model.predict_classes(reshape_3d(X[i], h, w, channel))

    lbl = np.where(y[i] == 1)[0]

    plt.imshow(X[i][:, :, 0], cmap=plt.get_cmap(clr))

    if lbl == pred:

        status = "OK"

    else:

        status = "NOK"

    plt.title("{}\nLabel: {}, Prediction: {}".format(status, lbl[0], pred[0]))
show_sample(11)
show_sample(100)