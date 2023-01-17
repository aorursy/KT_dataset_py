PLACEHOLDER_DATASET_PATH = "/kaggle/input/mnist.npz"

PLACEHOLDER_MODEL_SAVE_PATH = "knn.joblib"

PLACEHOLDER_K = 5

PLACEHOLDER_NTRAIN = 10000

PLACEHOLDER_NTEST = 100
## Init

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from joblib import dump



DATASET_PATH = PLACEHOLDER_DATASET_PATH

with np.load(DATASET_PATH, allow_pickle=True) as f:

    x_train, y_train = f['x_train'], f['y_train']

    x_test, y_test = f['x_test'], f['y_test']



x_train = x_train.reshape((x_train.shape[0], -1))

x_test = x_test.reshape((x_test.shape[0], -1))
## Model Config

K = PLACEHOLDER_K

NTRAIN = PLACEHOLDER_NTRAIN

NTEST = PLACEHOLDER_NTEST



x_train_sample = x_train[:NTRAIN]

y_train_sample = y_train[:NTRAIN]

x_test_sample = x_test[:NTEST]

y_test_sample = y_test[:NTEST]
## Run

model = KNeighborsClassifier(n_neighbors=K)

_ = model.fit(x_train_sample, y_train_sample)

print((model.predict(x_test_sample) == y_test_sample).sum() / x_test_sample.shape[0])

_ = dump(model, PLACEHOLDER_MODEL_SAVE_PATH)
PLACEHOLDER_DATASET_PATH = "/kaggle/input/mnist.npz"

PLACEHOLDER_MODEL_PATH = "knn.joblib"
## Init

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from joblib import load

from PIL import Image

import urllib.request

import io

import json



DATASET_PATH = PLACEHOLDER_DATASET_PATH

with np.load(DATASET_PATH, allow_pickle=True) as f:

    x_train, y_train = f['x_train'], f['y_train']

    x_test, y_test = f['x_test'], f['y_test']



model = load(PLACEHOLDER_MODEL_PATH)
PLACEHOLDER_IMAGE_URL = "https://conx.readthedocs.io/en/latest/_images/MNIST_44_0.png"
## Run

with urllib.request.urlopen(PLACEHOLDER_IMAGE_URL) as url:

    f = io.BytesIO(url.read())

image = Image.open(f).convert("L")

image = image.resize((28, 28))

image = np.array(image).reshape(1, -1)



# probabilities

print(json.dumps(model.predict_proba(image)[0].tolist()))



size_pixel_ratio = 1/12



# plot neighbors

for i in model.kneighbors(image)[1][0]:

    pixel = 28

    height = pixel * size_pixel_ratio

    plt.figure(figsize=(height, height))

    plt.imshow(x_train[i], cmap="gray")

    plt.axis("off")

    plt.show()

    print(y_train[i])
PLACEHOLDER_DATASET_PATH = "/kaggle/input/mnist.npz"

PLACEHOLDER_MODEL_SAVE_PATH = "activation.h5"
## Init

import numpy as np

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dense, Activation, InputLayer



DATASET_PATH = PLACEHOLDER_DATASET_PATH

with np.load(DATASET_PATH, allow_pickle=True) as f:

    x_train, y_train = f['x_train'], f['y_train']

    x_test, y_test = f['x_test'], f['y_test']



x_train = np.expand_dims(x_train, -1)

x_test = np.expand_dims(x_test, -1)

x_train = x_train / 255

x_test = x_test / 255

input_shape = x_train.shape[1:]

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
## Model Config

model = Sequential()

model.add(InputLayer(input_shape))

model.add(Conv2D(32, 3, padding="same"))

model.add(Activation("relu"))

model.add(Conv2D(32, 3, padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D(2))

model.add(Activation("relu"))

model.add(Flatten())

model.add(Dense(200))

model.add(Activation("relu"))

model.add(Dense(10))

model.add(Activation("softmax"))



model.compile(

    loss = "categorical_crossentropy",

    optimizer = "rmsprop",

    metrics = ["accuracy"]

)
model.summary()
## Run

history = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=1)

print(history.history["val_acc"][-1])



from keras.models import Model

render_layers = [model.output]

prev_layer = ""

for layer in model.layers:

    if "activation" in layer.name and ("conv2d" in prev_layer or "max_pooling" in prev_layer):

        render_layers.append(layer.output)

    prev_layer = layer.name

render_model = Model(inputs=model.input, outputs=render_layers)

render_model.save(PLACEHOLDER_MODEL_SAVE_PATH)
PLACEHOLDER_MODEL_PATH = "activation.h5"
## Init

%matplotlib inline

from keras.models import load_model

import matplotlib.pyplot as plt

import json

import numpy as np

from PIL import Image

import urllib.request

import io



model = load_model(PLACEHOLDER_MODEL_PATH)
PLACEHOLDER_IMAGE_URL = "https://conx.readthedocs.io/en/latest/_images/MNIST_44_0.png"
## Run



with urllib.request.urlopen(PLACEHOLDER_IMAGE_URL) as url:

    f = io.BytesIO(url.read())

image = Image.open(f).convert("L")

image = image.resize((28, 28))

image = np.array(image).reshape(1, 28, 28, 1)

image = image / 255



res = model.predict(image)



# probabilities

print(json.dumps(res[0][0].tolist()))



size_pixel_ratio = 1/12



# plot original image

pixel = image.shape[-2]

height = pixel * size_pixel_ratio

plt.figure(figsize=(height, height))

plt.imshow(image[0][:, :, 0], cmap="gray")

plt.axis("off")

plt.show()



# plot each activation layer

for layer in res[1:]:

    channel = layer.shape[-1]

    pixel = layer.shape[-2]

    height = pixel * size_pixel_ratio

    f, axarr = plt.subplots(1, channel, figsize=(channel*height, height))

    for i in range(channel):

        axarr[i].imshow(layer[0][:, :, i], cmap="gray")

        axarr[i].axis("off")

    plt.show()