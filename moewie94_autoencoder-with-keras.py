# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import cv2

import tqdm

import tarfile

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.layers import Dense, Flatten, Reshape, Input, InputLayer

from keras.models import Sequential, Model
# http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt

ATTRS_NAME = "../input/lfw_attributes.txt"



# http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz

IMAGES_NAME = "../input/lfw-deepfunneled.tgz"



# http://vis-www.cs.umass.edu/lfw/lfw.tgz

RAW_IMAGES_NAME = "../input/lfw.tgz"
def decode_image_from_raw_bytes(raw_bytes):

    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
def load_lfw_dataset(

        use_raw=False,

        dx=80, dy=80,

        dimx=45, dimy=45):



    # Read attrs

    df_attrs = pd.read_csv(ATTRS_NAME, sep='\t', skiprows=1)

    df_attrs = pd.DataFrame(df_attrs.iloc[:, :-1].values, columns=df_attrs.columns[1:])

    imgs_with_attrs = set(map(tuple, df_attrs[["person", "imagenum"]].values))



    # Read photos

    all_photos = []

    photo_ids = []



    # tqdm in used to show progress bar while reading the data in a notebook here, you can change

    # tqdm_notebook to use it outside a notebook

    with tarfile.open(RAW_IMAGES_NAME if use_raw else IMAGES_NAME) as f:

        for m in tqdm.tqdm_notebook(f.getmembers()):

            # Only process image files from the compressed data

            if m.isfile() and m.name.endswith(".jpg"):

                # Prepare image

                img = decode_image_from_raw_bytes(f.extractfile(m).read())



                # Crop only faces and resize it

                img = img[dy:-dy, dx:-dx]

                img = cv2.resize(img, (dimx, dimy))



                # Parse person and append it to the collected data

                fname = os.path.split(m.name)[-1]

                fname_splitted = fname[:-4].replace('_', ' ').split()

                person_id = ' '.join(fname_splitted[:-1])

                photo_number = int(fname_splitted[-1])

                if (person_id, photo_number) in imgs_with_attrs:

                    all_photos.append(img)

                    photo_ids.append({'person': person_id, 'imagenum': photo_number})



    photo_ids = pd.DataFrame(photo_ids)

    all_photos = np.stack(all_photos).astype('uint8')



    # Preserve photo_ids order!

    all_attrs = photo_ids.merge(df_attrs, on=('person', 'imagenum')).drop(["person", "imagenum"], axis=1)



    return all_photos, all_attrs
X, attr = load_lfw_dataset(use_raw=True, dimx=32, dimy=32)

# os.listdir('../input')
X = X.astype('float32') / 255.0 - 0.5
def show_image(x):

    plt.figure(figsize=(1,1))

    plt.imshow(np.clip(x + 0.5, 0, 1))
show_image(X[0])
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
def build_autoencoder(img_shape, code_size):

    # The encoder

    encoder = Sequential()

    encoder.add(InputLayer(img_shape))

    encoder.add(Flatten())

    encoder.add(Dense(code_size))



    # The decoder

    decoder = Sequential()

    decoder.add(InputLayer((code_size,)))

    decoder.add(Dense(np.prod(img_shape))) # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072

    decoder.add(Reshape(img_shape))



    return encoder, decoder
# Same as (32,32,3), we neglect the number of instances from shape

IMG_SHAPE = X.shape[1:]

encoder, decoder = build_autoencoder(IMG_SHAPE, 256)



inp = Input(IMG_SHAPE)

code = encoder(inp)

reconstruction = decoder(code)



autoencoder = Model(inp,reconstruction)

autoencoder.compile(optimizer='adam', loss='mse')



autoencoder.summary()
history = autoencoder.fit(x=X_train, y=X_train, epochs=5,

                validation_data=[X_test, X_test])
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
def visualize(img,encoder,decoder):

    """Draws original, encoded and decoded images"""

    # img[None] will have shape of (1, 32, 32, 3) which is the same as the model input

    code = encoder.predict(img)

    reco = decoder.predict(code)



    plt.subplot(1,3,1)

    plt.title("Original")

    plt.imshow(img.squeeze())



    plt.subplot(1,3,2)

    plt.title("Code")

    plt.imshow(code.reshape([code.shape[-1]//16,-1]))



    plt.subplot(1,3,3)

    plt.title("Reconstructed")

    plt.imshow(reco.squeeze())

    plt.show()
for _ in range(5):

    i = np.random.choice(len(X_test))

    img = X_test[i]

    img = np.expand_dims(img, 0)

    visualize(img,encoder,decoder)