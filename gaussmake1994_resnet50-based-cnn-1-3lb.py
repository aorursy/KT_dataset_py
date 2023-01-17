%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("../input/labels.csv")

df.head()
df.describe(include="all")
breed_codes = list(set(df["breed"]))

breed_codes.sort()



for code in breed_codes:

    df[code] = 1.0 * (df["breed"] == code)
from sklearn.preprocessing import LabelEncoder



plt.hist(LabelEncoder().fit_transform(df["breed"]), bins=50);
from scipy.misc import imread, imresize





def load_train_image(id):

    return imresize(imread("../input/train/{0}.jpg".format(id)), 

                    (224, 224))





def load_test_image(id):

    return imresize(imread("../input/test/{0}.jpg".format(id)),

                    (224,224))
from ipywidgets import IntProgress

from IPython.display import display





def log_progress(sequence, every=10):

    progress = IntProgress(min=0, max=len(sequence), value=0)

    display(progress)

    for index, record in enumerate(sequence):

        if index % every == 0:

            progress.value = index

        yield record

    progress.value = len(sequence)

    

    

def chunks(lst, size):

    """Yield successive n-sized chunks from l."""

    result = []

    for i in range(0, len(lst), size):

        result.append(lst[i:i + size])

    return result
plt.imshow(load_train_image("000bec180eb18c7604dcecc8fe0dba07"));
plt.imshow(load_test_image("00a3edd22dc7859c487a64777fc8d093"));
from keras.applications import ResNet50





resnet = ResNet50(include_top=False, weights='imagenet')





def get_resnet_features(ids, loader):

    id_chunks = chunks(ids, 10)

    resnet_output = {}

    for chunk in log_progress(id_chunks, every=1):

        images = []

        for image_id in chunk:

            image = loader(image_id)

            images.append(image)

        predictions = resnet.predict(np.array(images))

        for i, image_id in enumerate(chunk):

            resnet_output[image_id] = predictions[i]

    return resnet_output
train_resnet_features = get_resnet_features(df["id"], load_train_image)
from keras.models import Sequential

from keras.layers import Flatten, Dense, Dropout

from keras import regularizers



model = Sequential()

model.add(Flatten(input_shape=train_resnet_features[df["id"][0]].shape))

model.add(Dropout(0.5))

model.add(Dense(2 * len(breed_codes),

                activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(len(breed_codes),

                activation='softmax'))

model.compile("sgd", "categorical_crossentropy")
class_counts = np.array([(df["breed"] == breed).sum() 

                         for breed in breed_codes])

class_weights = class_counts.mean() / class_counts

class_weights_dict = {cls: class_weights[i] for i, cls in enumerate(breed_codes)}

sample_weights = np.array([class_weights_dict[breed]

                           for breed in df["breed"]])
from keras.callbacks import EarlyStopping





X_train = np.array([train_resnet_features[image_id]

                    for image_id in df["id"]])

y_train = np.array(df[breed_codes])



model.fit(X_train, y_train, 

          epochs=1000,

          sample_weight=sample_weights,

          verbose=True,

          validation_split=0.3,

          callbacks=[

              EarlyStopping(min_delta=1e-4, patience=10)

          ])
from os import listdir

from os.path import isfile, join





def get_test_image_ids():

    test_dir = "../input/test"

    test_files = filter(isfile, map(lambda fname: join(test_dir, fname), listdir(test_dir)))

    ids = map(lambda fname: fname.split('/')[-1].split('\\')[-1].split('.')[0], test_files)

    return list(ids)
test_image_ids = get_test_image_ids()

test_resnet_features = get_resnet_features(test_image_ids, load_test_image)
X_test = np.array([test_resnet_features[image_id]

                   for image_id in test_image_ids])

test_prediction = model.predict(X_test)
from collections import OrderedDict



test_df_dict = OrderedDict([("id", test_image_ids)])

for breed_index, breed in enumerate(breed_codes):

    test_df_dict[breed] = test_prediction[:, breed_index]

pd.DataFrame(test_df_dict).to_csv("../output/resnet50-dense-dense.csv", index=False)
from keras.layers import InputLayer





result_model = Sequential()

result_model.add(InputLayer(input_shape=(224, 224, 3)))

result_model.add(resnet)

result_model.add(model)

result_model.compile("sgd", "categorical_crossentropy")

result_model.save("../output/result_model.h5")
result_model.predict(np.array([load_train_image("000bec180eb18c7604dcecc8fe0dba07")]))