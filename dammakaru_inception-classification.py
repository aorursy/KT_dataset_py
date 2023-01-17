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
WIDTH = 299

HEIGHT = 299

BATCH_SIZE = 32
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



from keras.preprocessing import image

from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.models import load_model





def predict(model, img):

    """Run model prediction on image

    Args:

        model: keras model

        img: PIL format image

    Returns:

        list of predicted labels and their probabilities 

    """

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)

    return preds[0]





def plot_preds(img, preds):

    """Displays image and the top-n predicted probabilities in a bar graph

    Args:

        preds: list of predicted labels and their probabilities

    """

    labels = ("code", "notcode")

    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    plt.figure(figsize=(8,8))

    plt.subplot(gs[0])

    plt.imshow(np.asarray(img))

    plt.subplot(gs[1])

    plt.barh([0, 1], preds, alpha=0.5)

    plt.yticks([0, 1], labels)

    plt.xlabel('Probability')

    plt.xlim(0, 1)

    plt.tight_layout()
MODEL_FILE = '/kaggle/input/test-inception/image_classifier_inception.model'
model = load_model(MODEL_FILE)
for l in model.layers:

    l.trainable = False
for l in model.layers:

    print(l.name, l.trainable)
img = image.load_img('/kaggle/input/code-images/datasetv2/dataset/code/00000076.png', target_size=(HEIGHT, WIDTH))

preds = predict(model, img)



plot_preds(np.asarray(img), preds)

preds
def predict_frame_label(frame_path):

    img = image.load_img(frame_path, target_size=(HEIGHT, WIDTH))

    preds = predict(model, img)

    prediction = 'code' if np.argmax(preds) == 0 else 'notcode' 

    return prediction
for dirname, _, filenames in os.walk('/kaggle/input/code-images/datasetv2/dataset/code/'):

    for filename in filenames:

        pathname = os.path.join(dirname, filename)

        print("{} : {}".format(filename, predict_frame_label(pathname)))
MODEL_FILE = 'Inceptionv3_code_frame_classifier.model'

model.save(MODEL_FILE)