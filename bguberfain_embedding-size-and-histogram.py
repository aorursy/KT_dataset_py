import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from PIL import Image

import requests

from io import BytesIO



%matplotlib inline
REQUIRED_SIGNATURE = 'serving_default'

REQUIRED_OUTPUT = 'global_descriptor'
model = tf.saved_model.load(str('../input/baseline-landmark-retrieval-model/baseline_landmark_retrieval_model/'))
found_signatures = list(model.signatures.keys())



outputs = model.signatures[REQUIRED_SIGNATURE].structured_outputs



embedding_fn = model.signatures[REQUIRED_SIGNATURE]
response = requests.get('https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/The_Gate_%28177019927%29.jpeg/640px-thumb.jpg')

img = Image.open(BytesIO(response.content))

img
def get_embedding(img):

    image_data = np.array(img.convert('RGB'))

    image_tensor = tf.convert_to_tensor(image_data)

    return embedding_fn(image_tensor)[REQUIRED_OUTPUT].numpy()
emb = get_embedding(img)
emb.shape
plt.hist(emb.ravel(), bins=21);