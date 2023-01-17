import os

os.listdir('../input')
!pip install keras_efficientnets
from PIL import Image, ImageOps

import gc



import pandas as pd

import numpy as np



import keras

from keras.models import load_model



from keras_efficientnets import EfficientNetB5, preprocess_input
IMG_SIZE = (456, 456)

BATCH_SIZE = 32
from PIL import ImageOps



class TestGenerator(keras.utils.Sequence):

    

    def __init__(self, x, mirror=True):

        self.x = x

        self.mirror = mirror

        

    def __len__(self):

        return int(np.ceil(len(self.x)/BATCH_SIZE))

    

    def _transform(self, path):

        image = Image.open('../input/3rd-ml-month-car-image-cropping-dataset/test_crop/' + path).convert("RGB").resize(IMG_SIZE)

        if self.mirror:

            image = ImageOps.mirror(image)

        image = preprocess_input(np.array(image))

        return image

    

    def __getitem__(self, index):

        x = np.array(list(map(self._transform, self.x[index*BATCH_SIZE:(index+1)*BATCH_SIZE])))

        return  x
test = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/test.csv')['img_file'].values
def predict_one(x, file):

    test_preds = np.zeros((len(x), 196))

    model = load_model(file)

    for mirror in [False, True]:

        test_generator = TestGenerator(x, mirror=mirror)

        test_preds += model.predict_generator(

            test_generator,

            verbose=2,

        )

    return test_preds



def predict_oof(x, files, weights=None):

    if weights is None:

        weights = [1]*len(files)

    test_preds = np.zeros((len(x), 196))

    for file, weight in zip(files, weights):

        test_preds += predict_one(x, file)*weight

    return test_preds
files = [

    '../input/car-classification-models/eff0.h5',

    '../input/car-classification-models/eff1.h5',

    '../input/car-classification-models/eff2.h5',

    '../input/car-classification-models/eff3.h5',

    '../input/car-classification-models/eff4.h5',

    

    '../input/car-classification-models/eff5.h5', # 0.9484

    '../input/car-classification-models/eff6.h5', # 0.95~~

    '../input/car-classification-models/eff7.h5', # 0.9461

    '../input/car-classification-models/eff8.h5', # 0.9448

    '../input/car-classification-models/eff9.h5', # 0.9487

]

weights = None
test_preds = predict_oof(test, files, weights)

    

pred = test_preds.argmax(axis=1)+1



submission = pd.DataFrame({'img_file': test, 'class': pred})

submission.to_csv('submission.csv', index=False)

submission.head()