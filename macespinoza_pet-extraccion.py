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
from keras.applications.densenet import preprocess_input, DenseNet121

from keras.models import Model

from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D

import keras.backend as K

from tqdm import tqdm, tqdm_notebook
img_size = 256

batch_size = 16
inp = Input((256,256,3))

backbone = DenseNet121(input_tensor = inp, include_top = False,weights='../input/densenet121-h5/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')

x = backbone.output

x = GlobalAveragePooling2D()(x)

x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)

x = AveragePooling1D(4)(x)

out = Lambda(lambda x: x[:,:,0])(x)



m = Model(inp,out)
def load_image_num(path, pet_id, num):

    image = cv2.imread(f'{path}{pet_id}-{num}.jpg')

    new_image = resize_to_square(image)

    new_image = preprocess_input(new_image)

    return new_image



def features_densenet(pet_ids, rutaimg, numberImg, m):

    img_size = 256

    batch_size = 16

    n_batches = len(pet_ids) // batch_size + 1

    features = {}

    for b in tqdm_notebook(range(n_batches)):

        start = b*batch_size

        end = (b+1)*batch_size

        batch_pets = pet_ids[start:end]

        batch_images = np.zeros((len(batch_pets),img_size,img_size,3))

        for i,pet_id in enumerate(batch_pets):

            try:

                batch_images[i] = load_image_num(rutaimg, pet_id,numberImg)

            except:

                pass

        batch_preds = m.predict(batch_images)

        for i,pet_id in enumerate(batch_pets):

            features[pet_id] = batch_preds[i]

    return features
train_df = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')

pet_ids = train_df['PetID'].values

n_batches = len(pet_ids) // batch_size + 1



features = features_densenet(pet_ids, "../input/petfinder-adoption-prediction/train_images/", 1,m)
test_df = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')



pet_ids = test_df['PetID'].values

n_batches = len(pet_ids) // batch_size + 1

features = features_densenet(pet_ids, "../input/petfinder-adoption-prediction/test_images/", 1,m)