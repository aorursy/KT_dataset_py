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
import sys

sys.path.append("/kaggle/input/test-modules3/")
#https://github.com/qubvel/classification_models/tree/master/classification_models/models

import keras

from models_factory import ModelsFactory





class KerasModelsFactory(ModelsFactory):



    @staticmethod

    def get_kwargs():

        return {

            'backend': keras.backend,

            'layers': keras.layers,

            'models': keras.models,

            'utils': keras.utils,

        }





Classifiers = KerasModelsFactory()
ResNet18, preprocess_input = Classifiers.get('resnet18')

model = ResNet18((224, 224, 3))
test_batch = np.random.rand(32,224,224,3)
predict = model.predict(test_batch)
predict