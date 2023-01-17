# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import keras
from keras.models import load_model
from keras import Model

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
inception_cnn = load_model("../input/mnist-inception-cnn-keras/model_acc.best.hdf5")
inception_cnn_datagen = load_model("../input/mnist-inception-cnn-data-gen/model_acc_inception_datagen.best.hdf5")
cnn = load_model("../input/mnist-cnn-kernel/model_cnn_acc.best.hdf5")
cnn_datagen = load_model("../input/mnist-cnn-using-imagedatagenerator/model_gen_cnn_acc.best.hdf5")
inception_cnn_datagen_even = load_model("../input/mnist-inception-cnn-data-gen-even/model_acc_inception_datagen.best.hdf5")
test_data = pd.read_csv("../input/digit-recognizer/test.csv")
test_data = test_data / 255
X_test = np.array(test_data.loc[:,:])
X_test = X_test.reshape(X_test.shape[0],28,28,1)
X_test.shape
predictions_inception_cnn = inception_cnn.predict(X_test)
predictions_inception_cnn_datagen = inception_cnn_datagen.predict(X_test)
predictions_cnn = cnn.predict(X_test)
predictions_cnn_datagen = cnn_datagen.predict(X_test)
predictions_cnn_datagen_even = inception_cnn_datagen_even.predict(X_test)
predictions = predictions_inception_cnn + predictions_inception_cnn_datagen + predictions_cnn + predictions_cnn_datagen + predictions_cnn_datagen_even
predictions = predictions / 5
predictions = np.argmax(predictions,axis = 1)
predictions = pd.Series(predictions,name="Label")

submission = pd.concat([pd.Series(range(1,test_data.shape[0]+1),name = "ImageId"),predictions],axis = 1)
submission.to_csv("cnn_mnist_load_model.csv",index=False)

