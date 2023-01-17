from shutil import copyfile

# copy our file into the working directory (make sure it has .py suffix)

copyfile(src = "/kaggle/input/mylibs/datautils.py", dst = "../working/datautils.py")
from keras.models import load_model

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import confusion_matrix

import itertools

import datautils as du
model = load_model('/kaggle/input/lstm-cnn-5-level-sentiment/LSTM-5-level-sentiment-400k-v1.h5')

X_val = np.load("../input/valida/x-validation-50k.npy")

Y_val = np.load("../input/valida/y-validation-50k.npy")
score, acc = model.evaluate(X_val, Y_val, batch_size=15)

print('Test score:', score)

print('Test accuracy:', acc)
# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(Y_val,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

du.plot_confusion_matrix(confusion_mtx, classes = range(5)) 