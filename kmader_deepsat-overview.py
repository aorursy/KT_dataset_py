import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # showing and rendering figures

# io related

from skimage.io import imread

import os

from glob import glob

# not needed in Kaggle, but required in Jupyter

%matplotlib inline 
def read_Xy(in_path, nrows = 5000):

    in_df = pd.read_csv(os.path.join('..', 'input', in_path), 

                       nrows = nrows, header=None)

    X = in_df.values.reshape((-1, 28, 28, 4)).clip(0, 255).astype(np.uint8)

    y = np.argmax(pd.read_csv(os.path.join('..', 'input', in_path.replace('X_', 'y_')), 

                       nrows = nrows, header=None).values, 1)

    return X, y
tX, tY = read_Xy('X_train_sat6.csv', 16)

print(tX.shape, tY.shape)

fig, m_axs = plt.subplots(4, tX.shape[0]//4, figsize = (12, 12))

for (x, y, c_ax) in zip(tX, tY, m_axs.flatten()):

    c_ax.imshow(x[:,:,:3], # since we don't want NIR in the display

                interpolation = 'none')

    c_ax.axis('off')

    c_ax.set_title('Cat:{}'.format(y))
tX, tY = read_Xy('X_train_sat6.csv', 40000)
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline



class PipeStep(object):

    """

    Wrapper for turning functions into pipeline transforms (no-fitting)

    """

    def __init__(self, step_func):

        self._step_func=step_func

    def fit(self,*args):

        return self

    

    def transform(self,X):

        return self._step_func(X)



norm_step = PipeStep(lambda in_image: (in_image - in_image.mean())/in_image.std())

flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])

    

rfc_image_pipeline = Pipeline([

    ('Normalize Image', norm_step),

    ('Flatten Image', flatten_step),

    ('RF', RandomForestClassifier())])



rfc_image_pipeline.fit(tX.reshape((-1, np.prod(tX.shape[1:]))), tY)
test_X, test_Y = read_Xy('X_test_sat6.csv', None)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pred_Y = rfc_image_pipeline.predict(test_X)

print(classification_report(test_Y, pred_Y))

print('Overall Accuracy: %2.2f%%' % (100*accuracy_score(test_Y, pred_Y)))

plt.matshow(confusion_matrix(test_Y, pred_Y), cmap = plt.cm.nipy_spectral)