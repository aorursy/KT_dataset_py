# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir, makedirs
from os.path import join, exists, expanduser
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score
from keras.preprocessing import image

from keras.applications import xception
from keras.applications import inception_v3
from keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.linear_model import LogisticRegression
inception_v3_weights = "../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
xception_weights = "../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"

labels = pd.read_csv("../input/dog-breed-identification/labels.csv")
sample_submssion = pd.read_csv("../input/dog-breed-identification/sample_submission.csv")
print(len(listdir("../input/dog-breed-identification/train")),len(labels))
print(len(listdir("../input/dog-breed-identification/test")),len(sample_submssion))
def read_img(img_id,train_or_test,size):
    img = image.load_img(join("../input/dog-breed-identification",train_or_test,"%s.jpg" % img_id),
                        target_size=size)
    img = image.img_to_array(img)
    return img
labels.head()
target_series = pd.Series(labels['breed'])
one_hot = pd.get_dummies(target_series,sparse=True)
one_hot_labels = np.asarray(one_hot)
#INPUT_SIZE = 299
#输入设置为299，内存开销太大了
y_train = []
INPUT_SIZE = 260
POOLING = 'avg'
y_train = np.zeros((len(labels),120),dtype=np.uint8)
x_train = np.zeros((len(labels),INPUT_SIZE,INPUT_SIZE,3),dtype='float32')
for i,img_id in tqdm(enumerate(labels['id'])):
    img = read_img(img_id,'train',(INPUT_SIZE,INPUT_SIZE))
    x = xception.preprocess_input(np.expand_dims(img.copy(),axis=0))
    x_train[i] = x
    label = one_hot_labels[i]
    y_train[i] = label
print("Train Images shape: {} size: {:,}".format(x_train.shape,x_train.size))
print("Target Images shape: {} size: {:,}".format(y_train.shape,y_train.size))
np.save("x_train.npy",x_train)
np.save("y_train.npy",y_train)
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x_train,y_train,test_size=0.3,random_state=7)

xception_bottleneck = xception.Xception(weights=xception_weights, include_top=False, pooling=POOLING)
train_x_bf = xception_bottleneck.predict(X_train, batch_size=32, verbose=1)
valid_x_bf = xception_bottleneck.predict(X_test, batch_size=32, verbose=1)
print('Xception train bottleneck features shape: {} size: {:,}'.format(train_x_bf.shape, train_x_bf.size))
print('Xception valid bottleneck features shape: {} size: {:,}'.format(valid_x_bf.shape, valid_x_bf.size))
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=7)
logreg.fit(train_x_bf, (Y_train* range(120)).sum(axis=1))
valid_probs = logreg.predict_proba(valid_x_bf)
valid_preds = logreg.predict(valid_x_bf)
print('Validation Xception LogLoss {}'.format(log_loss(Y_test, valid_probs)))
print('Validation Xception Accuracy {}'.format(accuracy_score((Y_test * range(120)).sum(axis=1), valid_preds)))
