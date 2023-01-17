# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from keras.utils import np_utils

import tensorflow as tf



tf.logging.set_verbosity(tf.logging.INFO)



import warnings

warnings.filterwarnings('ignore')
train_file = "../input/train.csv"

test_file = "../input/test.csv"



# Loading data

X_train = pd.read_csv(train_file)

X_test  = pd.read_csv(test_file)



# Separating data and labels

y_train=X_train['label']

X_train=X_train.drop('label',axis=1)
# Normalizing data

X_train /= 255.0

X_test /= 255.0



#I don't believe tensorflow supports float64

X_train = X_train.astype("float32")

X_test  = X_test.astype("float32")

y_train = y_train.astype('int32')



#number of features

num_features = len(X_train.columns)



#arrays

X_train=np.array(X_train)

X_test = np.array(X_test)
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(X_train,y_train,every_n_steps=100,

                                                                 early_stopping_metric="accuracy",

                                                                 early_stopping_metric_minimize=False,

                                                                 early_stopping_rounds=200)



# Build 3 layer DNN with 10, 20, 10 units respectively.

classifier = tf.contrib.learn.DNNClassifier(feature_columns=[tf.contrib.layers.real_valued_column("", dimension=num_features)],

                                            hidden_units=[500,800,500],

                                            n_classes=10,

                                            dropout=0.5)



# Fit model

classifier.fit(x=X_train, y=y_train, steps=100000,monitors=[validation_monitor])



accuracy_score = classifier.evaluate(x=X_train, y=y_train)["accuracy"]

print('Accuracy: {0:f}'.format(accuracy_score))
predictions=list(classifier.predict(X_test, as_iterable=True))



submission=pd.read_csv('../input/sample_submission.csv')

submission['Label']=predictions

submission.to_csv('submission_.csv',index=False)