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
import tensorflow as tf

import keras



from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import KFold, cross_val_score



from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Layer
df_train = pd.read_csv('../input/train.csv')
y_train = df_train.target

X_train = df_train.ix[:, 'feat_1':'feat_93']
y_train = pd.get_dummies(y_train)
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
import tensorflow as tf

import keras
df_train = pd.read_csv('../input/train.csv')
y_train = df_train.target

X_train = df_train.ix[:, 'feat_1':'feat_93']