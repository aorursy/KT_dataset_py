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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from pdpbox import pdp, get_dataset, info_plots

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense

from keras import optimizers

from keras.layers import Dropout

from keras.constraints import maxnorm

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

from IPython.core.interactiveshell import InteractiveShell
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

df.head()
df.shape
df.isnull().any()
df = df.drop_duplicates()

df.shape
without_class_df = df.drop(['class'], axis=1)

with_class_df = df['class']

df = pd.get_dummies(without_class_df, columns=without_class_df.columns)

df = df.join(with_class_df, how='left')

df.head()
sns.countplot(df['class'])
X = df.drop(['class'], axis=1)

Y = df['class']

Y = Y.replace({'p':0, 'e': 1})

Y
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.2, random_state=0)
model = Sequential()

model.add(Dense(12, input_dim=117, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.add(Dropout(0.3))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_features, train_labels)

model_pred_train = model.predict_classes(train_features)

model_pred_test = model.predict_classes(test_features)
print(classification_report(test_labels,model_pred_test))

print('Neural Network baseline: ' + str(roc_auc_score(train_labels, model_pred_train)))

print('Neural Network: ' + str(roc_auc_score(test_labels, model_pred_test)))