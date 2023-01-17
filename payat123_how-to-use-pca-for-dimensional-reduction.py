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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
%matplotlib inline
df = pd.read_csv('../input/aps_failure_training_set_processed_8bit.csv')
features = df.drop(['class'], axis=1)
target = df['class']

# split dataset into train and dev set 90:10 %
f_train, f_dev, t_train, t_dev = train_test_split(features, target, test_size=0.1, random_state=42)

np.unique(target)
# threshold classes into 0 and 1(failure)
lb = Binarizer()
lb.fit(t_train.values.reshape(-1, 1))
t_train_bi = lb.transform(t_train.values.reshape(-1, 1))
t_dev_bi = lb.transform(t_dev.values.reshape(-1, 1))
np.unique(t_train_bi)
# normalize using standard method
scaler = StandardScaler()
scaler.fit(f_train)
f_nor_train = scaler.transform(f_train)
f_nor_dev = scaler.transform(f_dev)

# select principle components by PCA
pca = PCA(n_components=60)
pca.fit(f_nor_train)
f_pca_train = pca.transform(f_nor_train)
f_pca_dev = pca.transform(f_nor_dev)
# varince of principle components
plt.bar(range(60),pca.explained_variance_ratio_)


# make a training and dev set
X_train = f_pca_train
X_dev = f_pca_dev
y_train = t_train_bi.reshape(-1,1)
y_dev = t_dev_bi.reshape(-1,1)
LR = LinearRegression()
LR.fit(X_train, y_train)
prediction = LR.predict(X_dev)
pred = (prediction > 0.5).astype(int)
print(classification_report(y_dev, pred))
print(confusion_matrix(y_dev, pred))
