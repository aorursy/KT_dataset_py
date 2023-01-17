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
import pandas as pd



df = pd.read_csv('../input/mushrooms.csv', skiprows=1)

df.head()



import numpy as np

import tensorflow as tf

from sklearn import preprocessing



data = pd.DataFrame()



for col in df.columns:

    data[col] = preprocessing.LabelEncoder().fit_transform(df[col])



data.head()
from keras.utils import to_categorical



X = data.iloc[:, 1:]

y = to_categorical(np.array(data.iloc[:, 0]))



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)



X_train = X_train.astype('float32')

X_train /= 255.0



X_test = X_test.astype('float32')

X_test /= 255.0
from sklearn.ensemble import RandomForestClassifier



Rfclf = RandomForestClassifier(random_state=0, max_depth=5, n_jobs = 5)

Rfclf.fit(X_train, y_train)



print("Accuracy on training set: {:.3f}".format(Rfclf.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(Rfclf.score(X_test, y_test)))