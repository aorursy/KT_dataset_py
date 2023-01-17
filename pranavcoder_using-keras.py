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
train = pd.read_csv("../input/kag_risk_factors_cervical_cancer.csv")

for c in train.columns:

    if train[c].dtype == 'object':

        train[c] =pd.to_numeric(train[c], errors='coerce')
train.head()
from sklearn.preprocessing import Imputer
imputer = Imputer (missing_values='NaN',strategy = 'mean' ,axis=0)
X= train.iloc[:,0:28].values

Y = train.iloc[:,29].values
imputer.fit(X[:,0:28])

X[:,0:28]=imputer.transform(X[:,0:28])
from mlxtend.preprocessing import one_hot

Y = one_hot(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

import keras

from keras.models import Sequential

from keras.layers import Dense

def build():

    seq = Sequential()

    seq.add(Dense(units = 200, activation = 'relu', kernel_initializer = 'uniform', input_dim = 28))

    seq.add(Dense(units = 200, activation = 'relu', kernel_initializer = 'uniform'))

    seq.add(Dense(units = 200, activation = 'relu', kernel_initializer = 'uniform'))

    seq.add(Dense(units = 2, activation = 'softmax', kernel_initializer = 'uniform'))

    seq.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return seq

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score, KFold



#kfold = KFold(n_splits=10, shuffle=True)



classifier = KerasClassifier(build_fn = build, batch_size = 32, nb_epoch= 20)

csv = cross_val_score(estimator= classifier, X = X_train, y = Y_train, cv = 10, n_jobs=-1)



mean = csv.mean()

std = csv.std()



classifier.fit(X_train, Y_train)
print("Accuracy: {}%\n".format(classifier.score(X_test, Y_test) *100))
classifier.predict(X_test)