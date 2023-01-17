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
data = pd.read_csv('../input/train.csv')

ID = data.pop('id')

or_data = data.copy()
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import train_test_split

from sklearn.model_selection import GridSearchCV
from keras.models import Sequential

from keras.layers import Dense,Dropout,Activation

from keras.utils.np_utils import to_categorical

from keras.wrappers.scikit_learn import KerasClassifier

from keras.layers import BatchNormalization
y = data.pop('species')

y = LabelEncoder().fit(y).transform(y)

y_cat = to_categorical(y)

y_cat.shape, y.shape
X = StandardScaler().fit(data).transform(data)

X.shape



test = pd.read_csv('../input/test.csv')

index = test.pop('id')

test = StandardScaler().fit(data).transform(test)
def create_model(dropout_rate_l1=0.3 , dropout_rate_l2=0.3):

    

    model = Sequential()

    model.add(Dense(600, input_dim=192,  init='uniform'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(dropout_rate_l1))

    

    model.add(Dense(300, init='uniform'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(dropout_rate_l2))

    

    model.add(Dense(99, activation='softmax'))

    

    model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])

    

    return model
from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test = train_test_split( X, y_cat, test_size = 0.1, random_state = 7)
model = create_model()

history_main = model.fit(X_train,y_train,batch_size=192, nb_epoch=400, verbose=2, validation_data=(X_test, y_test))
model.evaluate(X, y_cat)


yPred = model.predict_proba(test)

col = or_data.species.unique()

col.sort()

yPred = pd.DataFrame(yPred, index=index, columns=col)

yPred.to_csv('result_2.csv', index=True)