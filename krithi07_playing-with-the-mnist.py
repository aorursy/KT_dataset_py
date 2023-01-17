# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Training Data Shape:", train.shape)
print("Test Data Shape:", test.shape)
y = to_categorical(train["label"])
train = train.drop(["label"], axis=1)
train.shape
train = train.divide(255)
test = test.divide(255)
np.random.seed(123)

def baseline():
# create model
    model = Sequential()
    model.add(Dense(784, input_dim=784, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.38))
    model.add(Dense(392, kernel_initializer='normal', activation = 'relu'))
    model.add(Dropout(0.33))
    model.add(Dense(196, kernel_initializer='normal', activation = 'relu'))
    model.add(Dropout(0.28))
    model.add(Dense(98, kernel_initializer='normal', activation = 'relu'))
    model.add(Dropout(0.22))
    model.add(Dense(49, kernel_initializer='normal', activation = 'relu'))
    model.add(Dropout(0.11))
    model.add(Dense(10, kernel_initializer='normal', activation = 'softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #try rmsprop
    
    return model
estimator = KerasClassifier(build_fn=baseline, epochs=20, batch_size=128, verbose=1)

X_train, X_test, Y_train, Y_test = train_test_split(train, y, test_size=0.3, random_state=123)
estimator.fit(X_train, Y_train)
predictions = estimator.predict(X_test)

#kfold = KFold(n_splits=10, shuffle=True, random_state=123)
#results = cross_val_score(estimator, train, y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
y_classes = [np.argmax(y, axis=None, out=None) for y in Y_test]
from sklearn.metrics import accuracy_score
accuracy_score(y_classes, predictions)
pred_test = estimator.predict(test)
print(pred_test)

submissions=pd.DataFrame({"ImageId": list(range(1,len(pred_test)+1)),
                         "Label": pred_test})
submissions.to_csv("MLP_baseline.csv", index=False, header=True)