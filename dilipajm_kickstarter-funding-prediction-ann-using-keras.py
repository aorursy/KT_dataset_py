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
dataframe = pd.read_csv('../input/train.csv')

dataframe.head()
X = dataframe.iloc[:, [3, 5, 6, 7, 8, 9, 10, 11, 12]].values

y = dataframe.iloc[:, -1]

index = 0

print('X: ',X[index, :])

print('y: ',y[index])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_disable_com = LabelEncoder()

X[:, 1] = label_encoder_disable_com.fit_transform(X[:, 1])



label_encoder_country = LabelEncoder()

X[:, 2] = label_encoder_country.fit_transform(X[:, 2])



label_encoder_currency = LabelEncoder()

X[:, 3] = label_encoder_currency.fit_transform(X[:, 3])



index = 1

print('Before hot encoding: ', X[index,:])

print('Shape before hot encoding: ', X.shape)



# one hot encoding in last for multiple columns

one_hot_encoding_country = OneHotEncoder(categorical_features = [1, 2, 3])

X = one_hot_encoding_country.fit_transform(X).toarray()

print('After hot encoding: ', X[index,:])

print('Shape after hot encoding: ', X.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, X_test.shape)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

print('After feature scaling: ', X_train[10])
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout



classifier = Sequential()



input_dim = int(X_train.shape[1])

output_dim = int(X_train.shape[1] / 2)

print('Input dim: ', input_dim)

print('Output dim: ', output_dim)



#input layer

classifier.add(Dense(output_dim=output_dim, init='uniform', activation='relu', input_dim=input_dim))

classifier.add(Dropout(p=0.1))



# hidden layer

classifier.add(Dense(output_dim=output_dim, init='uniform', activation='relu'))

classifier.add(Dropout(p=0.1))



# output layer

classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))



# compile ann

classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])



# fit ANN

classifier.fit(X_train, y_train, batch_size=50, nb_epoch=50)
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

y_pred
from sklearn.metrics import accuracy_score

metrics = accuracy_score(y_test, y_pred)

metrics