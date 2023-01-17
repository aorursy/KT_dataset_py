import pandas as pd

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import Adam

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score
data = pd.read_csv('../input/data.csv')
data.head()
data = data[data.columns[1:-1]]

data.head()
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})
def createNN():

    model = Sequential()

    model.add(Dense(units=8, activation='relu', kernel_initializer='random_uniform', input_dim = 30))

    model.add(Dropout(0.1))

    model.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))

    model.add(Dropout(0.2))

    model.add(Dense(units=1, activation='sigmoid'))

    

    optimizer = Adam(lr=1e-3, decay=1e-5)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    return model
classifier = KerasClassifier(build_fn=createNN)
y = data['diagnosis']

X = data.drop(columns=['diagnosis'])



results = cross_val_score(classifier, X, y,

                          fit_params = {'epochs':100, 'batch_size':32},

                           scoring = 'accuracy',

                           cv = 10, verbose=1)
accuracy_mean = results.mean()

accuracy_std = results.std()
print('accuracy: %.4f' % accuracy_mean)

print('std: %.4f' % accuracy_std)