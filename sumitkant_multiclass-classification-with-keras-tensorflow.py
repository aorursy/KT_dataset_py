import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import to_categorical

# from keras.utils.np_utils import to_categorical

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline
seed = 42

np.random.seed(seed)
df = pd.read_csv('../input/iris/Iris.csv')

df.head()
X = df.drop(['Id','Species'], axis = 1).to_numpy()

Y = df.Species.to_numpy()

print ('Unique values in the output : ', set(Y))

print ('Shape X :', X.shape)

print ('Shape Y :', Y.shape)
encoder = LabelEncoder()

Y_encoded = encoder.fit_transform(Y)

Y_dummy = to_categorical(Y_encoded)

print ('Shape of Y after One hot encoding :', Y_dummy.shape)
def baseline_model():

    model = Sequential([

        Dense(4, input_dim = (4), activation = 'relu'),

        Dense(12, activation = 'relu'),

        Dense(3, activation = 'sigmoid')

    ])

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model
single_model = baseline_model()

history = single_model.fit(X, Y_dummy, validation_split = 0.1, epochs = 200, verbose = 0)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Training & Validation loss')

plt.xlabel('epochs')

plt.show()



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Training & Validation Accuracy')

plt.xlabel('epochs')

plt.show()
%%time 

estimator = KerasClassifier(build_fn = baseline_model, epochs = 200, batch_size = 5, verbose = 0)

kFold = KFold(n_splits = 3, shuffle = True, random_state = seed)

results = cross_val_score(estimator, X, Y_dummy, cv = kFold)

print (f'Mean Accuracy : {round(results.mean()*100, 2)} % ; Std Dev : {round(results.std()*100, 2)}')