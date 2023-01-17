%matplotlib inline

import numpy as np

import pandas as pd
base_set = pd.read_csv('../input/digit-recognizer/train.csv')
label_column = 'label'



X = base_set.drop(columns=[label_column]).values

Y = base_set[label_column]
# Normalize

X = X / 255



def reshape_data(np_array):

    return np_array.reshape(len(np_array), 28, 28, 1)



# Make data correct structure

X = reshape_data(X)
from keras.utils import to_categorical



Y = to_categorical(Y, 10)
from sklearn.model_selection import train_test_split



train_to_valtest_ratio = .32

validate_to_test_ratio = .5



# First split our main set

(X_train,

 X_validation_and_test,

 Y_train,

 Y_validation_and_test) = train_test_split(X, Y, test_size=train_to_valtest_ratio)



# Then split our second set into validation and test

(X_validation,

 X_test,

 Y_validation,

 Y_test) = train_test_split(X_validation_and_test, Y_validation_and_test, test_size=validate_to_test_ratio)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D



model = Sequential([

    Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=X.shape[1:]),

    Conv2D(64, kernel_size=3, padding='same', activation='relu'),

    Conv2D(128, kernel_size=3, padding='same', activation='relu'),

    MaxPooling2D(pool_size=2),

    

    Conv2D(128, kernel_size=3, padding='same', activation='relu'),

    Conv2D(192, kernel_size=3, padding='same', activation='relu'),

    MaxPooling2D(pool_size=2),

    

    Conv2D(192, kernel_size=5, padding='same', activation='relu'),

    MaxPooling2D(pool_size=2, padding='same'),

    

    Dropout(0.25),

    Flatten(),

    Dense(256, activation='relu'),

    Dropout(0.5),

    Dense(10, activation='softmax')

])



model.summary()
model.compile(optimizer='adam', # adam, sgd, adadelta

              loss='categorical_crossentropy',

              metrics=['accuracy']) # sparse_categorical_crossentropy, accuracy
from keras.callbacks import ReduceLROnPlateau, EarlyStopping



reduce_learning_reducer = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.3, min_lr=0.00001)

#early_stopper = EarlyStopping(patience=3)

early_stopper = EarlyStopping(monitor = 'val_loss', min_delta = 1e-10, patience = 10, verbose = 1, restore_best_weights = True)



training_result = model.fit(X_train, Y_train,

                            batch_size=96,

                            epochs=150,

                            validation_data=(X_validation, Y_validation),

                            callbacks=[reduce_learning_reducer, early_stopper])
test_result = model.test_on_batch(X_test, Y_test)

test_result
score = model.evaluate(X_test, Y_test, verbose=0)



print('Test loss:', score[0])

print('Test accuracy:', score[1])
benchmark = pd.read_csv('../input/digit-recognizer/test.csv')



benchmark = benchmark / 255

benchmark = reshape_data(benchmark.values)
prediction = model.predict(benchmark)
submission = pd.DataFrame({

    'ImageId': [i + 1 for i in range(len(benchmark))],

    'Label': [values.argmax() for values in prediction]

})



submission.head()
submission.to_csv('submission.csv', index=False)