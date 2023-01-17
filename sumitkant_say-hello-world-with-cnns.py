import pandas as pd

import numpy as np



train = pd.read_csv('../input/digit-recognizer/train.csv')

test  = pd.read_csv('../input/digit-recognizer/test.csv')

sub   = pd.read_csv('../input/digit-recognizer/sample_submission.csv')



print ('Train Shape :', train.shape)

print ('test Shape :' , test.shape)
SEED = 42

from tensorflow.random import set_seed

from numpy.random import seed

seed(SEED)

set_seed(SEED)
import xgboost as xgb

X = train.drop('label', axis = 1).values

Y = train.label.values





from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = SEED)

print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)



train_dm = xgb.DMatrix(data = X_train, label = y_train)

test_dm  = xgb.DMatrix(data = X_test,  label = y_test)



params = {

    'max_depth'             : 5, 

    'eta'                   : 0.03, 

    'min_child_weight'      : 50, 

    'num_boost_round'       : 250, 

    'objective'             :'multi:softprob', 

    'seed'                  : SEED, 

    'num_class'             : 10,

    'silent'                : 1,

    'colsample_bytree'      : 0.5

}

%time model  = xgb.train(params, train_dm, num_boost_round = params['num_boost_round'])
from sklearn.metrics import accuracy_score, classification_report

print ('TRAIN ACCURACY : ', accuracy_score(y_train, [x.argmax() for x in model.predict(train_dm)]))

print ('VAL ACCURACY : '  , accuracy_score(y_test,  [x.argmax() for x in model.predict(test_dm)]))
score_dm = xgb.DMatrix(data = test.values)

sub_xgb = pd.Series([x.argmax() for x in model.predict(score_dm)], index  = np.arange(test.shape[0]) + 1).reset_index()

sub_xgb.columns = sub.columns

sub_xgb.to_csv('submission_xgboost.csv', index = False)
X_train = train.drop('label', axis = 1).values.reshape(train.shape[0], 784).astype('float')/255

y_train = train.label.values

X_test  = test.values.reshape(test.shape[0], 784).astype('float')/255

print (X_train.shape, X_test.shape)
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)

print (y_train.shape)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense



def baseline_model():

    model = Sequential([

        Dense(784, input_dim = (784), activation = 'relu'),

        Dense(10, activation = 'softmax'),

    ])

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model
model = baseline_model()

model.summary()
%time history = model.fit(X_train, y_train, validation_split = 0.1, epochs=40, batch_size=200, verbose=0)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('LOSS')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()





plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('ACCURACY')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()    
import numpy as np

sub_mlp = pd.Series([x.argmax() for x in model.predict(X_test)], index  = np.arange(test.shape[0]) + 1).reset_index()

sub_mlp.columns = sub.columns

sub_mlp.to_csv('submission_baseline.csv', index = False)
X_train = train.drop('label', axis = 1).values.reshape(train.shape[0], 28, 28, 1).astype('float')/255

y_train = to_categorical(train.label.values)

X_test  = test.values.reshape(test.shape[0], 28, 28, 1).astype('float')/255

print (X_train.shape, X_test.shape, y_train.shape)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten



def basic_cnn_model():

    model = Sequential([

        Conv2D(32, (5,5), input_shape = (28,28,1), activation = 'relu'),

        MaxPooling2D(2,2),

        Dropout(0.2),

        Flatten(),

        Dense(128, activation = 'relu'),

        Dense(10, activation = 'softmax')

    ])

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model
model = basic_cnn_model()

model.summary()
%time history = model.fit(X_train, y_train, validation_split = 0.1, epochs=40, batch_size=200, verbose=0)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('LOSS')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()





plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('ACCURACY')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()
# saving submission file

sub_cnn = pd.Series([x.argmax() for x in model.predict(X_test)], index  = np.arange(test.shape[0]) + 1).reset_index()

sub_cnn.columns = sub.columns

sub_cnn.to_csv('submission_basic_CNN.csv', index = False)
def large_cnn_model():

    model = Sequential([

        Conv2D(32, (5,5), input_shape = (28,28,1), activation = 'relu'),

        MaxPooling2D(2,2),

        Conv2D(32, (3,3), activation = 'relu'),

        MaxPooling2D(2,2),

        Dropout(0.2),

        Flatten(),

        Dense(128, activation = 'relu'),

        Dense(10, activation = 'softmax')

    ])

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model
model = large_cnn_model()

model.summary()
%time history = model.fit(X_train, y_train, validation_split = 0.1, epochs=40, batch_size=200, verbose=0)
# saving submission file

sub_large_cnn = pd.Series([x.argmax() for x in model.predict(X_test)], index  = np.arange(test.shape[0]) + 1).reset_index()

sub_large_cnn.columns = sub.columns

sub_large_cnn.to_csv('submission_large_CNN.csv', index = False)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('LOSS')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()





plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('ACCURACY')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()
ls -l