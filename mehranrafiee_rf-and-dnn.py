# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



from sklearn.preprocessing import MinMaxScaler, OneHotEncoder,StandardScaler

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import lightgbm as lgb

from lightgbm import LGBMClassifier



from sklearn.svm import SVC

from sklearn.metrics import accuracy_score



from keras.layers import *

from keras.models import Model

from keras import losses, metrics, optimizers

from keras.callbacks import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/learn-together/train.csv')

df.head()
df.columns
to_normalize_columns = ['Elevation', 'Aspect', 'Slope',

       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',

       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
def preprocess(X):

    

    cols = X.columns

    scaler = StandardScaler()

    X = pd.DataFrame(scaler.fit_transform(X), columns=cols)

    

    return X
X = df.drop(['Id','Cover_Type'], axis=1)

X[to_normalize_columns] = preprocess(X[to_normalize_columns]) 

y = df['Cover_Type'] - 1



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X.head()
sns.distplot(y)
model = RandomForestClassifier(n_estimators=300, random_state=2019)



# model.fit(X_train, y_train)

# accuracy_score(model.predict(X_test), y_test)

model.fit(X, y)
test_df = pd.read_csv('../input/learn-together/test.csv')

# test_df.drop('Id', axis=1,inplace=True)

test_df[to_normalize_columns] = preprocess(test_df[to_normalize_columns])

test_df.head()
rf_preds = model.predict(test_df.drop('Id', axis=1))
def DNN(input_shape, dropout_rate=0):

    

    activation = 'elu'

    

    input_X = Input(input_shape)

    

    # Layer 1

    X = Dense(512) (input_X)

    X = BatchNormalization()(X)

    X = Activation(activation) (X)

    

    # Layer 2

    X = Dense(1024) (input_X)

    X = BatchNormalization()(X)

    X = Activation(activation) (X)

    

    # Layer 3

    X = Dense(512) (input_X)

    X = BatchNormalization()(X)

    X = Activation(activation) (X)

    

    # Layer 4

    X = Dense(512) (input_X)

    X = BatchNormalization()(X)

    X = Activation(activation) (X)

    

    # Layer 5

    X = Dense(512) (input_X)

    X = BatchNormalization()(X)

    X = Activation(activation) (X)

    

    # Layer 6

    X = Dense(512) (input_X)

    X = BatchNormalization()(X)

    X = Activation(activation) (X)

    

    # Layer 7

    X = Dense(512) (input_X)

    X = BatchNormalization()(X)

    X = Activation(activation) (X)

    

    # Layer 8

    X = Dense(512) (input_X)

    X = BatchNormalization()(X)

    X = Activation(activation) (X)

    

    # Layer 9

    X = Dense(512) (input_X)

    X = BatchNormalization()(X)

    X = Activation(activation) (X)

    

    

    # Layer 10

    X = Dense(512) (input_X)

    X = BatchNormalization()(X)

    X = Activation(activation) (X)

    

    

    # Layer 11

    X = Dense(512) (input_X)

    X = BatchNormalization()(X)

    X = Activation(activation) (X)

    

    

    # Layer 12

    X = Dense(512) (input_X)

    X = BatchNormalization()(X)

    X = Activation(activation) (X)

    

    

    # Layer 13

    X = Dense(512) (input_X)

    X = BatchNormalization()(X)

    X = Activation(activation) (X)

    

    # output Layer

    X = Dense(7)(X)

    out = Softmax() (X)

    

    return Model(input_X, out)

    
dnn_model = DNN(input_shape=(54,), dropout_rate=0.0)
optimizer = optimizers.Adam(lr=0.01, decay=1e-6)

dnn_model.compile(optimizer, loss=losses.categorical_crossentropy, metrics=['acc'])



check_point = ModelCheckpoint('model.h5',monitor='val_loss',verbose=1, mode='min', save_best_only=True)

reduce_lr = ReduceLROnPlateau(factor=0.01, mode='min')

dnn_model.fit(X_train, to_categorical(y_train), epochs=300, batch_size=1024,

              validation_data=(X_test, to_categorical(y_test)), callbacks=[check_point,reduce_lr])
from keras.models import load_model

dnn_model = load_model('model.h5')



dnn_preds = dnn_model.predict(test_df.drop('Id', axis=1))
dnn_preds = dnn_preds.argmax(axis=1)
def save_submission(test_df, preds, path='submission'):    

    '''

    test_path: test csv file

    preds    : predicted label from your model

    path     : where you want to save the csv file

    '''

    

    df = pd.DataFrame({'Id': test_df['Id'], 'Cover_Type': preds +1})

    df.to_csv(path+ '.csv', index=False)

    return df
rf_submission = save_submission(test_df, rf_preds, path='rf_submission')

dnn_submission = save_submission(test_df, dnn_preds, path='dnn_submission')
dnn_submission.head()