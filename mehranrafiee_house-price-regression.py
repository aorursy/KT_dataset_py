# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

from matplotlib import pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_log_error





import tensorflow as tf

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Dropout

from tensorflow.keras import optimizers, losses 

from tensorflow.keras.callbacks import *

from tensorflow.keras import backend as K

from tensorflow.keras.metrics import mean_squared_logarithmic_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../working/"))



# Any results you write to the current directory are saved as output.
seed_value= 0



# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

import os

os.environ['PYTHONHASHSEED']=str(seed_value)



# 2. Set `python` built-in pseudo-random generator at a fixed value

import random

random.seed(seed_value)



# 3. Set `numpy` pseudo-random generator at a fixed value

import numpy as np

np.random.seed(seed_value)



# 4. Set `tensorflow` pseudo-random generator at a fixed value

import tensorflow as tf

tf.set_random_seed(seed_value)



# 5. Configure a new global `tensorflow` session

from keras import backend as K

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)
df = pd.read_csv('../input/train.csv')

df.head()



df.describe()
def preprocess(df):

    df.fillna('0', inplace=True)

    

    # Categorical boolean mask

    categorical_feature_mask = df.dtypes==object

    # filter categorical columns using mask and turn it into a list

    categorical_cols = df.columns[categorical_feature_mask].tolist()

    

    le = LabelEncoder()

    for col in categorical_cols:

        try:

            df[col] = le.fit_transform(df[col])

        except:

            pass

        

    

    min_max_scaler = preprocessing.MinMaxScaler()

    np_scaled = min_max_scaler.fit_transform(df)

    df = pd.DataFrame(np_scaled, columns=df.columns)

    

    return df





def root_mean_squerd_log_error(y_pred, y_true):

    return np.sqrt(np.mean(np.power(np.log(y_pred+1) - np.log(y_true+1), 2)))
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

# df['SalePrice'].hist()

sns.distplot(df['SalePrice'],bins = 15);
# sns.distplot(np.log1p(df['SalePrice']), bins=15)
sns.heatmap(df.corr())
df['SalePrice'].plot.line()
random_state = 20
y = df['SalePrice']

X = preprocess(df[df.columns[:-1]])







X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)



print('shape of X_train: {} and shape of y_train: {}'.format(X_train.shape, y_train.shape))

print('shape of X_val: {} and shape of y_val: {}'.format(X_val.shape, y_val.shape))
X.head()
def create_submission(model, name='model', fit=True):

    if fit:

        model.fit(X, y)

    name = name + '_submission.csv'

    

    test_data = pd.read_csv('../input/test.csv')

    test_data['SalePrice'] = model.predict(preprocess(test_data))

    

    result = test_data[(['Id', 'SalePrice'])]

    result.to_csv(name, index=False)

regressor = DecisionTreeRegressor(max_depth=20,max_leaf_nodes=800, random_state=random_state)

regressor.fit(X_train, y_train)

# regressor.score(X_val, y_val)

root_mean_squerd_log_error(regressor.predict(X_val), y_val)
create_submission(regressor, 'DT')
regressor = SVR(kernel='linear', gamma='scale',epsilon=1, C=100000)

regressor.fit(X_train, y_train)

root_mean_squerd_log_error(regressor.predict(X_val), y_val)
create_submission(regressor, 'SVM_linear')
regressor = AdaBoostRegressor(random_state=random_state)

regressor.fit(X_train, y_train)

root_mean_squerd_log_error(regressor.predict(X_val), y_val)
create_submission(regressor, 'AdaBoost')
regressor = RandomForestRegressor(n_estimators=100, random_state=random_state)

regressor.fit(X_train, y_train)

root_mean_squerd_log_error(regressor.predict(X_val), y_val)
create_submission(regressor, 'RandomForest')


def build_model(input_shape, l1_rate=0.0, dropout_rate=0.5):

    

    activations = 'elu'

    X_input = Input(shape=input_shape)

    

    X = Dense(300, name='dense1')(X_input)

#     X = BatchNormalization()(X)

    X = Activation(activations)(X)

    

    X = Dense(500, name='dense2') (X)

#     X = BatchNormalization()(X)

    X = Activation(activations)(X)

    

    X = Dense(500, name='dense3') (X)

#     X = BatchNormalization()(X)

    X = Activation(activations)(X)

    

#     X = Dense(600, name='dense4') (X)

#     X = BatchNormalization()(X)

#     X = Activation(activations)(X)

    

#     X = Dense(600, name='dense5') (X)

#     X = BatchNormalization()(X)

#     X = Activation(activations)(X)

    

#     X = Dense(500, name='dense6') (X)

#     X = BatchNormalization()(X)

#     X = Activation(activations)(X)

    

#     X = Dense(500, name='dense7') (X)

#     X = BatchNormalization()(X)

#     X = Activation(activations)(X)

    

#     X = Dense(500, name='dense8') (X)

#     X = BatchNormalization()(X)

#     X = Activation(activations)(X)

    

#     X = Dense(500, name='dense9') (X)

#     X = BatchNormalization()(X)

#     X = Activation(activations)(X)

    

    X = Dense(500, name='dense10') (X)

#     X = BatchNormalization()(X)

    X = Activation(activations)(X)

    

    X = Dense(500, activation=activations, name='dense11') (X)

    X = Dense(500, activation=activations, name='dense12') (X)

    X = Dense(300, activation=activations, name='dense13') (X)

    X = Dense(200, activation=activations, name='dense14') (X)

    X = Dense(100, activation=activations, name='dense15') (X)

    X = Dense(100, activation=activations, name='dense16') (X)

    X = Dense(100, activation=activations, name='dense17') (X)

    

    output = Dense(1, name='out') (X)

    

    return Model(X_input, output)
def RMSLE(y_pred, y_true):

    return K.sqrt(K.mean(K.pow(K.log(y_pred+1) - K.log(y_true+1), 2)))




reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=5, verbose=1)



checkpoint = ModelCheckpoint('checkpoint.h5', monitor='val_loss', 

                             verbose=0, save_best_only=True, mode='min')



model = build_model((80, ))

optimizer = optimizers.Adam(lr=0.001, decay=0.0001)



model.compile(optimizer, loss=RMSLE)

history = model.fit(X, y, epochs=50, validation_data=(X_val, y_val), callbacks=[reduce_lr, checkpoint])
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['loos', 'val_loss'])
root_mean_squerd_log_error(np.squeeze(model.predict(X_val)), y_val)
# create_submission(load_model('checkpoint.h5'), name='ANN', fit=False)
create_submission(model, name='ANN_reg',fit=False)
# model.evaluate(X_val, y_val)

# load_model('checkpoint.h5').evaluate(X_val, y_val)