# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
X_1 = pd.read_csv("/kaggle/input/tuberculosis-drug-resistance-prediction-rif/X_trainData_1.csv")

Y_1 = pd.read_csv("/kaggle/input/tuberculosis-drug-resistance-prediction-rif/Y_trainData_1.csv")
X_1.info()
Y_inh = Y_1['RIF']

x = np.size(Y_inh)
for i in range(x):

    if(Y_inh[i] == -1):

        Y_inh.drop(i, inplace = True)

        X_1.drop(i,  inplace = True)
X = X_1.to_numpy()

Y = Y_inh.to_numpy()
from tensorflow import keras

from keras.layers import Dense, Dropout, Input, BatchNormalization

from keras.models import Model

from keras import regularizers

from keras.layers import concatenate

from keras.optimizers import Adam
#WDNN Implementation



input_  = keras.layers.Input(shape= (222,))

hidden1 = keras.layers.Dense(256, activation="relu")(input_)

hidden1 = BatchNormalization()(hidden1)

hidden1 = Dropout(0.5)(hidden1)

hidden2 = keras.layers.Dense(256, activation="relu")(hidden1)

hidden2 = BatchNormalization()(hidden2)

hidden2 = Dropout(0.5)(hidden2)

hidden2 = keras.layers.Dense(256, activation="relu")(hidden2)

hidden2 = BatchNormalization()(hidden2)

hidden2 = Dropout(0.5)(hidden2)

concat = keras.layers.concatenate([input_, hidden2])

preds = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-8))(concat)

model = keras.models.Model(inputs = [input_], outputs = [preds])

opt = Adam(lr=np.exp(-1.0*9))

model.compile(optimizer=opt,loss= keras.losses.BinaryCrossentropy(),metrics= keras.metrics.AUC())

model.summary()
history = model.fit(X,Y,epochs = 200,validation_data = None)
X_t = pd.read_csv("/kaggle/input/tuberculosis-drug-resistance-prediction-rif/X_testData_1.csv")

pred = model.predict(X_t.drop(columns = 'ID'))
k = pred
k = k.reshape((1000,))
Y_n = pd.read_csv("/kaggle/input/tuberculosis-drug-resistance-prediction-rif/Y_testData_1_nolabels_RIF.csv")
INH = pd.DataFrame({'ID':X_t['ID'],'RIF':k})

INH = pd.merge(Y_n,INH, on = 'ID')

INH.drop(columns = 'RIF_x',inplace = True)

INH.rename(columns = {'RIF_y':'RIF'}, inplace = True)

INH.to_csv('rif_res2.csv', index = False)
INH.head()
INH