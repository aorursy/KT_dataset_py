import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
sample_submission = pd.read_csv("../input/sample_submission.csv")

test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
print(train.info())

display(train)
print(test.info())

display(test)
train.describe()
test.describe()
import matplotlib.pyplot as plt

import seaborn as sns



sns.heatmap(train.corr(), annot=True, linewidth=0.2)

fig=plt.gcf()

fig.set_size_inches(20,20)

plt.show()
Y = train['DEFCON_Level']



X = train.drop(['DEFCON_Level', 'ID'], axis=1)

X['Closest_Threat_log'] = np.log(X['Closest_Threat_Distance(km)'])

X['Troops_Mobilized_log'] = np.log(X['Troops_Mobilized(thousands)'])



test['Closest_Threat_log'] = np.log(test['Closest_Threat_Distance(km)'])

test['Troops_Mobilized_log'] = np.log(test['Troops_Mobilized(thousands)'])



combined = pd.concat([X, test.drop(['ID'], axis=1)], axis=0)

display(combined)
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Multiply

from tensorflow.keras.optimizers import Adam, SGD, Adagrad

from tensorflow.keras.regularizers import l1_l2

from tensorflow.keras.initializers import he_normal

from sklearn.model_selection import train_test_split as tts

from keras import backend as K



def xavier(shape, dtype=None):

    return np.random.rand(*shape)*np.sqrt(1/(17))

def output_w(shape, dtype=None):

    return np.ones(shape)*np.array([0, 0.5, 0.22, 0.22, 0.05, 0.01])



def c_model(shape, lr=0.001):

    i = Input(shape)

    x = Dense(100, activation='relu', kernel_initializer=xavier)(i)

    x = Dense(100, activation='relu', kernel_initializer=xavier)(x)

    x = Dense(100, activation='relu', kernel_initializer=xavier)(x)

    o = Dense(6, activation='softmax', kernel_initializer=xavier)(x)

    

    opt = Adam(lr=lr, amsgrad=True)

    #opt = SGD(lr=lr, momentum=0.25, nesterov=True)

    #opt = Adagrad(lr=lr)

    x = Model(inputs=i, outputs=o)

    x.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return x
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

lrr = ReduceLROnPlateau(monitor = 'val_accuracy',

                         patience = 7,

                         verbose = 1,

                         factor = 0.5,

                         min_lr = 1e-5)



es = EarlyStopping(monitor='val_loss',

                   mode='min',

                   verbose=1,

                   patience=50,

                   restore_best_weights=True)



epochs = 2000

batch_size = 128



from sklearn.preprocessing import StandardScaler as ss

scale = ss()

scale.fit(combined)



tx, vx, ty, vy = tts(scale.transform(X), Y, test_size=0.25, random_state=121)

model = c_model(tx.shape[1:], 0.003)



history = model.fit(tx, ty, validation_data=(vx, vy),

                    epochs=epochs, batch_size=batch_size,

                    verbose=2, callbacks=[lrr, es])
model.evaluate(scale.transform(X), Y)

pred = np.argmax(model.predict(scale.transform(test[X.columns])), axis=1)

n = len(Y)/100

print([(x, sum(Y==x), sum(Y==x)/n) for x in range(1, 6)])

n = len(pred)/100

print([(x, sum(pred==x), sum(pred==x)/n) for x in range(1, 6)])



predictions = pd.DataFrame()

predictions['ID'] = test['ID']

predictions['DEFCON_Level'] = pred

predictions.to_csv("submission.csv", index=False)