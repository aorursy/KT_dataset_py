# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping

from keras import losses

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error, f1_score

from sklearn.preprocessing import StandardScaler

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/dollar-prices-and-infos/database_15min.csv')

df.set_axis(

        ['Date', 'OpenP', 'MaxP', 'MinP', 'CloseP', 'Volume', 'fin', 'negociation', 'mme13', 'mme72', 'high_mean',

         'low_mean', 'diffmacd', 'deamacd', 'macdlh', 'difflh', 'dealh', 'Result'], axis=1, inplace=True)

df.head()
eps = 0.001  # prevent form log(0)

df['finantial'] = np.log(df.pop('fin') + eps)





df['vol_open'] = np.log(df['Volume']) * df['OpenP']

df['negoc_close'] = np.log(df['negociation']) * df['CloseP']
df['Result Class'] = 0 # 'DO NOTHING'

df['diffence'] = df['OpenP'] - df['Result']



df.loc[df['diffence'] > 6.5, 'Result Class'] = 1 # 'BUY'

df.loc[df['diffence'] < -6.5, 'Result Class'] = 2 # 'SELL'



df = df.drop(['Date', 'Result', 'diffence'], axis=1) # remove columns useless to the model

df.head()
X = shuffle(df.iloc[:20000, :])

X_train = X.iloc[:, :-1]

y_train = X.iloc[:, -1]

X_val = df.iloc[20000:, :-1]

y_val = df.iloc[20000:, -1]
def get_class_weight(classes, exp=1):

    '''

    Weight of the class is inversely proportional to the population of the class.

    There is an exponent for adding more weight.

    '''

    hist, _ = np.histogram(classes, bins=np.arange(4)-0.5)

    class_weight = hist.sum()/np.power(hist, exp)

    

    return class_weight



class_weight = get_class_weight(y_train)

print(class_weight)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)



model = Sequential()

model.add(Dense(units=500, activation='relu', input_dim=18))

model.add(Dense(units=500, activation='relu'))

model.add(Dense(units=314, activation='relu'))

model.add(Dense(units=314, activation='relu'))

model.add(Dense(units=314, activation='relu'))

model.add(Dense(units=314, activation='relu'))

model.add(Dense(units=195, activation='relu'))

model.add(Dense(units=120, activation='relu'))

model.add(Dense(units=1, activation='softmax'))





model.compile(loss='mse', optimizer='adam',

              metrics=['mae', 'mse'])



print(model.summary())

def plot_history(history):

    hist = pd.DataFrame(history.history)

    hist['epoch'] = history.epoch



    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('Mean Abs Error')

    plt.plot(hist['epoch'], hist['mae'],

             label='Train Error')

    plt.plot(hist['epoch'], hist['val_mae'],

             label='Val Error')

    plt.legend()

    plt.show()
EPOCHS = 1000

early_stop = EarlyStopping(monitor='val_loss', patience=50)



train_history = model.fit(X_train, y_train, epochs=EPOCHS,

                          validation_split=0.2, verbose=1, callbacks=[early_stop], class_weight=class_weight)



model.save('tf_classicator.h5')



plot_history(train_history)
X_val = scaler.transform(X_val)



preds = model.predict(X_val)



mae = mean_absolute_error(y_val, preds)



print(mae)
