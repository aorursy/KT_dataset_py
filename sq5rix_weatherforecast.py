# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_dir = '/kaggle/input/jena-climate/'

fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
# importujemy narzędzie do tworzenia wykresów, jedno z wielu (bardzo wielu) 

from matplotlib import pyplot as plt

# ładujemy dane do pand

df = pd.read_csv(fname)
# importujemy narzędzie do tworzenia wykresów, jedno z wielu (bardzo wielu) 

from matplotlib import pyplot as plt

# ładujemy dane do pand

df = pd.read_csv(fname)
# działają?

df[400000:500000].to_csv('duzyplik.csv')
df.columns
# malujemy wykresik

df.loc[400000:420000,[ 'Tdew (degC)', 'T (degC)']].plot()
# Zamieniamy index liczb naturalnych na DateTime. 

# df.index = pd.to_datetime(df['Date Time'])
# w ten sposób możemy malować wykresy ograniczone datami/czasem 

# df['rho (g/m**3)']['2016-10-01':'2016-12-01':12].plot()
df.hist(bins=20,figsize=(10,60), layout=(20,2))
float_data  = df.iloc[:,1:].to_numpy()
df.iloc[:,1:].to_numpy()
float_data.shape
mean = float_data[:200000].mean(axis=0)

float_data -= mean
std = float_data[:200000].std(axis=0)

float_data /= std
# Correlation matrix

def plotCorrelationMatrix(df):

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, dpi=120, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title('Correlation Matrix', fontsize=15)

    plt.show()
plotCorrelationMatrix(df)
def generator(data, lookback, delay, min_index, max_index,

              shuffle=False, batch_size=128, step=6):

    if max_index is None:

        max_index = len(data) - delay - 1

    i = min_index + lookback

    while 1:

        if shuffle:

            rows = np.random.randint(

                min_index + lookback, max_index, size=batch_size)

        else:

            if i + batch_size >= max_index:

                i = min_index + lookback

            rows = np.arange(i, min(i + batch_size, max_index))

            i += len(rows)



        samples = np.zeros((len(rows),

                           lookback // step,

                           data.shape[-1]))

        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):

            indices = range(rows[j] - lookback, rows[j], step)

            samples[j] = data[indices]

            targets[j] = data[rows[j] + delay][1]

        yield samples, targets

        
lookback = 1440

step = 6

delay = 144

batch_size = 128



train_gen = generator(float_data,

                      lookback=lookback,

                      delay=delay,

                      min_index=0,

                      max_index=200000,

                      shuffle=True,

                      step=step, 

                      batch_size=batch_size)

val_gen = generator(float_data,

                    lookback=lookback,

                    delay=delay,

                    min_index=200001,

                    max_index=300000,

                    step=step,

                    batch_size=batch_size)

test_gen = generator(float_data,

                     lookback=lookback,

                     delay=delay,

                     min_index=300001,

                     max_index=None,

                     step=step,

                     batch_size=batch_size)



# This is how many steps to draw from `val_gen`

# in order to see the whole validation set:

val_steps = (300000 - 200001 - lookback) // batch_size



# This is how many steps to draw from `test_gen`

# in order to see the whole test set:

test_steps = (len(float_data) - 300001 - lookback) // batch_size
val_gen
def evaluate_naive_method():

    batch_maes = []

    for step in range(val_steps):

        samples, targets = next(val_gen)

        preds = samples[:, -1, 1]

        mae = np.mean(np.abs(preds - targets))

        batch_maes.append(mae)

    print(np.mean(batch_maes))

    

evaluate_naive_method()
from keras.models import Sequential

from keras import layers

from keras.optimizers import RMSprop



model = Sequential()

model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(1))



model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen,

                              steps_per_epoch=500,

                              epochs=20,

                              validation_data=val_gen,

                              validation_steps=val_steps)
test_gen, test_steps
preds = model.predict_generator(test_gen, test_steps)
print (type(preds))

preds.shape
preds *= std

preds += mean