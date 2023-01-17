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
import os
import time

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
def averaged_by_N_rows(a, n):
    """ Набросок функции усредняющий n-ки строк в матрице
    """
    shape = a.shape
    assert len(shape) == 2
    assert shape[0] % n == 0
    b = a.reshape(shape[0] // n, n, shape[1])
    mean_vec = b.mean(axis=1)
    return mean_vec
    
demographic = pd.read_csv("/kaggle/input/button-tone-sz/demographic.csv")
demographic
demographic[" group"].mean()
for i, t in enumerate(list(demographic[" group"])):
    if t:
        print(f"{i})   SZ")
    else:
        print(f"{i})   HC")
        
diagnosis_dict = dict(zip(demographic.subject, demographic[" group"]))
del demographic
diagnosis_dict[25]
electrodes_list = list(pd.read_csv("/kaggle/input/button-tone-sz/columnLabels.csv").columns[4:])
print(electrodes_list)
(9216 * len(electrodes_list))
9216 / 4
if input("Начинаем отбор данных заново, затирая то, что есть?").lower() in ("yes", "да"):

    N_AVERAGED = 8
    X = np.zeros((81 * 100,  9216 * len(electrodes_list) // N_AVERAGED), dtype="float32")
    Y = np.zeros(len(X))

    part1_path = "../input/button-tone-sz"
    part2_path = "../input/buttontonesz2"

    # Вытаскиваем только те эксперименты, где измерений было 9216 (чаще всего имено столько раз производ)
    x_counter = 0
    column_list = pd.read_csv("/kaggle/input/button-tone-sz/columnLabels.csv").columns
    for person_number in tqdm(range(1, 81 + 1)):
                
        
        csv_path = f"{part1_path}/{person_number}.csv/{person_number}.csv"
        if not os.path.exists(csv_path):
            csv_path = f"{part2_path}/{person_number}.csv/{person_number}.csv"
        df = pd.read_csv(csv_path, 
                    header=None,
                    names=column_list
                    )
        #df = df[column_list]
        trials_list = set(df.trial)

        
        

        for t1, trial_number in enumerate(trials_list):
            number_of_trials = len(df[df.trial == trial_number])
            if number_of_trials == 9216.0:
                current_sample_matrix = df[df.trial == trial_number][electrodes_list].values
                averaged_by_N = averaged_by_N_rows(current_sample_matrix, n=N_AVERAGED)
                averaged_by_N_big_vec = averaged_by_N.reshape(-1)
                X[x_counter] = averaged_by_N_big_vec.astype(np.float32)
                Y[x_counter] = diagnosis_dict[person_number]
                x_counter += 1
            #print(f"Испытание под номером {trial_number} содержит измерений {number_of_trials} ")
    print("Всего испытаний с подходящим числом измерений - ", x_counter)
    X = X[: x_counter]
    Y = Y[: x_counter]
        

print("Всего испытаний с подходящим числом измерений - ", x_counter)


if input("Записать X_big и Y_big в файл?").lower() in ("yes", "да"):
    X.tofile("goodX_every4.bin")
    Y.astype("uint8").tofile("goodY_every4.bin")
from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)
obj2delete_list = "X_test X_train Y_train, nn model".split()
for t in obj2delete_list:
    try:
        exec(f"del {t}")
    except:
        print(t, "is no longer present")
import gc; gc.collect()

X[1856].max()
if input("Считать X_big и Y_big из файла?").lower() in ("yes", "да"):
    Y = np.fromfile("./goodY_every4.bin", dtype=np.uint8)    
    X = np.fromfile("./goodX_every4.bin", dtype=np.float32).reshape(len(Y), -1)
    print("Готово! Время:", time.ctime())
import keras
from keras.models import Sequential
from keras.layers import Dense
     
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras import backend as K

from keras.callbacks import ModelCheckpoint


filepath="./best_model2.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
len(X.reshape(-1)) / len(electrodes_list)

#normalize?
from sklearn.preprocessing import normalize
X_norm = (normalize(X.reshape(-1, 70), axis=0, norm='max')).reshape(X.shape)
X_train_norm, X_test_norm, Y_train_norm, Y_test_norm = train_test_split(X_norm, Y, test_size=0.2, shuffle=True, random_state=42)
_norm = X


X_train_norm.shape, len(Y_train_norm), 161280 / len(electrodes_list)
X_train_2d = X_train_norm.reshape(X_train_norm.shape[0], len(electrodes_list), X_train_norm.shape[1] // len(electrodes_list), 1)
X_test_2d = X_test_norm.reshape(X_test_norm.shape[0], len(electrodes_list), X_test_norm.shape[1] // len(electrodes_list), 1)
#X_train_2d = X_train.reshape(X_train.shape[0], len(electrodes_list), X_train.shape[1] // len(electrodes_list), 1)
#X_test_2d = X_test.reshape(X_test.shape[0], len(electrodes_list), X_test.shape[1] // len(electrodes_list), 1)

X_train_2d.max()
X_train_2d.shape
(X_train_2d.shape[1:] +(1, ))
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(X_train_2d.shape[1:])))
model.add(MaxPooling2D(pool_size=(5, 5)))


model.add(Flatten())
model.add(Dense(350, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(2),
              metrics=['acc'])

X_train_2d.shape, X_train_norm.shape, X_train_2d.shape
model = Sequential()
model.add(Conv2D(5, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(X_train_2d.shape[1:])))
model.add(MaxPooling2D(pool_size=(5, 5)))


model.add(Flatten())
model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['acc'])
model.fit(X_train_2d, Y_train_norm,
          batch_size=50,
          epochs=300,
          verbose=1,
          validation_data=(X_test_2d, Y_test), callbacks=[checkpoint])

