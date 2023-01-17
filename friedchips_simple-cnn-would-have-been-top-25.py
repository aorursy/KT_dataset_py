import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten

from keras.regularizers import l1
df_train = pd.read_csv('../input/train.csv', dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float32} ) # float32 is enough :)



train_X = np.resize(df_train['acoustic_data'].values, (len(df_train['acoustic_data']) // 150000, 150000)).astype(np.float32) # rearrange into 150k windows

train_X = (train_X - np.mean(train_X)) ** 2 # calculate power density

train_X = np.mean(train_X.reshape(-1,300,500), axis=2) # downsample 500x



train_y = np.resize(df_train['time_to_failure'].values, (len(df_train['time_to_failure']) // 150000, 150000))[:,-1] # train_y is ttf on right window edge



del df_train # free some memory



print(train_X.shape, train_y.shape)
df_subm = pd.read_csv('../input/sample_submission.csv')



test_X = []



for fname in df_subm['seg_id'].values:

    test_X.append(pd.read_csv('../input/test/' + fname + '.csv').acoustic_data.values.astype(np.int16))

test_X = np.array(test_X).astype(np.float32)



test_X = (test_X - np.mean(test_X)) ** 2  # calculate power density

test_X = np.mean(test_X.reshape(-1,300,500), axis=2) # downsample 500x



print(test_X.shape)
fig, axes = plt.subplots(1,2)

fig.set_size_inches(16,5)

axes[0].grid(True)

axes[1].grid(True)



axes[0].plot(train_X[28]);

axes[0].plot(train_X[38]);

axes[1].plot(test_X[28]);

axes[1].plot(test_X[38]);
ratio_mean_train_test = np.mean(test_X) / np.mean(train_X)

print('Ratio of mean power in train/test : ', np.mean(test_X) / np.mean(train_X))
def prepare_X_for_cnn(data):

    data = np.log10(data) # take log10 to handle the huge peaks

    data -= np.mean(data) # remove mean

    data /= np.std(data) # set std to 1.0

    data = np.expand_dims(data, axis=-1) # reshaping for CNN/RNN

    return data



train_X = prepare_X_for_cnn(train_X)

test_X  = prepare_X_for_cnn(test_X)
fig, axes = plt.subplots(1,2)

fig.set_size_inches(16,5)

axes[0].grid(True)

axes[1].grid(True)



axes[0].plot(train_X[28]);

axes[0].plot(train_X[38]);

axes[0].plot(train_X[29]);

axes[1].plot(test_X[28]);

axes[1].plot(test_X[38]);
def model_cnn():

    model = Sequential([

        Conv1D(filters=16, kernel_size=3, activation='relu', input_shape = train_X.shape[1:]),

        MaxPooling1D(2),

        Conv1D(filters=128, kernel_size=3, activation='relu'),

        MaxPooling1D(2),

        Conv1D(filters=16, kernel_size=3, activation='relu'),

        MaxPooling1D(2),

        Flatten(),

        Dropout(0.1),

        Dense(16, activation='relu', kernel_regularizer=l1(0.01)),

        Dense(16, activation='relu', kernel_regularizer=l1(0.01)),

        Dense(1, activation='linear') # regression

    ])

    model.compile(

        loss='mse',

        optimizer='adam',

        metrics=['mae']

    )

    return model



model = model_cnn()

model.summary()

model.save_weights('/tmp/model_weights_init.h5')
test_y_pred = []

num_iter = 64



for i in range(num_iter):

    model.load_weights('/tmp/model_weights_init.h5')

    model.fit(train_X, train_y, epochs=16,  verbose=0)

    test_y_pred.append(model.predict(test_X))

    

test_y_pred = np.array(test_y_pred).reshape(num_iter,-1)
test_y_pred_avg = np.mean(test_y_pred, axis=0)

test_y_pred_avg
fig, axes = plt.subplots(1,1)

fig.set_size_inches(16,5)

axes.grid(True)



axes.plot(model.predict(train_X));

axes.plot(train_y);
df_subm['time_to_failure'] = test_y_pred_avg * ratio_mean_train_test

df_subm.to_csv('submission.csv', index=False)