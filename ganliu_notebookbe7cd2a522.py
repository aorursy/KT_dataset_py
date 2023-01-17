import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from keras.models import Model

from keras.layers import Input, Dense, Dropout, merge
np.random.seed(12)



train = pd.read_csv(r'../input/train.csv')

test = pd.read_csv(r'../input/test.csv')


new_index = train.index.tolist()

np.random.shuffle(new_index)

train = train.iloc[new_index, :].reset_index(drop=True)



ss = StandardScaler()

le = LabelEncoder()

ohe = OneHotEncoder()



train.iloc[:, 2:] = ss.fit_transform(train.iloc[:, 2:].values)

label_encoded = le.fit_transform(train['species'])

label_ohe = ohe.fit_transform(label_encoded[:, np.newaxis]).toarray()





training_ratio = 0.8

training_size = int(train.shape[0] * training_ratio)



X_train_margin = train.loc[:training_size-1, 'margin1':'margin64'].values

X_train_shape = train.loc[:training_size-1, 'shape1':'shape64'].values

X_train_texture = train.loc[:training_size-1, 'texture1':'texture64'].values

Y_train = label_ohe[:training_size]



X_valid_margin = train.loc[training_size:, 'margin1':'margin64'].values

X_valid_shape = train.loc[training_size:, 'shape1':'shape64'].values

X_valid_texture = train.loc[training_size:, 'texture1':'texture64'].values

Y_valid = label_ohe[training_size:]



X_test_margin = test.loc[:, 'margin1':'margin64'].values

X_test_shape = test.loc[:, 'shape1':'shape64'].values

X_test_texture = test.loc[:, 'texture1':'texture64'].values


input_margin = Input(shape=(64,), name='input_margin')

input_shape = Input(shape=(64,), name='input_shape')

input_texture = Input(shape=(64,), name='input_texture')



level_margin = Dense(512, activation='relu')(input_margin)

level_margin = Dropout(0.3)(level_margin)

#level_margin = Dense(99, activation='softmax')(level_margin)



level_shape = Dense(512, activation='relu')(input_shape)

level_shape = Dropout(0.3)(level_shape)

#level_shape = Dense(99, activation='softmax')(level_shape)



level_texture = Dense(512, activation='relu')(input_texture)

level_texture = Dropout(0.3)(level_texture)

#level_texture = Dense(99, activation='softmax')(level_texture)



merged = merge([level_margin, level_shape, level_texture], mode='concat')

merged = Dropout(0.3)(merged)



level_output = Dense(99, activation='softmax', name='level_output')(merged)

model = Model(input=[input_margin, input_shape, input_texture], output=[level_output])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(

    x={'input_margin': X_train_margin, 'input_shape': X_train_shape, 'input_texture': X_train_texture},

    y={'level_output': Y_train},

    validation_data=({'input_margin': X_valid_margin, 'input_shape': X_valid_shape, 'input_texture': X_valid_texture}, {'level_output': Y_valid}),

    nb_epoch=50, batch_size=128, verbose=2

)
import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(history.history['val_loss'])

plt.plot(history.history['loss'])

plt.xlim([10,300])

plt.ylim([0,0.1])
plt.plot(history.history['val_acc'])
model.layers
model.layers[4].get_weights()[1].shape