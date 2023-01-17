import numpy as np

import pandas as pd



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



from keras.models import Sequential

from keras.utils import np_utils

from keras.layers import Dense, Dropout, GaussianNoise, Conv1D

from keras.preprocessing.image import ImageDataGenerator



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')



Y_train = train['label'].values.astype('int32')

Y_train = np_utils.to_categorical(Y_train) 

train.drop(['label'], axis=1, inplace=True)



X_train = (train.values).astype('float32')

X_test = (test.values).astype('float32')
print('Y_train value form: {}'.format(Y_train[1]))

print('Which is 0 (1 in [0] position of the vector).')

plt.imshow(X_train[1].reshape(28,28))

plt.show()
scaler = StandardScaler()

scaler.fit(X_train)

X_sc_train = scaler.transform(X_train)

X_sc_test = scaler.transform(X_test)
pca = PCA(n_components=500)

pca.fit(X_train)



plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')
NCOMPONENTS = 100



pca = PCA(n_components=NCOMPONENTS)

X_pca_train = pca.fit_transform(X_sc_train)

X_pca_test = pca.transform(X_sc_test)

pca_std = np.std(X_pca_train)



print(X_sc_train.shape)

print(X_pca_train.shape)
inv_pca = pca.inverse_transform(X_pca_train)

inv_sc = scaler.inverse_transform(inv_pca)
def side_by_side(indexes):

    org = X_train[indexes].reshape(28,28)

    rec = inv_sc[indexes].reshape(28,28)

    pair = np.concatenate((org, rec), axis=1)

    plt.figure(figsize=(4,2))

    plt.imshow(pair)

    plt.show()

    

for index in range(0,10):

    side_by_side(index)
model = Sequential()

layers = 1

units = 128



model.add(Dense(units, input_dim=NCOMPONENTS, activation='relu'))

model.add(GaussianNoise(pca_std))

for i in range(layers):

    model.add(Dense(units, activation='relu'))

    model.add(GaussianNoise(pca_std))

    model.add(Dropout(0.1))

model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])



model.fit(X_pca_train, Y_train, epochs=100, batch_size=256, validation_split=0.15, verbose=2)
predictions = model.predict_classes(X_pca_test, verbose=0)



def write_predictions(predictions, fname):

    pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions}).to_csv(fname, index=False, header=True)



write_predictions(predictions, "pca-keras-mlp.csv")