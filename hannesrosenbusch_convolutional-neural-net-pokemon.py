import numpy as np
import pandas as pd
import random
from PIL import Image
import time
from matplotlib.pyplot import imshow, plot
random.seed(42)
#read in data
labels = pd.read_csv('../input/names_and_strengths.csv', header = 'infer')
data = np.load('../input/poke_image_data.npy')
data = np.dot(data[:,:,:,:3], [0.299, 0.587, 0.114]) #convert to grayscale
data = np.expand_dims(data, axis=4)
#there are multiple images per pokemon
#i need the names of the pokemon to make sure that the same pokemon does not appear in both training and testing
names = list(sorted(set(labels.name)))
print(len(names)) #number pokemon
#random split not on all the data rows, but the unique pokemon (names)
name_sample = random.sample(names, round(0.7*len(names)))
print(len(name_sample)) #number different pokemon in training set
#100x100 pixels are a bit much
#I will do 64x64
resized_data = np.ndarray((data.shape[0], 64,64))
for im in range(data.shape[0]):
    resized_pic = Image.fromarray(data[im,:,:,0])
    resized_pic = resized_pic.resize((64,64))
    resized_pic = np.array(resized_pic)
    resized_data[im] = resized_pic
#probably good enough quality?
imshow(Image.fromarray(resized_data[42]))
#split all rows into train and test set but keep same pokemon in either training or test (no overlap)
resized_data = np.expand_dims(resized_data, axis=4)
y_train = labels[labels.name.isin(name_sample)]
x_train = resized_data[labels.name.isin(name_sample)]
y_test = labels[~labels.name.isin(name_sample)]
x_test = resized_data[~labels.name.isin(name_sample)]
print(y_train.shape)
print(x_train.shape)
print(y_test.shape)
print(x_test.shape)
#check if reshaping etc didn't break anything
example_pic = Image.fromarray(x_train[42,:,:,0])
imshow(example_pic)
print(x_train[42,:,:,0].shape)
#verify that same pokemons do not show up in both train and test
np.intersect1d(y_train.name, y_test.name).size == 0
#small batch, lots of epochs, lots of regularization, train on MAE for interpretability

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

batch_size = 32
epochs = 100

model = Sequential()
model.add(Conv2D(40, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=[64,64,1]))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))


# model.add(BatchNormalization())
# model.add(Conv2D(50, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_absolute_error',
              optimizer=keras.optimizers.Adam(),
              metrics=['mae'])

model.summary()
#fit model
history = model.fit(x_train, y_train.strength,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
#make predictions
scores = model.predict(x_test)
#model is pretty overfit
print(np.mean(abs(scores[:,0]-y_test.strength)))
#combine individual predictions made for each pokemon
y_test['preds'] = scores[:,0]
y_test_pok = y_test.groupby(['name']).mean()
print(np.mean(abs(y_test_pok.preds-y_test_pok.strength)))
print(np.corrcoef(y_test_pok.preds, y_test_pok.strength))
#looks a little better
#plot true vs predicted
import seaborn as sns
sns.lmplot(x='preds',y='strength',data=y_test_pok,fit_reg=True) 