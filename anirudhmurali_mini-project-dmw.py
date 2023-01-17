import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout # components of network

x_train_set_fpath = '../input/X_test_sat4.csv'
y_train_set_fpath = '../input/y_test_sat4.csv'
print ('Loading Training Data')
X_train = pd.read_csv(x_train_set_fpath)
print ('Loaded 28 x 28 x 4 images')
Y_train = pd.read_csv(y_train_set_fpath)
print ('Loaded labels')

X_train = X_train.as_matrix()
Y_train = Y_train.as_matrix()
print ('We have',X_train.shape[0],'examples and each example is a list of',X_train.shape[1],'numbers with',Y_train.shape[1],'possible classifications.')

#First we have to reshape each of them from a list of numbers to a 28*28*4 image.
X_train_img = X_train.reshape([99999,28,28,4]).astype(float)
print (X_train_img.shape)
Y_train[0]
# # model = Sequential([
# #     Dense(4, input_shape=(3136,), activation='softmax')
# # ])

# model = Sequential([
#     Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28,28,4)), 
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(64, (5, 5), activation='relu'),
#    MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(1024, activation='relu'),
#     Dropout(0.25),
#     Dense(256, activation='relu'),
# #     Dropout(0.5),
#     Dense(4, activation='softmax')
# ])

X_train = (X_train-X_train.mean())/X_train.std()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
# model.fit(X_train_img,Y_train,batch_size=32, epochs=3, verbose=1, validation_split=0.01)

model = Sequential([
    Dense(1024, input_shape=(3136,), activation='relu'),
    Dropout(0.25),
    Dense(256,activation="relu"),
    
    Dense(32,activation="relu"),
    Dense(4,activation="softmax")
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train,Y_train,batch_size=32, epochs=5, verbose=1, validation_split=0.1)
preds = model.predict(X_train[-1000:], verbose=1)
ix = 54 #Type a number between 0 and 999 inclusive
#Tells what the image is
print ('Prediction:\n{:.1f}% probability barren land,\n{:.1f}% probability trees,\n{:.1f}% probability grassland,\n{:.1f}% probability other\n'.format(preds[ix,0]*100,preds[ix,1]*100,preds[ix,2]*100,preds[ix,3]*100))

print ('Ground Truth: ',end='')
if Y_train[99999-(1000-ix),0] == 1:
    print ('Barren Land')
elif Y_train[99999-(1000-ix),1] == 1:
    print ('Trees')
elif Y_train[99999-(1000-ix),2] == 1:
    print ('Grassland')
else:
    print ('Other')
ix = 56 #Type a number between 0 and 999 inclusive
#Tells what the image is
print ('Prediction:\n{:.1f}% probability barren land,\n{:.1f}% probability trees,\n{:.1f}% probability grassland,\n{:.1f}% probability other\n'.format(preds[ix,0]*100,preds[ix,1]*100,preds[ix,2]*100,preds[ix,3]*100))

print ('Ground Truth: ',end='')
if Y_train[99999-(1000-ix),0] == 1:
    print ('Barren Land')
elif Y_train[99999-(1000-ix),1] == 1:
    print ('Trees')
elif Y_train[99999-(1000-ix),2] == 1:
    print ('Grassland')
else:
    print ('Other')

model.save_weights("landuse.h5")