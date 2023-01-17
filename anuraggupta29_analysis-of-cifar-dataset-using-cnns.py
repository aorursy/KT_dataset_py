import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns

(xtrain, ytrain),(xtest, ytest) = cifar10.load_data()
labelMap = {0: 'airplane',
           1: 'car',
           2: 'bird',
           3: 'cat',
           4: 'deer',
           5: 'dog',
           6: 'frog',
           7: 'horse',
           8: 'ship',
           9: 'truck'}
plt.imshow(xtrain[5])
ytrain_cat = to_categorical(ytrain, 10)
ytest_cat = to_categorical(ytest, 10)
print(xtrain.shape)
print(xtest.shape)
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.shape)
print(xtest.shape)
print(xtrain[0].max())
print(xtrain[0].min())
print(xtrain[0].mean())
def cnnmodel():
    model = Sequential()
    
    model.add(Conv2D(filters=32, kernel_size = (4,4), input_shape=(32,32,3), activation='relu'))
    model.add(MaxPool2D(pool_size = (4,4)))
    model.add(Conv2D(filters=64, kernel_size = (2,2), input_shape=(32,32,3), activation='relu'))
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    return model
model = cnnmodel()
early_stop = EarlyStopping(monitor='val_loss',patience=2)
model.fit(xtrain, ytrain_cat, 
          validation_data=(xtest, ytest_cat), 
          callbacks=[early_stop], 
          verbose=1, 
          epochs=10)
hist_df = pd.DataFrame(model.history.history)
display(hist_df.head())
hist_df[['loss', 'val_loss']].plot(title='Loss vs Epoch')
hist_df[['accuracy', 'val_accuracy']].plot(title='Accuracy vs Epoch')
pred = model.predict_classes(xtest)
print(pred.shape)
print(ytest.shape)
report = classification_report(ytest, pred)
print(report)
sns.heatmap(confusion_matrix(ytest, pred))
plt.imshow(xtrain[99])
print(labelMap[ytrain[99][0]])
pred_sample = model.predict_classes(xtrain[99].reshape(1,32,32,3))
print(labelMap[pred_sample[0]])