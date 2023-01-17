import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

%matplotlib inline

np.random.seed(7)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten, Conv2D, MaxPool2D,BatchNormalization 
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator 
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='deep')

train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")

Y_train=train["label"]
X_train = train.drop(labels = ["label"],axis = 1) 

del train
# print(train)
g = sns.countplot(Y_train)
Y_train.value_counts()

X_train.isnull().any().describe()
test.isnull().any().describe()
X_train.head()
Y_train.head()
#Normalize the data
X_train = X_train / 255.0
test = test / 255.0
print(X_train.shape)
print(type(test))
X_train = X_train.to_numpy()
X_train=X_train.reshape(-1,28,28,1)
test = test.to_numpy()
test=test.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)
random_seed=3
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
g = plt.imshow(X_train[0][:,:,0])
datagen = ImageDataGenerator(zoom_range = 0.1, width_shift_range = 0.1, height_shift_range = 0.1, rotation_range = 10) 
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3), padding='Same', activation='relu', input_shape=(28,28,1),name='first'))
model.add(BatchNormalization())

model.add(Conv2D(filters=64,kernel_size=(3,3), activation='relu', name='second'))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu',name='third'))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation = "relu"))
model.add(BatchNormalization())
model.add(Dropout(0.15))

model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.summary()
opt=Adam(learning_rate=0.001)
model.compile(optimizer = opt , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
# history=model.fit(X_train,Y_train,batch_size=32,epochs=30,validation_data=(X_val, Y_val),verbose=1)
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = 32), epochs = 40, 
                              validation_data = (X_val, Y_val), callbacks = [learning_rate_reduction])
final_loss, final_acc = model.evaluate(X_val, Y_val, verbose=1)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
predictions = model.predict_classes(test, verbose=1)
test = test.reshape(-1, 28, 28, 1) / 255
y_pred = model.predict(test, batch_size = 64)

y_pred = np.argmax(y_pred,axis = 1)
y_pred = pd.Series(y_pred,name="Label")
y_pred
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("aksr.csv", index=False, header=True)
# model.save_weights(filepath='final_weight.h5')
model.save("model.h5")