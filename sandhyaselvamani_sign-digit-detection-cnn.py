import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
x = np.load('input/Sign-language-digits-dataset/X.npy')
y = np.load('input/Sign-language-digits-dataset/Y.npy')
img_size = 64
plt.subplot(1,2,1)
plt.imshow(x[1500].reshape(img_size,img_size))
plt.axis('off')
plt.subplot(1,2,2)
print(x[250].reshape(img_size,img_size).shape)
plt.imshow(x[250].reshape(img_size,img_size))
plt.axis('off')
# As you can see, y (labels) are already one hot encoded
print(y.max())
print(y.min())
print(x[300])
print(y[300])
print(x[400])
print(y[400])

# And x (features) are already scaled between 0 and 1
print(x.max())
print(x.min())
print(x.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
#reshape
x_train = x_train.reshape(-1,64,64,1)
x_test = x_test.reshape(-1,64,64,1)
#print x_train and y_train shape
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
#Data Augumentation to increase the number of parameters to avoid underfitting
#datagen = ImageDataGenerator(
#    rotation_range=16,
#   width_shift_range=0.12,
# height_shift_range=0.12,
# zoom_range=0.12 )
#datagen.fit(x_train)
#print(x_train.shape)
from sklearn.metrics import confusion_matrix
import itertools
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential # to create a cnn model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,BatchNormalization
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()

model.add(Conv2D(filters = 8, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (64,64,1)))
model.add(MaxPool2D(pool_size = (2,2)))
#dropout To reduce overfitting model when Validation loss > Training loss
model.add(Dropout(0.25))



model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))
#model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))
#model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))
#model.add(BatchNormalization())


# fully connected
model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()
optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)

#Training will stop when the chosen performance measure stops improving after 15 epochs
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

#we are interested in only the very best model observed during training, rather than the best compared to the previous epoch, which might not be the best overall if training is noisy. 
#Ensure to save best model in best_model.h5
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train,y_train,epochs=50,validation_data=(x_test,y_test),callbacks=[es,mc])
# Saved model which contains val_loss,val_acc,train_loss,train_acc for each iterations
history.history
#Plot Training Loss and Validation Loss
# Training Loss < Validation Loss=Overfitting
# Training Loss < Validation Loss=Underfitting
# Training Loss ~= Validation Loss=Perfect

plt.figure(figsize=(24,8))

plt.subplot(1,2,1)
plt.plot(history.history["val_accuracy"], label="validation_accuracy", c="red", linewidth=4)
plt.plot(history.history["accuracy"], label="training_accuracy", c="green", linewidth=4)
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history["val_loss"], label="validation_loss", c="red", linewidth=4)
plt.plot(history.history["loss"], label="training_loss", c="green", linewidth=4)
plt.legend()
plt.grid(True)

plt.suptitle("ACC / LOSS",fontsize=18)

plt.show()

test_image = x_test[100]
test_image_array = test_image.reshape(64, 64)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

plt.imshow(test_image_array, cmap='gray')
print(np.round(result, 2))
print(y_test[100])
from keras.models import model_from_json
from keras.models import load_model

# serialize model to JSON
#  the keras model which is trained is defined as 'model' in this example
json_model = model.to_json()
#save the model architecture to JSON file
with open('sign_model.json', 'w') as json_file:
    json_file.write(json_model)
#saving the weights of the model
model.save_weights('sign_weights.h5')
#Model loss and accuracy
#loss,acc = model.evaluate(test_images,  test_labels, verbose=2)

with open('sign_model.json', 'r') as json_file:
    json_savedModel= json_file.read()
    model_j=model_from_json(json_savedModel)
#load the model architecture 
model_j.load_weights('sign_weights.h5')
model_j.summary()
#Reuse the same model or train with different dataset and optimization techniques
           

#Deserialize the model and fit the same model in different dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 30)
#reshape
x_train = x_train.reshape(-1,64,64,1)
x_test = x_test.reshape(-1,64,64,1)
#optimizer=optimizers.RMSprop(lr=0.0001)
optimizer = RMSprop(lr=0.0001)
model_j.compile(loss="categorical_crossentropy",
                             optimizer=optimizer,
                             metrics=["accuracy"])
history = model_j.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))


train_scores = model_j.evaluate(x_train, y_train, verbose=0)
test_scores  = model_j.evaluate(x_test, y_test, verbose=0)
print("Train accuracy:{:.3f}".format(train_scores[1]))
print("Test accuracy:{:.3f}".format(test_scores[1]))


        
       
import h5py 
with h5py.File('test.hdf5', 'r') as f:  
    data = f['default'] 
      
    # get the minimum value 
    print(min(data))  
      
    # get the maximum value 
    print(max(data)) 
      
    # get the values ranging from index 0 to 15 
    print(data[:15]) 