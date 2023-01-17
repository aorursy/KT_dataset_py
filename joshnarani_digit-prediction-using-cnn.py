import pandas as pd

import numpy as np
test=pd.read_csv('../input/test.csv')
train=pd.read_csv('../input/train.csv')
Y_train = train["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 

# Check the missing data

X_train.isnull().any().describe()
test.isnull().any().describe()
# Normalize the data

X_train = X_train / 255.0

test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
from tensorflow import keras as tfk
Y_train = tfk.utils.to_categorical(Y_train) 
from sklearn.model_selection import train_test_split
X_train,X_val,Y_train,Y_val=train_test_split(X_train, Y_train, test_size = 0.1, random_state=0)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
tb=tfk.callbacks.TensorBoard()


# CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out



model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.compile(optimizer='RMSprop', loss=tfk.losses.categorical_crossentropy, metrics=["acc"])
model_history = model.fit(X_train, Y_train, batch_size=600, epochs=5, validation_split=0.2, callbacks=[tb])
model.summary()
model.evaluate(X_val, Y_val, batch_size=600)
y_test_model = model.predict(X_val, batch_size=600)
y_test_model[0]
import matplotlib.pyplot as plt

%matplotlib inline
image_index = 4144

plt.imshow(X_val[image_index].reshape(28, 28),cmap='Greys')

pred = model.predict(X_val[image_index].reshape(1, 28, 28, 1))

print(pred.argmax())
y_test_model = np.argmax(y_test_model, axis=1)
y_test_original = np.argmax(Y_val, axis=1)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test_model,y_test_original)
import seaborn as sns
plt.figure(figsize=(20, 20))

sns.heatmap(cm,annot=True,square=True,cmap="Reds")

plt.show()
from sklearn.metrics import classification_report
classification_report(y_test_model,y_test_original)
model_history.history.keys()
model_history.params
plt.plot(model_history.history["val_acc"], label="Validation Acc")

plt.plot(model_history.history["acc"], label="Training Accuracy")

plt.legend()
plt.plot(model_history.history.get("loss") ,label="Losses")

plt.plot(model_history.history.get("val_loss"), label="Validation Loss")

plt.legend()
results=model.predict(test)

results=np.argmax(results,axis=1)

results = pd.Series(results,name="Label")

submission = pd.DataFrame([pd.Series(range(1,28001),name = "ImageId"),results])

submission.to_csv('submission.csv',header=True,index=False)
submission.shape
submission.head()