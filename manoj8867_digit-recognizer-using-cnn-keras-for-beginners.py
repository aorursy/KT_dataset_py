import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D, MaxPool2D,Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.activations import relu,sigmoid,softmax
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau,EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

%matplotlib inline

print("all libraries are imported")
train=pd.read_csv('../input/digit-recognizer/train.csv')
test=pd.read_csv('../input/digit-recognizer/test.csv')
print("datasets are imported")
train.head(2)
X=train.drop(columns='label',axis=1)
y=train['label']

sns.countplot(y)
X.isnull().any().describe()

test.isnull().any().describe()

#normalization
X=X/255
test=test/255

#reshape
X=X.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)

#one hot encoding
y=to_categorical(y,10)

#train val split
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.1,random_state=101)
X_train.shape
X_val.shape
plt.imshow(X_train[0][:,:,0],cmap='gray')
early_stop=EarlyStopping(monitor='val_loss',patience=2)
model=Sequential()

model.add(Conv2D(filters = 32, kernel_size = (4,4), 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
model.fit(X_train,y_train,batch_size=128,
          epochs=20,validation_data=(X_val,y_val),
          callbacks=[early_stop],verbose=1)
model_metrics=pd.DataFrame(model.history.history)
model_metrics.head(2)
model_metrics[["loss","val_loss"]].plot(title="loss curve")
plt.show()
model_metrics[["accuracy","val_accuracy"]].plot(title="accuracy curve")
plt.show()
print(model.metrics_names)
print(model.evaluate(X_val,y_val,verbose=0))
prediction=model.predict_classes(X_val)
y_val=np.argmax(y_val,axis=1)
print(classification_report(y_val,prediction))
sns.heatmap(confusion_matrix(y_val,prediction),annot=True,cmap='Blues', fmt='g')

lr_decay = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=1, 
                                            verbose=1, 
                                            factor=0.4, 
                                            min_lr=0.00001)

aug_gen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.1,  
        height_shift_range=0.1, 
        horizontal_flip=False, 
        vertical_flip=False) 


aug_gen.fit(X_train)
model=Sequential()

model.add(Conv2D(filters = 32, kernel_size = (4,4), 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
y_val=to_categorical(y_val,10)
y_val
model.fit_generator(aug_gen.flow(X_train,y_train, batch_size=128),
                              epochs = 20, validation_data = (X_val,y_val),
                              verbose = 2
                              , callbacks=[lr_decay,early_stop])
model_metrics=pd.DataFrame(model.history.history)

model_metrics[["loss","val_loss"]].plot(title="loss curve")
plt.show()
model_metrics[["accuracy","val_accuracy"]].plot(title="accuracy curve")
plt.show()

print(model.metrics_names)
print(model.evaluate(X_val,y_val,verbose=0))
prediction=model.predict_classes(X_val)

y_val=np.argmax(y_val,axis=1)

print(classification_report(y_val,prediction))
sns.heatmap(confusion_matrix(y_val,prediction),annot=True,cmap='Blues', fmt='g')
results_pred=model.predict_classes(test)
im_id = np.arange(1,28001)
Result = pd.DataFrame({"ImageId": im_id,"Label": results_pred})
Result.to_csv('mnist_prediction_001.csv', index=False)