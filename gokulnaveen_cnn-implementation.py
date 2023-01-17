import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten,Dropout
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


model=Sequential()

model.add(Conv2D(128,(3,3),input_shape=(128,128,3)))
model.add(Activation('relu'))
model.add(Conv2D(128,(3,3),input_shape=(128,128,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),input_shape=(128,128,3)))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3),input_shape=(128,128,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1024,activation='relu'))


model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

aug = ImageDataGenerator( rescale=1./255,rotation_range=180,zoom_range=[0.9,1.2],width_shift_range=0.1,height_shift_range=0.1,fill_mode="nearest")

train_generator = aug.flow_from_directory(
        '../input/train-dataset/project_db',
        target_size=(128, 128),
  
    color_mode = 'rgb',
        batch_size=16,
        class_mode='categorical',shuffle=True)
aug1 = ImageDataGenerator( rescale=1./255)

test_generator = aug1.flow_from_directory(
        '../input/dataset/test',
        target_size=(128, 128),
  
    color_mode = 'rgb',
        batch_size=8,
        class_mode='categorical',shuffle=False)
history=model.fit_generator(
        generator=train_generator,validation_data=test_generator,steps_per_epoch=5443 // 16,epochs=100,validation_steps=1000 // 8)
predict1=model.predict_generator(test_generator,1000)
y_pred = np.argmax(predict1, axis=1)
targets=test_generator.class_indices
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
cf=confusion_matrix(test_generator.classes, y_pred)
print(cf)
print(targets)
print("accuracy : ",accuracy_score(test_generator.classes, y_pred)*100)
import seaborn as sns
sns.heatmap(cf, annot=True)
import sklearn
print("Kappa coefficient : ",sklearn.metrics.cohen_kappa_score(test_generator.classes, y_pred))
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

