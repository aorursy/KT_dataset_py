from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D,BatchNormalization
from keras.layers import Dense,Dropout
from keras.layers import Flatten
def create_model():
    def add_conv(model, num_filters):
        
        #Adding convolution and MaxPooling
        model.add(Convolution2D(filters=num_filters, kernel_size=(3,3), input_shape=(224,224,3),activation='relu',padding ='same' ))
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=num_filters, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.5))
        return model
    
    #Initializing the CNN
    model = Sequential()
    
    
    model = add_conv(model, 32)
    model = add_conv(model , 64)
    model = add_conv(model , 128)
    
    #Flattening
    model.add(Flatten())
    
    #Full connection
    model.add(Dense(units=128,activation='relu'))
    model.add(Dense(units=38,activation ='softmax'))
    
    #Compiling the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam', metrics=['accuracy']
    )
    return model
model =create_model()

model.summary()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_data_path = '../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
valid_data_path = '../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(  
    train_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(
    valid_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')
for i, j in train_generator:
    print(i.shape)
    print(j.shape)
    break
# Sample image 
import matplotlib.pyplot as plt
plt.imshow(i[0])
plt.show()
print(j[0])
#Fitting image to CNN
history = model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=25,
                    validation_data=valid_generator,
                    validation_steps=len(valid_generator)
                   )
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure()
#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
