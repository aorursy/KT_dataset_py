import keras
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
# Create a Image Data Generator object 
train_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/train/'
val_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/val/'
test_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/test'
image_input_shape = (256, 256)
image_dims = (256, 256, 3)
train_batch_size = 32

data_augmentor = ImageDataGenerator(
                    samplewise_center=True,
                    samplewise_std_normalization=True,
                    )

train_generator = data_augmentor.flow_from_directory(train_dir, target_size = image_input_shape, batch_size= train_batch_size, shuffle = True)
val_generator = data_augmentor.flow_from_directory(val_dir, target_size = image_input_shape, batch_size= train_batch_size, shuffle = True)
test_generator = data_augmentor.flow_from_directory(test_dir, target_size = image_input_shape, batch_size= train_batch_size, shuffle = True)
def MyModel(data_input):
    X_input = Input(data_input)

    #First Layer
    X = Conv2D(64, kernel_size=(5, 5), strides=1, padding="SAME", activation='relu', name='conv0')(X_input)
    X = MaxPool2D(pool_size=(2, 2), strides=2, name='max-pool0')(X)

    #Second Layer
    X = Conv2D(128, kernel_size=(3, 3), strides=1, padding="SAME", activation='relu', name='conv1')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=2, name='max-pool1')(X)
    
    #Thrid Layer
    X = Conv2D(128, kernel_size=(3, 3), strides=1, padding="VALID", name='conv2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=2, name='max-pool2')(X)
    
    #Fourth Layer
    X = Conv2D(128, kernel_size=(3, 3), strides=1, padding="VALID", name='conv3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=2, name='max-pool3')(X)
    
    #Flatten
    X = Flatten()(X)

    # Fully connected Layer
    X = Dense(units=128, activation='relu', name='dense1')(X)
#     X = Dropout(0.20)(X)
    X = Dense(units=32, activation='relu', name='dense2')(X)
#     X = Dropout(0.20)(X)
    X = Dense(units=2, activation='softmax', name='output')(X)

    model = Model(inputs=X_input, outputs=X, name='pneumoniaClassifier')

    return model
#Build the model
model = MyModel(image_dims)

model.summary()
#Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',])
history = model.fit_generator(train_generator, epochs=4, validation_data=val_generator)
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'svg'
# Plot Model's Train v/s Validation Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Acuracy")
plt.legend(['Train', 'Validation'])
plt.show()
# Plot Model's Train v/s Validation Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Train', 'Validation'])
plt.show()
#Evaluate the model's perfomance
performance = model.evaluate_generator(test_generator)
print("Loss on Test Set: %.2f" % (performance[0]))
print("Accuracy on Test Set: %.2f" % (performance[1]*100))