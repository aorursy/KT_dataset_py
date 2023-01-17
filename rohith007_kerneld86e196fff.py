from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.preprocessing import image
import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D,Dropout,Input
# from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.models import Sequential,Model
from keras import backend as K
from IPython.display import display
base_model  = InceptionV3(weights = 'imagenet', include_top=False)
print('loaded model')
data_gen_args = dict(preprocessing_function=preprocess_input, #Define the dictionary for Image data Generator
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip = True)
#its just for image augmentation to increase the dataset size
#take of the original model 
train_datagen = image.ImageDataGenerator(**data_gen_args)
test_datagen = image.ImageDataGenerator(**data_gen_args)
train_generator = train_datagen.flow_from_directory("../input/Skin_Cancer_Capstone_Project.zip/Capstone Project/Skin_Cancer_Capstone_Project/Train",
                                                    target_size=(299,299),batch_size=32)

valid_generator = test_datagen.flow_from_directory("../input/Skin_Cancer_Capstone_Project.zip/Capstone Project/Skin_Cancer_Capstone_Project/Train - Copy",
                                                     target_size=(299,299),batch_size=32)
from keras.layers import Conv2D,MaxPooling2D,Flatten

benchmark = Sequential()
benchmark.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation = 'relu', input_shape = (299,299,3)))
benchmark.add(MaxPooling2D(pool_size=2,padding='same'))
benchmark.add(Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
benchmark.add(MaxPooling2D(pool_size=2,padding='same'))
benchmark.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
benchmark.add(MaxPooling2D(pool_size=2,padding='same'))
benchmark.add(Dropout(0.3))
benchmark.add(Flatten())
benchmark.add(Dense(512, activation='relu'))
benchmark.add(Dropout(0.5))
benchmark.add(Dense(3, activation='softmax'))

benchmark.summary()
benchmark.compile(loss = 'categorical_crossentropy',optimizer='rmsprop', metrics = ['accuracy'])

from  keras.callbacks  import ModelCheckpoint,EarlyStopping

# Save the model with best weights
checkpointer = ModelCheckpoint('C:/Users/Rohith007/Downloads/ISB_CBA/Capstone Project/Skin_Cancer_Capstone_Project/benchmark.hdf5', verbose=1,save_best_only=True)
# Stop the training if the model shows no improvement 
stopper = EarlyStopping(monitor='val_loss',min_delta=0.1,patience=0,verbose=1,mode='auto')
history = benchmark.fit_generator(train_generator, steps_per_epoch = 5,validation_data=valid_generator,validation_steps=3, epochs=1,verbose=1,callbacks=[checkpointer])
# Define the output layers for Inceptionv3
last = base_model.output
x = GlobalAveragePooling2D()(last)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(3,activation='softmax')(x)

model = Model(input=base_model.input,output=preds)
model.summary()
#Load the weights for the common layers from the benchmark model
base_model.load_weights(filepath='C:/Users/Rohith007/Downloads/ISB_CBA/Capstone Project/Skin_Cancer_Capstone_Project/benchmark.hdf5',by_name=True)
#Freeze the original layers of Inception3
for layer in base_model.layers:
    layer.trainable = False
#Compile the model
# Multiclass classification
model.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint,EarlyStopping

# Save the model with best weights
checkpointer = ModelCheckpoint('C:/Users/Rohith007/Downloads/ISB_CBA/Capstone Project/Skin_Cancer_Capstone_Project/transfer_learning.hdf5', verbose=1,save_best_only=True)
# Stop the traning if the model shows no improvement
stopper = EarlyStopping(monitor='val_loss',min_delta=0.1,patience=1,verbose=1,mode='auto')

# Train the model# Train  
history_transfer = model.fit_generator(train_generator, steps_per_epoch = 5,validation_data=valid_generator,validation_steps=3, epochs=1,verbose=1,callbacks=[checkpointer])
display(history_transfer.history)

# Unfreeze the last three inception modules# Unfree 
for layer in model.layers[:229]:
    layer.trainable = False
for layer in model.layers[229:]:
    layer.trainable = True
from keras.optimizers import SGD
#stochastic gradient descent 
# adam optimizer
# Use an optimizer with slow learning rate
model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),loss = 'categorical_crossentropy', metrics = ['accuracy'])
#Save the model with best validation loss
checkpointer = ModelCheckpoint('C:/Users/Rohith007/Downloads/ISB_CBA/Capstone Project/Skin_Cancer_Capstone_Project/fine_tuning.hdf5.hdf5', verbose=1,save_best_only=True,monitor='val_loss')

# Stop the traning if the validation loss doesn't improve
stopper = EarlyStopping(monitor='val_loss,val_acc',min_delta=0.1,patience=2,verbose=1,mode='auto')

# Train the model# Train  
history = model.fit_generator(train_generator, steps_per_epoch = 5,validation_data=valid_generator,validation_steps=3, epochs=1,verbose=1,callbacks=[checkpointer])
# Load the weights fromt the fine-tuned model
model.load_weights('C:/Users/Rohith007/Downloads/ISB_CBA/Capstone Project/Skin_Cancer_Capstone_Project/fine_tuning.hdf5.hdf5')
from keras.preprocessing.image import img_to_array,load_img
import matplotlib.pyplot as plt
#import cv2
%matplotlib inline
def pred(img_path):    
    img = load_img(img_path,target_size = (299,299)) #Load the image and set the target size to the size of input of our model
    x = img_to_array(img) #Convert the image to array
    x = np.expand_dims(x,axis=0) #Convert the array to the form (1,x,y,z) 
    x = preprocess_input(x) # Use the preprocess input function o subtract the mean of all the images
    p = np.argmax(model.predict(x)) # Store the argmax of the predictions
    if p==0:     # P=0 for basal,P=1 for melanoma , P=2 for squamous
        print("melonamo")
    elif p==1:
        print("nevus")
    elif p==2:
        print("seborrheic_keratosis")
pred("C:/Users/Rohith007/Downloads/ISB_CBA/Capstone Project/Skin_Cancer_Capstone_Project/Capstone/melanoma/ISIC_0000029.jpg")
z = plt.imread('C:/Users/Rohith007/Downloads/ISB_CBA/Capstone Project/Skin_Cancer_Capstone_Project/Capstone/melanoma/ISIC_0000029.jpg') 
plt.imshow(z);         #print the loaded image 
pred("C:/Users/Rohith007/Downloads/ISB_CBA/Capstone Project/Skin_Cancer_Capstone_Project/Capstone/nevus/ISIC_0000017.jpg")
z = plt.imread('C:/Users/Rohith007/Downloads/ISB_CBA/Capstone Project/Skin_Cancer_Capstone_Project/Capstone/nevus/ISIC_0000017.jpg') 
plt.imshow(z);         #print the loaded image


pred("C:/Users/Rohith007/Downloads/ISB_CBA/Capstone Project/Skin_Cancer_Capstone_Project/Capstone/seborrheic_keratosis/ISIC_0012090.jpg")
z = plt.imread('C:/Users/Rohith007/Downloads/ISB_CBA/Capstone Project/Skin_Cancer_Capstone_Project/Capstone/seborrheic_keratosis/ISIC_0012090.jpg') 
plt.imshow(z);         #print the loaded image

