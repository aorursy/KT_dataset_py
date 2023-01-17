import time
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.applications import ResNet50
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
train_dataset = "../input/skin-cancer-malignant-vs-benign/data/train"
test_dataset = "../input/skin-cancer-malignant-vs-benign/data/test"

print("Successful uploaded the ISIC dataset  ")
height, width = (224, 224)
epochs_no = 30 
classes_name = ('benign','malignant')
batch_size = 32
imgdatagen = ImageDataGenerator(rescale = 1./255,
                                     rotation_range=45, #10 
                                     zoom_range = 0.2, 
                                     width_shift_range=0.2,  
                                     height_shift_range=0.2) 
train_pre_dataset = imgdatagen .flow_from_directory(train_dataset, target_size = (height, width), 
                classes = classes_name,  batch_size = batch_size)

val_pre_dataset = imgdatagen .flow_from_directory(test_dataset, target_size = (height, width), 
                classes = classes_name, batch_size = batch_size)

print("Completed the preprosessing")
base_model= ResNet50(include_top=False, weights="imagenet", input_shape=(height, width,3))
ResNet_model= Sequential()
ResNet_model.add(base_model)
ResNet_model.add(Conv2D(64, (3, 3), activation = 'relu'))
ResNet_model.add(Dropout(0.2))
ResNet_model.add(Conv2D(64, (3, 3), activation = 'relu'))
ResNet_model.add(MaxPooling2D(pool_size = (2, 2)))
ResNet_model.add(Dropout(0.2))
ResNet_model.add(Flatten())
ResNet_model.add(Dense(512,activation='relu'))
ResNet_model.add(Dense(256,activation='relu'))
ResNet_model.add(Dropout(0.2))
ResNet_model.add(Dense(128,activation='relu'))
ResNet_model.add(Dense(64,activation='relu'))
ResNet_model.add(Dense(32,activation='relu'))
ResNet_model.add(Dense(16,activation='relu'))
ResNet_model.add(Dense(8,activation='relu'))
ResNet_model.add(Dense(2, activation='softmax'))
ResNet_model.summary()
ResNet_model.compile(optimizer=optimizers.Adam(lr=0.00001),loss="categorical_crossentropy",metrics=["accuracy"])
print("Training started it takes few minutes")
start_time = time.time()

learn_control = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=.5, min_lr=0.00001) #min_lr=0.0001
ResNet_history = ResNet_model.fit_generator(generator=train_pre_dataset,
                            steps_per_epoch=train_pre_dataset.samples//train_pre_dataset.batch_size,
                            validation_data=val_pre_dataset,
                            verbose=1,
                            validation_steps=val_pre_dataset.samples//val_pre_dataset.batch_size,
                            epochs=epochs_no,callbacks=[learn_control])
print("successfully completed the traing session")
print("--- took %d:%.2d minutes ---" % divmod(time.time() - start_time, 60))
# saving model to JSON file to save directory
model_json = ResNet_model.to_json()
with open("ResNet50_model_with.json", "w") as json_file:
    json_file.write(model_json)
    
# saving weights to HDF5
ResNet_model.save_weights("ResNet50_model_with.h5")
import matplotlib.pyplot as plt
# Plotting accuracy history
plt.figure(figsize=(7, 5))
plt.ylim(0.0, 1.1)
plt.plot(range(epochs_no), ResNet_history.history['accuracy'], color='blue', label='Training accuracy');
plt.plot(range(epochs_no), ResNet_history.history['val_accuracy'], color='r', label='Validation accuracy');
plt.legend();
plt.title('ResNet50 Accuracy with Data-augmentation');
plt.ylabel('Accuracy');
plt.xlabel('epoch');
plt.savefig('ResNet50_accuracy_with.jpg', dpi=300, bbox_inches='tight');

# Plotting loss history
plt.figure(figsize=(7, 5))
plt.ylim(0.0, 2)
plt.plot(range(epochs_no), ResNet_history.history['loss'], color='blue', label='Training loss');
plt.plot(range(epochs_no), ResNet_history.history['val_loss'], color='r', label='Validation loss');
plt.legend();
plt.title('ResNet50 Loss with Data-augmentation');
plt.ylabel('Loss');
plt.xlabel('epoch');
plt.savefig('ResNet50_loss_with.jpg', dpi=300, bbox_inches='tight')