import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec

from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
def get_images(directory):
    Images = []
    Labels = []  # [buildings, forest, glacier, mountain, sea, street]
    label = 0
    
    for labels in os.listdir(directory):
        if labels == 'glacier':
            label = 2
        elif labels == 'sea':
            label = 4
        elif labels == 'buildings':
            label = 0
        elif labels == 'forest':
            label = 1
        elif labels == 'street':
            label = 5
        elif labels == 'mountain':
            label = 3
        
        for image_file in os.listdir(directory + labels): # Extracting the file name of the image from Class Label folder
            image = cv2.imread(directory+labels+r'/'+image_file) # Reading the image (OpenCV)
            image = cv2.resize(image,(150, 150)) # Resize the image, Some images are different sizes. (Resizing is very Important)
            Images.append(image)
            Labels.append(label)
    
    return shuffle(Images,Labels,random_state=817328462) # Shuffle the dataset you just prepared.

def get_classlabel(class_code):
    labels = {2: 'glacier', 4: 'sea', 0: 'buildings', 1: 'forest', 5: 'street', 3: 'mountain'}
    
    return labels[class_code]

train_dir = "../input/seg_train/seg_train/"
valid_dir = "../input/seg_test/seg_test/"

Images, Labels = get_images(train_dir) # Extract the training images from the folders

Images = np.array(Images) # Converting the list of images to numpy array
Labels = np.array(Labels)

print("Shape of Images:", Images.shape)
print("Shape of Labels:", Labels.shape)
f,ax = plot.subplots(5, 5) 
f.subplots_adjust(0, 0, 3, 3)
for i in range(0, 5, 1):
    for j in range(0, 5, 1):
        rnd_number = randint(0, len(Images))
        ax[i,j].imshow(Images[rnd_number])
        ax[i,j].set_title(get_classlabel(Labels[rnd_number]))
        ax[i,j].axis('off')
img_width, img_height = 224, 224

conv_base = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=(img_width, img_height, 3))

# Show architecture
conv_base.summary()
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count,6))
    # Preprocess data
    generator = datagen.flow_from_directory(directory,
                                            target_size=(img_width,img_height),
                                            batch_size = batch_size,
                                            class_mode='categorical')
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 14034)  # Agree with our small dataset size
validation_features, validation_labels = extract_features(valid_dir, 3000)
# test_features, test_labels = extract_features(test_dir, test_size)
epochs2 = 5

model2 = Models.Sequential()
model2.add(Layers.GlobalAveragePooling2D(input_shape=(7,7,512)))
model2.add(Layers.Dense(6, activation='softmax'))
model2.summary()
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  

# Compile model
from keras.optimizers import Adam
model2.compile(optimizer=Optimizer.Adam(),
              loss='categorical_crossentropy',
              metrics=['acc'])

# Train model
history = model2.fit(train_features, train_labels,
                    epochs=epochs2,
                    batch_size=batch_size, 
                    callbacks=[checkpoint],
                    validation_data=(validation_features, validation_labels))
# Plot results
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
model = Models.Sequential()

model.add(Layers.Conv2D(200, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(Layers.Conv2D(180, kernel_size=(3, 3), activation='relu'))
model.add(Layers.MaxPool2D(5, 5))
model.add(Layers.Conv2D(180, kernel_size=(3, 3), activation='relu'))
model.add(Layers.Conv2D(140, kernel_size=(3, 3), activation='relu'))
model.add(Layers.Conv2D(100, kernel_size=(3, 3), activation='relu'))
model.add(Layers.Conv2D(50, kernel_size=(3, 3), activation='relu'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Flatten())
model.add(Layers.Dense(180, activation='relu'))
model.add(Layers.Dense(100, activation='relu'))
model.add(Layers.Dense(50, activation='relu'))
model.add(Layers.Dropout(rate=0.5))
model.add(Layers.Dense(6, activation='softmax'))

model.compile(optimizer=Optimizer.Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
SVG(model_to_dot(model).create(prog='dot', format='svg'))
Utils.plot_model(model,to_file='model.png',show_shapes=True)
trained = model.fit(Images, Labels, epochs=20, validation_split=0.30)
plot.plot(trained.history['acc'])
plot.plot(trained.history['val_acc'])
plot.title('Model Accuracy')
plot.ylabel('Accuracy')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'])
plot.show()

plot.plot(trained.history['loss'])
plot.plot(trained.history['val_loss'])
plot.title('Model Loss')
plot.ylabel('Loss')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'])
plot.show()
test_images,test_labels = get_images('../input/seg_test/seg_test/')
test_images = np.array(test_images)
test_labels = np.array(test_labels)
model.evaluate(test_images,test_labels, verbose=1)
print(len(test_images))
pred_images,no_labels = get_images('../input/seg_pred/')
pred_images = np.array(pred_images)
pred_images.shape
print(len(pred_images))
fig = plot.figure(figsize=(30, 30))
outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)

for i in range(25):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.1)
    rnd_number = randint(0,len(pred_images))
    pred_image = np.array([pred_images[rnd_number]])
    pred_class = get_classlabel(model.predict_classes(pred_image)[0])
    pred_prob = model.predict(pred_image).reshape(6)
    for j in range(2):
        if (j%2) == 0:
            ax = plot.Subplot(fig, inner[j])
            ax.imshow(pred_image[0])
            ax.set_title(pred_class)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
        else:
            ax = plot.Subplot(fig, inner[j])
            ax.bar([0,1,2,3,4,5], pred_prob)
            fig.add_subplot(ax)


fig.show()