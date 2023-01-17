import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pandas as pd
import glob
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/kaggle/input/solar-storm-recognition-dataset/project1/project1/trainimg/magnetogram',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(255, 255),
  batch_size=32)
class_names = train_ds.class_names
# print (class_names)
print(train_ds.class_names)
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image)) 
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/kaggle/input/solar-storm-recognition-dataset/project1/project1/testimg/magnetogram',
  seed=123,
  validation_split=0.2,
  subset="validation",
  image_size=(255, 255),
  batch_size=32)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
#     augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[0].numpy().astype("uint8"))
    plt.axis("off")
num_classes = 3
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(255, 255,3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)
model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(255, 255, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
def showEvaluationPlot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(15)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
showEvaluationPlot(history)
def imagePredict(path):
#     for image in test_images:
    img = keras.preprocessing.image.load_img(
    path, target_size=(255, 255))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
#         print(
#         "This image most likely belongs to {} with a {:.2f} percent confidence. --- The actual image is {} "
#         .format(class_names[np.argmax(score)], 100 * np.max(score), image['name']))
    return class_names[np.argmax(score)]
root = '/kaggle/input/solar-storm-recognition-dataset/project1/project1/testimg/magnetogram'
dirs = os.walk(root)
prediction_results = {
    'alpha' : {'result' :0, 'total' :0, 'tp':0, 'fn':0 ,'fp' :0 },
    'beta' : {'result' :0, 'total' :0, 'tp':0, 'fn':0 ,'fp' :0},
    'betax' : {'result' :0, 'total' :0, 'tp':0, 'fn':0 ,'fp' :0},  
    
}
def evaluateResult():
    for dr,f,g in dirs:
        for folder in f :
    #         print(root+'/'+folder)
    #         images = []
            images = os.walk(root+'/'+folder)
            c =0;
            for fol,g,image in images:
                for img in image:
                    if(c >10):
                        break;
                    predicted = imagePredict(root+'/'+folder+'/'+img)
                    if(predicted == folder):
                        prediction_results[folder]['result'] = prediction_results[folder]['result']+1
                        prediction_results[folder]['tp'] = prediction_results[folder]['tp']+1
                    else:
                        prediction_results[folder]['fp'] = prediction_results[folder]['fp']+1
                        prediction_results[predicted]['fn'] = prediction_results[folder]['fn']+1
                    prediction_results[folder]['total'] = prediction_results[folder]['total']+1
    print(prediction_results)
            
evaluateResult()
precision =0
recall =0
for pred in prediction_results:
    print(pred,' ---- ',prediction_results[pred]['result']/prediction_results[pred]['total']*100)
    precision = prediction_results[pred]['tp']/(prediction_results[pred]['tp']+prediction_results[pred]['fp'])
    recall = prediction_results[pred]['tp'] / (prediction_results[pred]['tp']+prediction_results[pred]['fn'])
print(precision)
print(recall)
f1_score = 2 * (precision*recall)/(precision+recall)
f1_score
# dirs = os.walk(root)
# data = {
#     'beta' :[],
#     'alpha' :[],
#     'betax' :[]
# }
# data_values = []
# data_labels = []
# for dr,f,g in dirs:
#     for folder in f:
#         subfold = root+'/'+folder
#         images = os.walk(subfold)
        
#         for fd,_,image in images:
#             print (folder,len(image))
#             for img in image:
#                 npImg = PIL.Image.open(subfold+'/'+img).convert('LA');
#                 data_values.append(np.asarray(npImg))
#                 data_labels.append(folder)
#                 data[folder].append(npImg)
# for label in data:
#     plt.figure(figsize=(10,10))
#     for i in range(3):
#         plt.subplot(3,3,i+1)
#         plt.imshow(data[label][i])
#         plt.title(label)
        
