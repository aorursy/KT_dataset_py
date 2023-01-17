import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
from tensorflow import keras
from PIL import Image
import re
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras import backend as K 
from keras.layers import *
from keras.optimizers import Adam, Nadam, RMSprop
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical

CSV_length = 4096;
N_classes = 29;
N_train_samples = 1000
N_validation_samples = 100
N_epochs = 50


trainingLabels = pd.read_csv('../input/au-eng-cvml2020/Train/trainLbls.csv', header=None);
trainingVectors = pd.read_csv('../input/au-eng-cvml2020/Train/trainVectors.csv', header=None).T;


validationLabels = pd.read_csv('../input/au-eng-cvml2020/Validation/valLbls.csv', header=None);
validationVectors = pd.read_csv('../input/au-eng-cvml2020/Validation/valVectors.csv', header=None).T;

testVectors = pd.read_csv('../input/au-eng-cvml2020/Test/testVectors.csv', header=None).T;

vectorsPreprocessed = False;


if vectorsPreprocessed == False:
  trainingVectors = trainingVectors.to_numpy()
  trainingVectors = trainingVectors/np.amax(trainingVectors)
  trainingLabels = trainingLabels.to_numpy()-1

  validationVectors = validationVectors.to_numpy()
  validationVectors = validationVectors/np.amax(validationVectors)
  validationLabels = validationLabels.to_numpy()-1

  testVectors = testVectors.to_numpy()
  testVectors = testVectors/np.amax(testVectors)
  
  vectorsPreprocessed = True;
print(trainingVectors.shape)
model = keras.Sequential([
    keras.layers.Dense(64, activation ='relu', input_dim = CSV_length),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(N_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
from keras.callbacks import *
stoppingCriterion = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=10, verbose=1, mode='max', baseline=None, restore_best_weights=True);
history_1=model.fit(trainingVectors, trainingLabels, validation_data = [validationVectors, validationLabels], epochs=100, callbacks = [stoppingCriterion])
import pickle as p
# Save model
model.save('model_vectors.h5')

print(history_1.history)

# save history object
with open('./trainHistory_vectors', 'wb') as file_pi:
        p.dump(history_1.history, file_pi)
# --- PLOT TRAINING DIAGRAM ---
plt.figure()
plt.plot(history_1.history['accuracy']);
plt.plot(history_1.history['val_accuracy']);
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.show()
test_loss, test_acc = model.evaluate(validationVectors,  validationLabels, verbose=2)

print('\nValidation accuracy:', test_acc)
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(testVectors)
predictions = np.argmax(predictions, axis=1)
predictions = predictions.astype(int)+1
print(predictions)
print(min(predictions))
print(max(predictions))
CSV = predictions
CSV = np.array([range(1,len(predictions)+1),predictions]).transpose()
CSV

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data_to_submit = pd.DataFrame({
    'ID':range(1,len(predictions)+1),
    'Label':predictions
})

data_to_submit.to_csv('Results_raw_vectors.csv', index = False)