import numpy as np 
import pandas as pd

import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.applications.vgg16 import VGG16
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.utils.vis_utils import plot_model
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
train_data = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
train_data.diagnosis.value_counts().plot(kind="bar")
train_data.head()

TRAIN_DATA_DIR = "/kaggle/input/aptos2019-blindness-detection/train_images/"
data =[]

def read_image_convert_to_array(filepath):
    image = load_img(TRAIN_DATA_DIR+filepath+".png", target_size=(224,224))
    image = img_to_array(image)
    image /= 255.0
    return image

train_data["img_data"] = train_data["id_code"].apply(lambda x: read_image_convert_to_array(x))
X = train_data["img_data"]
y = train_data["diagnosis"]
X = np.stack(X)
le = LabelEncoder()

y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)    
    
# Generate the trained model and set all layers to be trainable
trained_model = VGG16(input_shape=(224,224,3), include_top=False)

for layer in trained_model.layers:
    layer.trainable = True

# Construct the model and compile
mod1 = Flatten()
mod_final = Dense(5, activation='softmax')

model = Sequential([trained_model, mod1, mod_final])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
plot_model(model, to_file='vgg.png')
# Fit the model to the data and validate
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=15)

# Plot the model results using seaborn and matplotlib
sns.set(style='darkgrid')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
pred_Y = model.predict(X_test, batch_size = 32, verbose = True)
pred_Y_cat = np.argmax(pred_Y, -1)
test_Y_cat = np.argmax(y_test, -1)
print('Accuracy on Test Data: %2.2f%%' % (100*accuracy_score(test_Y_cat, pred_Y_cat)))
print(classification_report(test_Y_cat, pred_Y_cat))
sns.heatmap(confusion_matrix(test_Y_cat, pred_Y_cat), 
annot=True, fmt="d", cbar = False, cmap = plt.cm.Blues, vmax = X_test.shape[0]//16)
sick_vec = test_Y_cat>0
sick_score = np.sum(pred_Y[:,1:],1)
fpr, tpr, _ = roc_curve(sick_vec, sick_score)
fig, ax1 = plt.subplots(1,1, figsize = (5, 3), dpi = 150)
ax1.plot(fpr, tpr, 'b.-', label = 'Model Prediction (AUC: %2.2f)' % roc_auc_score(sick_vec, sick_score))
ax1.plot(fpr, fpr, 'g-', label = 'Random Guessing')
ax1.legend()
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate');
submission_csv = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/sample_submission.csv")
TEST_DATA_DIR = "/kaggle/input/aptos2019-blindness-detection/test_images/"
data =[]

def read_image_convert_to_array(filepath):
    image = load_img(TEST_DATA_DIR+filepath+".png", target_size=(224,224))
    image = img_to_array(image)
    image /= 255.0
    return image

submission_csv["img_data"] = submission_csv["id_code"].apply(lambda x: read_image_convert_to_array(x))
test_data = np.stack(submission_csv["img_data"])
pred_test_data = model.predict(test_data,verbose = True)
pred_test_data_category = np.argmax(pred_test_data, -1)
submission_csv["diagnose"] = np.array(pred_test_data_category)
submission_csv = submission_csv[["id_code", "diagnose"]]
submission_csv.to_csv("output.csv")
