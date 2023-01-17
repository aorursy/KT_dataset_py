import pandas as pd

file = '/kaggle/input/covid-ctscans/metadata.csv'
df = pd.read_csv(file)
p_id = 0
cov_fn = []
allfiles = []

for (i, row) in df.iterrows():
    n_id = row["patientid"]
    
    if n_id != p_id and len(cov_fn)>0 and p_id != 0:
        allfiles.append(cov_fn)
        cov_fn = []
    if row["finding"] == "COVID-19" and row["view"] == "PA":
        cov_fn.append(row["filename"])
    
    p_id = row["patientid"]
from sklearn.model_selection import train_test_split
x_train_c, x_test_c = train_test_split(allfiles, test_size=0.20, random_state=23)
x_test_c_imgs, x_train_c_imgs = [], []

for img in sum(x_test_c, []):
    x_test_c_imgs.append('/kaggle/input/covid-ctscans/images/' + img)

for img in sum(x_train_c, []):
    x_train_c_imgs.append('/kaggle/input/covid-ctscans/images/' + img)
import os
import random

n_samples = 201
normal_xrays = '/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/'
x_test_n_imgs, x_train_n_imgs = [], []

filenames = os.listdir(normal_xrays)
random.seed(42)
filenames = random.sample(filenames, n_samples)
for i in range(n_samples):
    if i < 165:
        x_train_n_imgs.append(normal_xrays + filenames[i])
    else:
        x_test_n_imgs.append(normal_xrays + filenames[i])
import numpy as np
import cv2

x_train_n, x_train_c, x_test_n, x_test_c = [], [], [], []

for p in x_train_n_imgs:
    x_train_n.append(cv2.imread(p))
    
for p in x_test_n_imgs:
    x_test_n.append(cv2.imread(p))
    
for p in x_train_c_imgs:
    x_train_c.append(cv2.imread(p))
    
for p in x_test_c_imgs:
    x_test_c.append(cv2.imread(p))
print('Totally {} covid train, {} covid test, {} normal train and {} normal test cases ({} covid, {} normal).'.format(len(x_train_c),len(x_test_c),len(x_train_n),len(x_test_n),len(x_train_c)+len(x_test_c),len(x_train_n),+len(x_test_n)))
for i in range(len(x_train_n)):
    x_train_n[i] = cv2.resize(x_train_n[i], (224, 224))
for i in range(len(x_train_c)):
    x_train_c[i] = cv2.resize(x_train_c[i], (224, 224))
for i in range(len(x_test_n)):
    x_test_n[i] = cv2.resize(x_test_n[i], (224, 224))
for i in range(len(x_test_c)):
    x_test_c[i] = cv2.resize(x_test_c[i], (224, 224))
x_train_n = np.array(x_train_n)
x_train_c = np.array(x_train_c)
x_test_n = np.array(x_test_n)
x_test_c = np.array(x_test_c)
x_train = np.concatenate((x_train_n, x_train_c))/255.0
x_test = np.concatenate((x_test_n, x_test_c))/255.0
y_train_n = np.zeros(x_train_n.shape[0])
y_train_c = np.ones(x_train_c.shape[0])
y_test_n = np.zeros(x_test_n.shape[0])
y_test_c = np.ones(x_test_c.shape[0])
y_train = np.concatenate((y_train_n, y_train_c))
y_test = np.concatenate((y_test_n, y_test_c))
y_train = np.expand_dims(y_train, -1)
y_test = np.expand_dims(y_test, -1)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input

baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
for layer in baseModel.layers:
    layer.trainable = False
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout

headModel = AveragePooling2D(pool_size=(4, 4))(baseModel.output)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
from tensorflow.keras.optimizers import Adam

INIT_LR = 1e-3
EPOCHS = 10
BS = 8
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])
print("[INFO] training head...")
H = model.fit_generator(
    trainAug.flow(x_train, y_train, batch_size=BS),
    steps_per_epoch=len(x_train) // BS,
    validation_data=(x_test, y_test),
    validation_steps=len(x_test) // BS,
    epochs=EPOCHS)
preds = model.predict(x_test)
preds[preds > 0.5] = 1
preds[preds < 0.5] = 0
preds
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(x_test, preds, normalize=False)

# cm = confusion_matrix(y_test, preds, normalize=False)
# total = sum(sum(cm))
# acc = (cm[0, 0] + cm[1, 1]) / total
# sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
# specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# print('accuracy: ', acc)
# print('confusion matrix: ', '\n', cm)
# print('sensitivity: ', sensitivity)
# print('specificity: ', specificity)