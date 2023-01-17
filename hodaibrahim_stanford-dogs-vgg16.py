path = "../input/stanford-dogs-dataset/images/Images/"
import os
len(os.listdir(path))
from PIL import Image

im =Image.open('../input/stanford-dogs-dataset/images/Images/n02087394-Rhodesian_ridgeback/n02087394_10238.jpg').resize((128,128))
im
'../input/stanford-dogs-dataset/images/Images/n02087394-Rhodesian_ridgeback'.split('/')[-1].split('-')[-1]
dogs_labels = set()

    
for d in os.listdir(path):
    dogs_labels.add(d)
len(dogs_labels)
dogs_labels = list(dogs_labels)
dogs_labels_path = [path + s for s in dogs_labels]
dogs_labels_path
import cv2
def load_and_preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224,224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
X = []
y = []

for type_path, label in zip(dogs_labels_path,dogs_labels) :
    for image_path in os.listdir(type_path):
        image = load_and_preprocess_image(type_path+"/"+image_path)
        
        X.append(image)
        y.append(label)
y
import matplotlib.pyplot as plt
plt.imshow(X[1000])
from sklearn.preprocessing import LabelBinarizer
import numpy as np
encoder = LabelBinarizer()

X = np.array(X)
y = encoder.fit_transform(np.array(y))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
base_model=VGG16(weights='imagenet',include_top=False)

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.5)(x)
x=Dense(512,activation='relu')(x)
preds=Dense(len(dogs_labels),activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers[:-5]:
    layer.trainable=False
for layer in model.layers[-5:]:
    layer.trainable=True
    
model.compile("adam",loss="categorical_crossentropy",metrics=["accuracy"])

print(model.summary())
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.optimizers import Adam,SGD
checkpoint = ModelCheckpoint(
    './base.model',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=30,
    verbose=1,
    mode='auto'
)
history = model.fit(X_train,y_train,batch_size=64,epochs=50,validation_data=(X_test,y_test), callbacks=[checkpoint,earlystop])
predictions = model.predict(X_test)
label_predictions = encoder.inverse_transform(predictions)
rows, cols = 5, 3
size = 25

fig,ax=plt.subplots(rows,cols)
fig.set_size_inches(size,size)
for i in range(rows):
    for j in range (cols):
        index = np.random.randint(0,len(X_test))
        ax[i,j].imshow(X_test[index])
        ax[i,j].set_title(f'Predicted: {label_predictions[index]}\n Actually: {encoder.inverse_transform(y_test)[index]}')
        
plt.tight_layout()
