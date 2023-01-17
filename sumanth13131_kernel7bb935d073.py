# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
imgs_path="/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
file_path="../input/siim-isic-melanoma-classification/train.csv"
import pandas as pd
import numpy as np
from PIL import Image
import os
file=pd.read_csv(file_path)
file.head()
file['target'].value_counts()
file=file.sort_values(by=['target'],ascending=False)
file.head()
file['target'].value_counts()

from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
data=[]
labels=[]
for i in range(0,2000):
    img_name=str(file.iloc[i,0])+'.jpg'
    labels.append(file.iloc[i,7])
    img=imgs_path+img_name
    image = load_img(img, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)
    if i%50 == 0:
        print(i,end=' ')
X_train,X_val, y_train, y_val=train_test_split(data,labels,test_size=0.2,shuffle=True)
np.array(X_train).shape,np.array(X_val).shape,np.array(y_train).shape,np.array(y_val).shape
y_val.count(0),y_val.count(1)
y_train.count(0),y_val.count(1)
# train_data = np.array(X_train, dtype="float32")
# train_labels = np.array(y_train)
# val_data = np.array(X_val, dtype="float32")
# val_labels = np.array(y_val)
# labels.count(1)
from keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest")
INIT_LR = 1e-3
EPOCHS = 60
BS = 20
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
H = model.fit(aug.flow(np.array(X_train, dtype="float32"), np.array(y_train), batch_size=BS),
              steps_per_epoch=len(X_train) //(6* BS),
              validation_data=(np.array(X_val, dtype="float32"),np.array(y_val)),
              validation_steps=len(X_val) // (6*BS),
              epochs=EPOCHS)
from matplotlib import pyplot as plt

N = EPOCHS
plt.style.use("ggplot")
plt.figure(figsize=(15,4))
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch--->")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="center right")
model.save('model.h5')
