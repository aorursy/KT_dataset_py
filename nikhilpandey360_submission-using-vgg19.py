
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
from tqdm import  tqdm
print(os.listdir("../input"))
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,GlobalAvgPool2D
from keras.layers import Conv2D, MaxPooling2D
import keras
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3
from keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
df = pd.read_csv("../input/labels.csv")
df.info()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cat = le.fit_transform(df.breed)
from keras.utils.np_utils import to_categorical
mat = to_categorical(cat)
import cv2
def readImgResize(name,path):
    img = cv2.imread(path+name)
    img = cv2.resize(img,(150,150))
    return image.img_to_array(img)
training_data = np.zeros(shape=(len(df.id),150,150,3))
for i,j in tqdm(enumerate(df.id)):
    training_data[i]=readImgResize(j+".jpg",path="../input/train/")
    
    
X_train, X_test, y_train, y_test = train_test_split( training_data, mat, test_size=0.05, random_state=11)
del training_data, mat
from keras.applications import InceptionV3
from keras.models import Model
from keras.optimizers import adam
num_class = 120
im_size = 150
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(im_size, im_size, 3))

# Add a new top layer
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_class, activation='softmax')(x)

# This is the model we wi`l train
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

model.compile(adam(lr=0.00001),loss='categorical_crossentropy', 
              metrics=["accuracy"])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.summary()
model.fit(X_train/255, y_train, epochs=25, validation_data=(X_test/255, y_test), verbose=1)
model.save("breed_vgg19.h5")
from keras.models import load_model
del X_train,X_test
te = os.listdir("../input/test/")
te_in = np.zeros((len(te),150,150,3))
for num , i in enumerate(te):
    img = readImgResize(i,path="../input/test/")/255
    te_in[num]=img
    
pred = model.predict(te_in)

submission = pd.DataFrame(pred , columns =le.classes_.tolist())
submission["id"]=[i.split(".")[0] for i in os.listdir("../input/test/")]
submission = submission[["id"]+submission.columns[:-1].tolist()]
submission.sort_values(by=['id'])
submission.to_csv("submission.csv",index = False)

