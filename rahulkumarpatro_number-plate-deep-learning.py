# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import urllib
import matplotlib.pyplot as plt
import cv2
import glob
import os
import time
from PIL import Image
df = pd.read_json("/kaggle/input/vehicle-number-plate-detection/Indian_Number_plates.json", lines=True)
df.head()
df.shape
new_csv=df.to_csv("indian_license_plates.csv", index=False)
new_df=pd.read_csv("/kaggle/working/indian_license_plates.csv")
new_df.head(10)
df['annotation'][0]
os.mkdir("Number Plates")
data = dict()
data["img_name"] = list()
data["img_width"] = list()
data["img_height"] = list()
data["top-x"] = list()
data["top-y"] = list()
data["bottom-x"] = list()
data["bottom-y"] = list()
data
df['annotation'][0]
df['annotation'][0][0]["points"]
new_df.head(5)
# for index,row in new_df.iterrows():
#     print(row)
count = 0
for index, row in df.iterrows():
    img = urllib.request.urlopen(row["content"])
    img = Image.open(img)
    img = img.convert('RGB')
    img.save("Number Plates/car{}.jpeg".format(count), "JPEG")
    
    data["img_name"].append("car{}".format(count))
    
    d = row["annotation"]
    
    data["img_width"].append(d[0]["imageWidth"])
    data["img_height"].append(d[0]["imageHeight"])
    data["top-x"].append(d[0]["points"][0]["x"])
    data["top-y"].append(d[0]["points"][0]["y"])
    data["bottom-x"].append(d[0]["points"][1]["x"])
    data["bottom-y"].append(d[0]["points"][1]["y"])
    
    count += 1
    
print("Done Successfully")    
# data
new_data=pd.DataFrame(data)
new_data.head()
new_data.dtypes
new_data.shape
new_data.describe()
new_data.dtypes
new_data['img_name']=new_data['img_name']+".jpeg"
new_data
width= 300
height= 300
channels= 3
def viewimage(t):
    
    image = cv2.imread("Number Plates/" + new_data["img_name"].iloc[t])
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(width,height))
    
    top_x=int(new_data['top-x'].iloc[t]* width)
    top_y=int(new_data['top-y'].iloc[t]*height)
    bot_x=int(new_data['bottom-x'].iloc[t]*height)
    bot_y=int(new_data['bottom-y'].iloc[t]*height)
    
    
    new_img=cv2.rectangle(image,(top_x,top_y),(bot_x,bot_y),(0, 0, 255), 1)
    
    plt.imshow(new_img)
    
    plt.show()
viewimage(10)
viewimage(100)
n = 5
drop_indices = np.random.choice(new_data.index, n, replace=False)
df_subset = new_data.drop(drop_indices)
df_subset
drop_indices
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_dataframe(
    df_subset,
    directory="Number Plates/",
    x_col="img_name",
    y_col=["top-x", "top-y", "bottom-x", "bottom-y"],
    target_size=(width,height),
    batch_size=32, 
    class_mode="raw",
    subset="training")

validation_generator = datagen.flow_from_dataframe(
    df_subset,
    directory="Number Plates/",
    x_col="img_name",
    y_col=["top-x", "top-y", "bottom-x", "bottom-y"],
    target_size=(width,height),
    batch_size=32, 
    class_mode="raw",
    subset="validation")
train_generator
len(train_generator)
len(validation_generator)
model = Sequential()
model.add(VGG16(weights="imagenet", include_top=False, input_shape=(width,height,channels)))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="sigmoid"))

model.layers[-6].trainable = False
model.summary()
adam = Adam(lr=0.0005)
model.compile(optimizer='adam', loss="mse",metrics=['accuracy'])
history = model.fit_generator(train_generator,
    steps_per_epoch=6,
    validation_data=validation_generator,
    validation_steps=2,
    epochs=20)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();
import pytesseract
from PIL import Image
for idx, row in new_data.iloc[drop_indices].iterrows():    
    
    img = cv2.resize(cv2.imread("Number Plates/" + row['img_name']) / 255.0, dsize=(width,height))
    y_hat = model.predict(img.reshape(1, width,height, 3)).reshape(-1) * width
    
    xt, yt = y_hat[0], y_hat[1]
    xb, yb = y_hat[2], y_hat[3]
    
    img = cv2.cvtColor(img.astype(np.float32),cv2.COLOR_BGR2RGB)
    image = cv2.rectangle(img, (xt, yt), (xb, yb), (0,0,255), 1)
    
    clone = image.copy() 
    
    # Cropping the predicted reactangle region into a new image
    crop_img = clone[int(yt):int(yb),int(xt):int(xb)] 
   
    plt.imshow(crop_img)
    im = Image.fromarray((crop_img * 255).astype(np.uint8))
   
#     plt.imshow(crop_img)
    
    
   
    plt.show()
    ## Detecting Car Number using pytesseract
    car_number = pytesseract.image_to_string(im, lang="eng")
    
    print("The Car Number is",car_number)