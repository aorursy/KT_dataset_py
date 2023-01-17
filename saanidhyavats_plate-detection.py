# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.optimizers import Adam
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
from keras.layers import Dense,Conv2D,Flatten
from keras.models import Sequential,Model
from keras.applications.vgg16 import VGG16

img_size=(224,224)
channel=3
height=224
width=224
df=[]
for i in range(0,237):
  s=["../input/license-plate-detection/Indian Number Plates/licensed_car",str(i),".jpeg"]
  f=cv2.imread(s[0]+s[1]+s[2])
  f=cv2.resize(f,img_size)
  df=df+[f]   
df=np.array(df)
data=pd.read_csv("../input/license-plate-detection/indian_license_plates.csv")
data
import matplotlib.pyplot as plt
import pytesseract
def show_img(index):
    image=df[index,:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(width, height))
    ty=int(data["top_y"].iloc[index]*height)
    tx=int(data["top_x"].iloc[index]*width)
    bx=int(data["bottom_x"].iloc[index]*width)
    by=int(data["bottom_y"].iloc[index]*height)
    image=cv2.rectangle(image,(tx,ty),(bx,by),(255,0,0),1)
    #rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    cropped= image[ty:by,tx:bx]
    text = pytesseract.image_to_string(image) 
    print(text)
    plt.imshow(cropped)
    plt.show
show_img(29)

X_train=df[0:200]
X_test=df[200:237]
Y_train=data.loc[0:199,["top_x","top_y","bottom_x","bottom_y"]]
Y_test=data.loc[200:236,["top_x","top_y","bottom_x","bottom_y"]]
X_train[1]
adam=Adam(lr=0.05)
model=Sequential()
model.add(VGG16(weights="imagenet",include_top=False,input_shape=(height,width,channel)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='sigmoid'))
model.add(Dense(4,activation='sigmoid'))
model.layers[-6].trainable=False

model.summary()
model.compile(optimizer='adam',loss='mse')
model.fit(X_train,Y_train,batch_size=10,epochs=20,validation_data=(X_test,Y_test))
import cv2 
import pytesseract 
  
  
# Read image from which text needs to be extracted 
#img = cv2.imread("sample.jpg") 
  
# Preprocessing the image starts 
  
# Convert the image to gray scale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
# Performing OTSU threshold 
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
  
# Specify structure shape and kernel size.  
# Kernel size increases or decreases the area  
# of the rectangle to be detected. 
# A smaller value like (10, 10) will detect  
# each word instead of a sentence. 
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
  
# Appplying dilation on the threshold image 
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
  
# Finding contours 
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
                                                 cv2.CHAIN_APPROX_NONE) 
  
# Creating a copy of image 
im2 = img.copy() 
  
# A text file is created and flushed 
file = open("recognized.txt", "w+") 
file.write("") 
file.close() 
  
# Looping through the identified contours 
# Then rectangular part is cropped and passed on 
# to pytesseract for extracting text from it 
# Extracted text is then written into the text file 
for cnt in contours: 
    x, y, w, h = cv2.boundingRect(cnt) 
      
    # Drawing a rectangle on copied image 
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
      
    # Cropping the text block for giving input to OCR 
    cropped = im2[y:y + h, x:x + w] 
      
    # Open the file in append mode 
    file = open("recognized.txt", "a") 
      
    # Apply OCR on the cropped image 
    text = pytesseract.image_to_string(cropped) 
      
    # Appending the text into file 
    file.write(text) 
    file.write("\n") 
      
    # Close the file 
    file.close 

t=X_test[1].reshape(1,224,224,3)
t.shape
print(t)
t=X_train[39].reshape(1,224,224,3)
img = t
y_out=model.predict(img)*width
xt,yt = y_out[0][0], y_out[0][1]
xb,yb = y_out[0][2], y_out[0][3]
print(y_out)
img=img.reshape(224,224,3)    
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image = cv2.rectangle(img, (xt, yt), (xb, yb), (255,255,0), 1)
plt.imshow(image)
plt.show()
"""
y_hat = model.predict(img.reshape(1, 224, 224, 3)).reshape(-1) * width
xt, yt = y_hat[0], y_hat[1]
xb, yb = y_hat[2], y_hat[3]
print(y_hat)
img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
image = cv2.rectangle(img, (xt, yt), (xb, yb), (0, 0, 255), 1)
plt.imshow(image)
plt.show()"""
cropped= image[int(yt):int(yb),int(xt):int(xb)]
text = pytesseract.image_to_string(cropped) 
print(text)
plt.imshow(cropped)
plt.show()

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

x=[[2,3]]
x[0][1]
