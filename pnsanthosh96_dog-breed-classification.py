import numpy as np
import pandas as pd 
from PIL import Image
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
print(os.listdir("../input"))
from matplotlib.pyplot import imshow
%matplotlib inline
img = Image.open('../input/train/001513dfcb2ffafc82cccf4d8bbaba97.jpg','r')
imshow(img)
np.array(img).shape
images_list = ([ x for x in os.listdir('../input/train/') if x.endswith('.jpg')])
len(images_list)
df_labels = pd.read_csv('../input/labels.csv')
df_labels.head()
# labels
# lb_np = np.array(labels)
# lb_np.shape
# type(lb_np)
train_slice = 2000
num_of_classes = 120
labels = df_labels['breed']
print(labels.shape)
train_im_lb = df_labels['id']+'.jpg'
for i in train_im_lb[:5]:
    print(i)

#train_images = np.array([cv2.imread('../input/train/'+x) for x in train_im_lb])
train_images = []
from tqdm import tqdm
for i in tqdm(train_im_lb[:train_slice]):
    img = cv2.imread('../input/train/'+i).astype(np.float32)
    img = cv2.resize(img, (500,300))
    img = img / 255.0
    train_images.append(img)
    
print(train_images[0].shape, train_images[13].shape)
x_train = train_images
y_train = df_labels['breed']

y_train = (pd.get_dummies(y_train))
print(y_train.shape)
y_train = y_train[:train_slice]
print(len(y_train),len(x_train))
#(y_train)[0]
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=1)
base_model = VGG19(weights = 'imagenet', include_top=False, input_shape=(300, 500, 3))
base_model.summary()
LastConv = base_model.output
LastConv = Flatten()(LastConv)
predictions = Dense(num_of_classes, activation='softmax')(LastConv)
base_model.input
#model.compile(optimizer='sgd' )
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])


model.fit((X_train), (Y_train), epochs=20, validation_data=((X_valid), (Y_valid)))

