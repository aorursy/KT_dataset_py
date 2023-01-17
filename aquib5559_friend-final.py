import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import os
from tqdm import tqdm
import cv2
from matplotlib.image import imread
images = []
labels = []

for dirname, _, filenames in os.walk('../input/friendship-goal-hackerearth/Data/train/Adults'):
    for filename in filenames:
        images.append((os.path.join(dirname, filename)))
        labels.append(0)
for dirname, _, filenames in os.walk('../input/friendship-goal-hackerearth/Data/train/Teenagers'):
    for filename in filenames:
        images.append((os.path.join(dirname, filename)))
        labels.append(1)
for dirname, _, filenames in os.walk('../input/friendship-goal-hackerearth/Data/train/Toddler'):
    for filename in filenames:
        images.append((os.path.join(dirname, filename)))
        labels.append(2)
images
c = images[1]
imread(c)
plt.imshow(imread(c))
print(labels[1])
img = []
for i in range(0,len(images)):
    imgs = cv2.imread(images[i])
    imgs = cv2.resize(imgs,(224,224))
    img.append(imgs)
data = np.array(img,dtype=np.float32)/255.0
data = np.reshape(data,(data.shape[0],224,224,3))
target = np.array(labels)
from tensorflow.keras.utils import to_categorical
new_target = to_categorical(target,num_classes=3)
data.shape, target.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data,new_target,test_size = 0.20,random_state=42)
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(Conv2D(filters = 100,kernel_size=(3,3),activation = 'relu',input_shape = (224,224,3)))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(200,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(300,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(400,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.6))


model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(3,activation = 'softmax'))


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics = ['accuracy'])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=15)
model.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test),batch_size=64,callbacks=[early_stop])
test_cvs = pd.read_csv("../input/friendship-goal-hackerearth/Data/Test.csv")
test = "../input/friendship-goal-hackerearth/Data/test"
test_img = []
path = test
for i in tqdm(test_cvs['Filename']):
    final_path = os.path.join(path,i)
    img = cv2.imread(final_path)
    img = cv2.resize(img,(224,224))
    test_img.append(img)
test_data = np.array(test_img)
test_data = np.reshape(test_data,(test_data.shape[0],224,224,3))
prediction = model.predict_classes(test_data)
prediction
class_map = {0:'Adults', 1 : "Teenagers", 2 : "Toddler"}
test_cvs['Category'] = prediction
test_cvs['Category'] = test_cvs['Category'].map(class_map)
test_cvs.head()
test_cvs.to_csv('submission.csv',index=False)
pred = pd.read_csv('./submission.csv')
actual = pd.read_csv("../input/sample/Sample Submission.csv")
pred.head()
actual.head()
inverse_class = {'Adults': 0 ,'Teenagers':1, 'Toddler':2}
pred['Category'] = pred['Category'].map(inverse_class)
actual['Category'] = actual['Category'].map(inverse_class)
pred.head()
actual.head()
y_pred = pred['Category'].to_numpy()
y_actual = actual['Category'].to_numpy()
from sklearn.metrics import recall_score
score = 100*recall_score(y_actual,y_pred,average='macro')
score

