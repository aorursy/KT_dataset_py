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
from matplotlib import pyplot as plt #for viewing images and plots
%matplotlib inline 
#So that Matplotlib plots don't open in separate windows outside the notebook

import urllib #For fetching data from Web URLs

import cv2   #For image processing

from sklearn.preprocessing import LabelEncoder    #For encoding categorical variables
from sklearn.model_selection import train_test_split #For splitting of dataset

#All tensorflow utilities for creating, training and working with a CNN
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
os.listdir("../input/thai-mnist-classification")
os.listdir("../input/thai-mnist-classification/train")
df = pd.read_csv(r'/kaggle/input/thai-mnist-classification/mnist.train.map.csv')
df.head()
def show_image_from_url(image_url):
    path = r'/kaggle/input/thai-mnist-classification/train/'+image_url
    nparr = np.fromstring(open(path),np.uint8)
    image = cv2.resize(image,None,fx=2.5, fy=2.5, interpolation = cv2.INTER_CUBIC)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb), plt.axis('off')
plt.figure()
#show_image_from_url(df['id'].loc[0])
df.info(), df.head()
print('All categories : \n ', df['category'].unique())
#Fetch the categories column from the dataframe, and tranform into to numerical labels

encoder = LabelEncoder()
Targets = encoder.fit_transform(df['category'])
Targets
Targets.shape
class getdata():
    def __init__(self,data_path,label_path):
        self.dataPath = data_path
        self.labelPath = label_path
        self.label_df = pd.read_csv(label_path)
        self.dataFile = self.label_df['id'].values
        self.label = self.label_df['category'].values
        self.n_index = len(self.dataFile)
        
    
    def get1img(self,img_index,mode='rgb',label = False):
        img = cv2.imread( os.path.join(self.dataPath,self.label_df.iloc[img_index]['id']) )
        if mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == 'gray':
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if label:
            return img,self.label_df.iloc[img_index]['category']
        return img
df_img_path = '/kaggle/input/thai-mnist-classification/train'
df_cate_path = '/kaggle/input/thai-mnist-classification/mnist.train.map.csv'
gdt = getdata(df_img_path,df_cate_path)
plt.gray()
#Image convex hull
from skimage.morphology import convex_hull_image
from skimage.util import invert
temp_img = invert(gdt.get1img(15,'gray'))
fig, [ax1,ax2] = plt.subplots(1, 2)
ax1.imshow(temp_img)
cvh =  convex_hull_image(temp_img)
ax2.imshow(cvh)
#Crop image
def convex_crop(img,pad=20):
    convex = convex_hull_image(img)
    r,c = np.where(convex)
    while (min(r)-pad < 0) or (max(r)+pad > img.shape[0]) or (min(c)-pad < 0) or (max(c)+pad > img.shape[1]):
        pad = pad - 1
    return img[min(r)-pad:max(r)+pad,min(c)-pad:max(c)+pad]
crop_img = convex_crop(temp_img,pad=10)
plt.imshow(crop_img)
def convex_resize(img):
    img = invert(img)
    img = convex_crop(img,pad=15)
    img = cv2.resize(img,(32,32))
    return img
def thes_resize(img,thes=40):
    img = invert(img)
    img = convex_crop(img,pad=15)
    img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 300):
        img = cv2.resize(img,(300,300))
        img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 150):
        img = cv2.resize(img,(150,150))
        img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(80,80))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(50,50))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(32,32))
    img = ((img > thes)*255).astype(np.uint8)
    return img
temp_img = gdt.get1img(64,'gray')
fig, [ax1,ax2] = plt.subplots(1, 2,figsize=(12,8))
ax1.imshow(convex_resize(temp_img))
ax1.set_title('Without thresholding')
ax2.imshow(thes_resize(temp_img))
ax2.set_title('Thresholding')
fig, ax = plt.subplots(5, 5, figsize=(17,17))
for i in range(5):
    for j in range(5):
        img_index = np.random.randint(0,gdt.n_index)
        ax[i][j].imshow(thes_resize(gdt.get1img(img_index,'gray')))
        ax[i][j].set_title('Class: '+str(gdt.label[img_index]))
        ax[i][j].set_axis_off()
X = []
for i in range(gdt.n_index):
    X.append(thes_resize(gdt.get1img(i,'gray')))
    if (i+1) % 100 == 0:
        print(i)
X = np.array(X)
y = gdt.label
X = X.reshape((-1,32,32,1))
X.shape,y.shape
import tensorflow as tf
y_cate = tf.keras.utils.to_categorical(y)
y_cate.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_cate, test_size=0.25, random_state=72)
X_train = X_train / 255.
X_test = X_test / 255.
X_train.shape, y_train.shape
#Convolutional Neural Network Model

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',
                 input_shape = (32,32,1)))
model.add(BatchNormalization())

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

learning_rate = 0.001

model.compile(loss = categorical_crossentropy,
              optimizer = Adam(learning_rate),
              metrics=['accuracy'])

model.summary()
#Save the model during training 

save_at = "/kaggle/working/THnum_Model.hdf5"
save_best = ModelCheckpoint (save_at, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
#Train the CNN
history = model.fit(X_train, y_train, 
                    epochs = 30, batch_size = 64, 
                    callbacks=[save_best], verbose=1, 
                    validation_data = (X_test,y_test))
# Plot the training history (Training accuracy & Validation accuracy)

plt.figure(figsize=(6, 5))
plt.plot(history.history['accuracy'], color='g')
plt.plot(history.history['val_accuracy'], color='b')
plt.plot(history.history['loss'], color='r')
plt.plot(history.history['val_loss'])
plt.title('Model Accuracy', weight='bold', fontsize=16)
plt.ylabel('Accuracy & Loss', weight='bold', fontsize=14)
plt.xlabel('epoch', weight='bold', fontsize=14)
plt.ylim(0.0, 1.0)
plt.xticks(weight='bold', fontsize=12)
plt.yticks(weight='bold', fontsize=12)
plt.legend(['train', 'val', 'loss', 'val loss'], loc='upper right', prop={'size': 14})
plt.grid(linewidth='0.5')
plt.show()
#Load model on the held-out test set

model = load_model('/kaggle/working/THnum_Model.hdf5')
score = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy over the test set: \n ', round((score[1]*100), 2), '%')
import numpy as np #for numerical computations
import pandas as pd #for dataframe operations

from matplotlib import pyplot as plt #for viewing images and plots
%matplotlib inline 
#So that Matplotlib plots don't open in separate windows outside the notebook

import urllib #For fetching data from Web URLs

import cv2   #For image processing

from sklearn.preprocessing import LabelEncoder    #For encoding categorical variables
from sklearn.model_selection import train_test_split #For splitting of dataset

#All tensorflow utilities for creating, training and working with a CNN
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
test_img_path = os.listdir('../input/thai-mnist-classification/test')
print(len(test_img_path))
import glob
test_list = []
for name in glob.glob(r'/kaggle/input/thai-mnist-classification/test/*'):
    test_list += [name.split('/kaggle/input/thai-mnist-classification/test/')[1]]
d = {'id':test_list,'category':[0]*(len(test_list))}
df_test = pd.DataFrame(data=d)
df_test
def image_processing2(image_url):

  #Download from image url and import to numpy array
  path = r'/kaggle/input/thai-mnist-classification/test/'+image_url
  img2bgr = cv2.imread(path)               

  #Convert to grayscale 
  img2gray = cv2.cvtColor(img2bgr, cv2.COLOR_BGR2GRAY)

  #Resize image
  img_32x32 = cv2.resize(img2gray, (32, 32))

  #Save image in array
  img_arr = img_32x32.flatten()
  return img_arr
test_image_list = []
test_image_id = test_list

for I in range(len(test_list)) :
    test_image_list.append(image_processing2(test_list[I]))
    X2 = np.array(test_image_list)
    X2 = X2/255
len(X2)
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 1)
X2 = X2.reshape(X2.shape[0], img_rows, img_cols, 1)
X2.shape
model = load_model('/kaggle/working/THnum_Model.hdf5') #
y_pred = np.round(model.predict(X2))
y_pred
encoder.classes_[np.where(y_pred[9] == 1)[0].sum()]
result_all = dict()
test_val = []
for i in range(len(y_pred)):
    test_val += [encoder.classes_[np.where(y_pred[i] == 1)[0].sum()]]
    result_all[test_list[i]] = test_val[i]
d = {'id':test_list,'category':test_val}
df_t = pd.DataFrame(data=d)
df_t.head(), df_t.tail()
df_t.to_csv('/kaggle/working/test_result.csv',index=False)

df_img2_path = '/kaggle/input/thai-mnist-classification/test' 
df_cate2_path = './test_result.csv'
gdt2 = getdata(df_img2_path,df_cate2_path)
fig, ax = plt.subplots(5, 5, figsize=(17,17))
for i in range(5):
    for j in range(5):
        img_index = np.random.randint(1,gdt2.n_index)
        ax[i][j].imshow(thes_resize(gdt2.get1img(img_index,'gray')))
        ax[i][j].set_title('Class: '+str(gdt2.label[img_index]))
        ax[i][j].set_axis_off()
#Image convex hull
from skimage.morphology import convex_hull_image
from skimage.util import invert
temp_img = invert(gdt2.get1img(99,'gray'))
fig, [ax1,ax2] = plt.subplots(1, 2)
ax1.imshow(temp_img)
cvh =  convex_hull_image(temp_img)
ax2.imshow(cvh)
#Crop image
def convex_crop(img,pad=20):
    convex = convex_hull_image(img)
    r,c = np.where(convex)
    while (min(r)-pad < 0) or (max(r)+pad > img.shape[0]) or (min(c)-pad < 0) or (max(c)+pad > img.shape[1]):
        pad = pad - 1
    return img[min(r)-pad:max(r)+pad,min(c)-pad:max(c)+pad]
crop_img = convex_crop(temp_img,pad=10)
plt.imshow(crop_img)
def convex_resize(img):
    img = invert(img)
    img = convex_crop(img,pad=15)
    img = cv2.resize(img,(32,32))
    return img
def thes_resize(img,thes=40):
    img = invert(img)
    img = convex_crop(img,pad=15)
    img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 300):
        img = cv2.resize(img,(300,300))
        img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 150):
        img = cv2.resize(img,(150,150))
        img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(80,80))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(50,50))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(32,32))
    img = ((img > thes)*255).astype(np.uint8)
    return img
temp_img = gdt2.get1img(99,'gray')
fig, [ax1,ax2] = plt.subplots(1, 2,figsize=(12,8))
ax1.imshow(convex_resize(temp_img))
ax1.set_title('Without thresholding')
ax2.imshow(thes_resize(temp_img))
ax2.set_title('Thresholding')
X3 = []
for i in range(gdt2.n_index):
    X3.append(thes_resize(gdt2.get1img(i,'gray')))
    if (i+1) % 100 == 0:
        print(i)
X3 = np.array(X3)
X3.shape
X3 = X3.reshape((-1,32,32,1))
X3.shape
model = load_model('/kaggle/working/THnum_Model.hdf5') # ./model_THnum.hdf5
y_pred2 = np.round(model.predict(X3))
y_pred2
result_all2 = dict()
test_val2 = []
for i in range(len(y_pred2)):
    test_val2 += [encoder.classes_[np.where(y_pred2[i] == 1)[0].sum()]]
    result_all2[test_list[i]] = test_val2[i]
d2 = {'id':test_list,'category':test_val2}
df_t2 = pd.DataFrame(data=d2)
df_t2.to_csv('/kaggle/working/test_result2.csv',index=False)
df_t2.to_csv('/kaggle/working/submit.csv',index=False)
df_img2_path = '/kaggle/input/thai-mnist-classification/test' 
df_cate3_path = './test_result2.csv'
gdt3 = getdata(df_img2_path,df_cate3_path)
fig, ax = plt.subplots(5, 5, figsize=(17,17))
for i in range(5):
    for j in range(5):
        img_index = np.random.randint(1,gdt3.n_index)
        ax[i][j].imshow(thes_resize(gdt3.get1img(img_index,'gray')))
        ax[i][j].set_title('Class: '+str(gdt3.label[img_index]))
        ax[i][j].set_axis_off()
TRAIN_PATH = '/kaggle/input/thai-mnist-classification/train/'
TEST_PATH = '/kaggle/input/thai-mnist-classification/test/'
test_rule = pd.read_csv('/kaggle/input/thai-mnist-classification/test.rules.csv')
train_rule = pd.read_csv('/kaggle/input/thai-mnist-classification/train.rules.csv')
train_rule.head()
def translate(val):
    lock = df_t2.iloc[0]['category']
    return lock
def translate(x):
    try:
        n = df_t2[df_t2['id']==x]
        return n['category'].values[0]
    except:
        return 10
test_rule['feauture1_trans'] = test_rule.apply(lambda x:  translate(x['feature1']) , axis=1)
test_rule['feauture2_trans'] = test_rule.apply(lambda x:  translate(x['feature2']) , axis=1)
test_rule['feauture3_trans'] = test_rule.apply(lambda x:  translate(x['feature3']) , axis=1)
test_rule
def calculate(f1,f2,f3):
    if(f1 == 0):
        return f2*f3
    elif(f1 == 1):
        return abs(f2-f3)
    elif(f1 == 2):
        return (f2 + f3)*abs(f2 - f3)
    elif(f1 == 3):
        return ((f2**2)+1)*(f2) +(f3)*(f3+1)
    elif(f1 == 4):
        return 50 + (f2 - f3)
    elif(f1 == 5):
        return min([f2 , f3])
    elif(f1 == 6):
        return max([f2 , f3])
    elif(f1 == 7):
        return (((f2*f3)/9)*11)
    elif(f1 == 8):
        return ((f3*(f3 +1) - f2*(f2-1))/2)
    elif(f1 == 9):
        return 50 + f2
    else:
        return f2 + f3
test_rule['predict'] = test_rule.apply(lambda x: calculate(x['feauture1_trans'] , x['feauture2_trans'] , x['feauture3_trans']) , axis=1)
test_rule
result = test_rule[['id', 'predict']]
result.head(20)
result.to_csv('submited.csv', index=False)