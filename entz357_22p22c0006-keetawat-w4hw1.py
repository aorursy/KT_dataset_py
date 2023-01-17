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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras import Model, layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import seaborn as sns
import cv2
from skimage import feature
from skimage import measure
from skimage.morphology import convex_hull_image
from skimage.util import invert
from sklearn.model_selection import train_test_split
from tqdm import tqdm

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
sns.set(style='whitegrid')
%matplotlib inline
train_fold = '../input/thai-mnist-classification/train'
test_fold = '../input/thai-mnist-classification/test'
thai_minst_map_path = '../input/thai-mnist-classification/mnist.train.map.csv'
train_rules = pd.read_csv("../input/thai-mnist-classification/train.rules.csv")
train_rules
test_rules = pd.read_csv('../input/thai-mnist-classification/test.rules.csv')
test_rules
thai_mnist_map = pd.read_csv(thai_minst_map_path)
thai_mnist_map.head()
thai_mnist_map.shape
drop_train_set = pd.read_csv('../input/thaimnist-droplist/drop_lists.csv')
drop_train_set.head()
drop_list = thai_mnist_map['id'].isin(drop_train_set['id'])
thai_mnist_map[drop_list==True].shape
drop_index = thai_mnist_map[drop_list==True].index
filter_df = thai_mnist_map.drop(drop_index, axis=0)
filter_df['category'].value_counts().plot(kind='bar')
class getdata():
    def __init__(self,df, group='train'):
        self.label_df = df 
        self.group = group
        self.dataFile = self.label_df['id'].values
        self.label = self.label_df['category'].values
        self.n_index = len(self.dataFile)
        
    def getImageByIndex(self,index):
        return self.label_df.iloc[index]['id']
    
    def getLabelByIndex(self,index):
        return self.label_df.iloc[index]['category']
        
    def getImage(self,index,mode='rgb',label = False ):
        image_id = self.label_df.iloc[index]['id']
        path = f"../input/thai-mnist-classification/{self.group}/{image_id}"
        img  = cv2.imread(path)
        if mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == 'gray':
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if label:
            return img,self.label_df.iloc[img_index]['category']
        return img
    
    def get1img(self,index,mode='rgb',label = False):
        image_id = self.label_df.iloc[index]['id']
        path = f"../input/thai-mnist-classification/{self.group}/{image_id}"
        img = cv2.imread(path)
        if mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == 'gray':
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if label:
            return img,self.label_df.iloc[index]['category']
        return img
gdt = getdata(filter_df, group='train')
gdt.dataFile.shape
# Prepare Validation set
test_images = [f for f in os.listdir(test_fold) if os.path.isfile(os.path.join(test_fold, f))]
valid_array = {'id': [], 'category': []}

for i in range(len(test_images)):
    valid_array['id'].append(test_images[i])
    valid_array['category'].append(0)
    # print(test_images[1])

valid_df = pd.DataFrame(valid_array , columns=['id','category'])
valid_df
gvalid = getdata(valid_df, group='test')
gvalid.dataFile.shape
temp_img = invert(gdt.get1img(555,'gray'))
fig, [ax1,ax2] = plt.subplots(1, 2)
ax1.imshow(temp_img)
cvh =  convex_hull_image(temp_img)
ax2.imshow(cvh)
def convex_crop(img,pad=20):
    convex = convex_hull_image(img)
    r,c = np.where(convex)
    while (min(r)-pad < 0) or (max(r)+pad > img.shape[0]) or (min(c)-pad < 0) or (max(c)+pad > img.shape[1]):
        pad = pad - 1
    return img[min(r)-pad:max(r)+pad,min(c)-pad:max(c)+pad]
crop_img = convex_crop(temp_img,pad=10)
plt.imshow(crop_img)
hight = 32
width = 32

def convex_resize(img):
    img = invert(img)
    img = convex_crop(img,pad=20)
    img = cv2.resize(img,(hight,width))
    return img
def thes_resize(img,thes=40):
    img = invert(img)
    img = convex_crop(img,pad=20)
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
temp_img = gdt.get1img(555,'gray')
fig, [ax1,ax2] = plt.subplots(1, 2,figsize=(10,7))
ax1.imshow(convex_resize(temp_img))
ax1.set_title('Without thresholding')
ax2.imshow(thes_resize(temp_img))
ax2.set_title('Thresholding')
fig, ax = plt.subplots(5, 5, figsize=(15,15))
for i in range(5):
    for j in range(5):
        img_index = np.random.randint(0,gdt.n_index)
        ax[i][j].imshow(thes_resize(gdt.get1img(img_index,'gray')))
        ax[i][j].set_title('Class: '+str(gdt.label[img_index]))
        ax[i][j].set_axis_off()
### Apply training set image
X = []

for i , image_path in enumerate(tqdm(gdt.dataFile)):
    X.append(thes_resize(gdt.get1img(i,'gray')))

X = np.array(X)   
y = gdt.label
X = X.reshape((-1,32,32,1))
X.shape,y.shape
### Apply validation set image
X_valid = []

for i , image_path in enumerate(tqdm(gvalid.dataFile)):
    image = gvalid.get1img(i,'gray')
    X_valid.append(thes_resize(image))

X_valid = np.array(X_valid)    
X_valid = X_valid.reshape((-1,32,32,1))
X_valid.shape
X = X.reshape((-1, hight, width, 1))
y = filter_df['category']

print("X Shape: {}\ny Shape: {}".format(X.shape, y.shape))
y_cat = tf.keras.utils.to_categorical(y)
print("y_cat Shape: {}".format(y_cat.shape))
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
X_train = X_train / 255.
X_test = X_test / 255.
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Dense(10, activation='softmax'))
# LeNet-5
model = Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D(strides=2))
model.add(layers.Conv2D(48, (5, 5), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(layers.MaxPooling2D(strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
# compile model
#opt = SGD(lr=0.001, momentum=0.9)
#model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

adam = Adam(lr=5e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
epochs = 50
BATCH_SIZE = 64

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  verbose=1,
                                                 patience=5,
                                                 mode='max',
                                                 restore_best_weights = True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=5,
                                                verbose=1,
                                                factor=0.2,
                                                min_lr=1e-6)
history = model.fit(X_train,
                   y_train,
                   validation_data=(X_test, y_test),
                   epochs=epochs,
                   batch_size=BATCH_SIZE,
                   callbacks = [early_stopping],
                   verbose=1)
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
score = model.evaluate(X_test, y_test, batch_size=32)
score
# save model
#model.save('./CNN_Model.h5')
model.save('./LeNet_Model.h5')
# Load Model
model_loaded = tf.keras.models.load_model('./LeNet_Model.h5')
model_loaded.summary()
def image_preprocess(image):
    return thes_resize(gdt.get1img(image,'gray'))
X_train.shape
range(len(X_train))
pred_index = 99
np.argmax(model_loaded.predict(X_train[pred_index].reshape(1, 32, 32, 1)))
np.argmax(y_train[pred_index])
y_pred_train = []

for img in range(len(X_train)):
    classes = model_loaded.predict(X_train[img].reshape(1, 32, 32, 1))
    y_pred_train.append(np.argmax(classes))
X_train[1]
y_pred_train[1]
np.argmax(y_train[1])
np.argmax(y_train[1]) == y_pred_train[1]
wrong_list1 = []
wrong_label1 = []
right_label1 = []

for image, label_actual, label_predict in zip(X_train, y_train, y_pred_train):
    if np.argmax(label_actual) != label_predict:
        wrong_list1.append(image)
        wrong_label1.append(label_predict)
        right_label1.append(label_actual)
        
len(wrong_list1)
y_pred_test = []

for img in range(len(X_test)):
    classes = model_loaded.predict(X_test[img].reshape(1, 32, 32, 1))
    y_pred_test.append(np.argmax(classes))
wrong_list2 = []
wrong_label2 = []
right_label2 = []

for image, label_actual, label_predict in zip(X_test, y_test, y_pred_test):
    if np.argmax(label_actual) != label_predict:
        wrong_list1.append(image)
        wrong_label1.append(label_predict)
        right_label1.append(label_actual)
        
len(wrong_list2)
fig, axs = plt.subplots(5, 5, figsize=(20, 20))

col = -1

for index, image, wrong, right in zip(range(25), wrong_list2[:25], wrong_label2[:25], right_label2[:25]):
    path = os.path.join(training_dir, image)
    image = Image.open(path)
    row = index%5
    if row == 0:
        col = col + 1

    axs[row,col].imshow(np.array(image))
    axs[row,col].set_title('Predict as {}, Actual {}'.format(wrong[0], right))
    axs[row,col].grid(False)

plt.show()
thai_mnist_map.head()
# filter_df.head()
train_rules.head()
df_train_rules_map = train_rules.drop('id',
                                      axis=1).merge(thai_mnist_map,
                                                    how = 'left',
                                                    left_on=['feature1'],
                                                    right_on=['id']
                                                   ).rename(columns={'category':'f1'}).drop(['id', 'feature1'], axis=1)
df_train_rules_map.head()
df_train_rules_map = df_train_rules_map.merge(thai_mnist_map,
                                                    how = 'left',
                                                    left_on=['feature2'],
                                                    right_on=['id']
                                                   ).rename(columns={'category':'f2'}).drop(['id', 'feature2'], axis=1)
df_train_rules_map.head()
df_train_rules_map = df_train_rules_map.merge(thai_mnist_map,
                                                    how = 'left',
                                                    left_on=['feature3'],
                                                    right_on=['id']
                                                   ).rename(columns={'category':'f3'}).drop(['id', 'feature3'], axis=1)
df_train_rules_map
df_train_rules_map['predict'].value_counts()
test_rules.tail()
test_image = os.listdir(test_fold)
test_image
df_test_predict = pd.DataFrame(test_image, columns = ['map_id'])
df_test_predict.head()
len(df_test_predict)
X_valid
y_pred = []

for img in range(len(X_valid)):
    classes = model_loaded.predict(X_valid[img].reshape(1, 32, 32, 1))
    y_pred.append(np.argmax(classes))
df_test_predict['category'] = y_pred
df_test_predict.head(10)
plt.figure(figsize=(22, 22))

number = 10
img_list = []

for i in range(number):
    temp = list(df_test_predict[df_test_predict['category'] == i]['map_id'][:10])
    img_list = img_list + temp

for index, file in enumerate(img_list):
    path = os.path.join(test_fold,file)
    plt.subplot(number,len(img_list)/number,index+1)
    img = mpimg.imread(path)
    plt.axis('off')
    plt.imshow(img)
test_rules
df_test_rules_map = test_rules.merge(df_test_predict,
                                                    how = 'left',
                                                    left_on=['feature1'],
                                                    right_on=['map_id']
                                                   ).rename(columns={'category':'f1'}).drop(['feature1'], axis=1)
df_test_rules_map = df_test_rules_map.drop('map_id',
                                      axis=1).merge(df_test_predict,
                                                    how = 'left',
                                                    left_on=['feature2'],
                                                    right_on=['map_id']
                                                   ).rename(columns={'category':'f2'}).drop(['map_id','feature2'], axis=1)
df_test_rules_map = df_test_rules_map.merge(df_test_predict,
                                                    how = 'left',
                                                    left_on=['feature3'],
                                                    right_on=['map_id']
                                                   ).rename(columns={'category':'f3'}).drop(['map_id', 'feature3'], axis=1)
df_test_rules_map
def calculate(f1,f2,f3):
    if(f1 == 0):
        return f2*f3
    elif(f1 == 1):
        return abs(f2-f3)
    elif(f1 == 2):
        return (f2 + f3)*abs(f2 - f3)
    elif(f1 == 3):
        return np.abs((f3 * (f3 + 1) - f2 * (f2 - 1))/2)
    elif(f1 == 4):
        return 50 + (f2 - f3)
    elif(f1 == 5):
        return min([f2 , f3])
    elif(f1 == 6):
        return max([f2 , f3])
    elif(f1 == 7):
        return ((f2 * f3) % 9) *11
    elif(f1 == 8):
        return (((f2 ** 2) + 1) * f2) + (f3 * (f3 + 1)) % 99
    elif(f1 == 9):
        return 50 + f2
    else:
        return f2 + f3
Test_predict_rule = []

for i in range(len(df_test_rules_map)):
    Test_predict_rule.append(calculate(df_test_rules_map['f1'].iloc[i],
                                      df_test_rules_map['f2'].iloc[i],
                                      df_test_rules_map['f3'].iloc[i]).astype(int))
    
df_test_rules_map['predict'] = Test_predict_rule
df_test_rules_map.sample(15)
df_result = df_test_rules_map[['id','predict']]

plt.figure(figsize=(15, 5))
df_result['predict'].value_counts()
df_result.to_csv('submit_predictions.csv', index = False)
