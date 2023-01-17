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
import cv2
import tensorflow as tf

import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from skimage import feature
from skimage import measure
from skimage.morphology import convex_hull_image
from skimage.util import invert
os.listdir('../input/thai-mnist-classification')
dt_train = pd.read_csv('../input/thai-mnist-classification/train.rules.csv')
dt_map = pd.read_csv('../input/thai-mnist-classification/mnist.train.map.csv')
dt_test = pd.read_csv('../input/thai-mnist-classification/test.rules.csv')
train_path = os.path.join('../input/thai-mnist-classification/train')
test_path = os.path.join('../input/thai-mnist-classification/test')
dt_drop = pd.read_csv('../input/thaimnistdroplist/drop_lists.csv')
drop_lists = dt_drop['id'].values

dt_filter = dt_map[~dt_map['id'].isin(drop_lists)]
dt_filter

test_images = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path,f))]
valid = {'id': [],'category': []}

for i in range(len(test_images)):
    valid['id'].append(test_images[i])
    valid['category'].append(0)

dt_valid = pd.DataFrame(valid,columns=['id','category'])
dt_valid
class getdata():
    def __init__(self,dt,group='train'):
        self.label_dt = dt 
        self.group = group
        self.dataFile = self.label_dt['id'].values
        self.label = self.label_dt['category'].values
        self.n_index = len(self.dataFile)
        
    def getImageByIndex(self,index):
        return self.label_dt.iloc[index]['id']
    
    def getLabelByIndex(self,index):
        return self.label_dt.iloc[index]['category']
        
    def getImage(self,index,mode='rgb',label = False ):
        image_id = self.label_dt.iloc[index]['id']
        path = f"../input/thai-mnist-classification/{self.group}/{image_id}"
        img  = cv2.imread(path)
        if mode == 'rgb':
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        elif mode == 'gray':
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if label:
            return img,self.label_dt.iloc[img_index]['category']
        return img
gdt = getdata(dt_filter,group='train')
gdt.dataFile.shape
gvalid = getdata(dt_valid,group='test')
gvalid.dataFile.shape
#Convex hull image
temp_img = invert(gdt.getImage(232,'gray'))
fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(10,10))
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
crop_img = convex_crop(temp_img,pad=30)
plt.imshow(crop_img)
#Resize image
def convex_resize(img):
    img = invert(img)
    img = convex_crop(img,pad=30)
    img = cv2.resize(img,(32,32))
    return img
#Resize image with threshold
def thres_resize(img,thres=40):
    img = invert(img)
    img = convex_crop(img,pad=30)
    img = ((img > thres)*255).astype(np.uint8)
    if(min(img.shape) > 300):
        img = cv2.resize(img,(300,300))
        img = ((img > thres)*255).astype(np.uint8)
    if(min(img.shape) > 150):
        img = cv2.resize(img,(150,150))
        img = ((img > thres)*255).astype(np.uint8)
    img = cv2.resize(img,(80,80))
    img = ((img > thres)*255).astype(np.uint8)
    img = cv2.resize(img,(50,50))
    img = ((img > thres)*255).astype(np.uint8)
    img = cv2.resize(img,(32,32))
    img = ((img > thres)*255).astype(np.uint8)
    return img
# Thesholding Train Data
temp_img = gdt.getImage(64,'gray')
fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(10,10))
ax1.imshow(convex_resize(temp_img))
ax1.set_title('Without thresholding')
ax2.imshow(thres_resize(temp_img))
ax2.set_title('Thresholding')
# Thesholding Valid Data
temp_valid_img = gvalid.getImage(4,'gray')
fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(10,10))
ax1.imshow(convex_resize(temp_valid_img))
ax1.set_title('Without thresholding')
ax2.imshow(thres_resize(temp_valid_img))
ax2.set_title('Thresholding')
fig, ax = plt.subplots(5, 5, figsize=(20,20))
for i in range(5):
    for j in range(5):
        img_index = np.random.randint(0,gdt.n_index)
        ax[i][j].imshow(thres_resize(gdt.getImage(img_index,'gray')))
        ax[i][j].set_title('Class: '+str(gdt.label[img_index]))
        ax[i][j].set_axis_off()
# Apply training set image
X = []

for i, image_path in enumerate(tqdm(gdt.dataFile)):
    X.append(thres_resize(gdt.getImage(i,'gray')))

X = np.array(X)   
y = np.array(gdt.label)
X = X.reshape((-1,32,32,1))
X.shape,y.shape
y_cat = tf.keras.utils.to_categorical(y)
y_cat.shape
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.25, random_state=1234)
X_train = X_train / 255.
X_test = X_test / 255.
def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(6, (5,5), input_shape=(32, 32, 1), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D()) 
    model.add(tf.keras.layers.Conv2D(16, (5,5), activation='relu')) 
    model.add(tf.keras.layers.MaxPool2D()) 
    model.add(tf.keras.layers.Flatten()) 
    model.add(tf.keras.layers.Dense(120, activation='relu'))
    model.add(tf.keras.layers.Dense(84, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model
create_model().summary()
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0000001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=1)
model = create_model()
history = model.fit(X_train, y_train, batch_size=64, validation_data=(X_test,y_test), epochs=100, callbacks=[learning_rate_reduction,early_stop], verbose=0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
# Prepare validation set
print(len(X_valid))

prediction = create_model.predict(X_valid)
label_dt = dt_valid

print(len(prediction))

# Prepare blank object
predict_array = {'id': [],'category': []}
count_error   = 0

id_val = label_dt['id'].values

# Prediction
for index in range(len(prediction)):
    result = np.where(prediction[index] == np.amax(prediction[index]))
    try:
        predict_label = result[0][0]
        predict_id = id_val[index] 
        predict_array['id'].append(predict_id)
        predict_array['category'].append(predict_label)
    except:
        count_error += 1

print(f"found ... {count_error} errors")
predict_dt = pd.DataFrame(predict_array, columns=['id','category'])
# Rule controlling
test_rule_dt = pd.read_csv('/kaggle/input/thai-mnist-classification/test.rules.csv')
test_rule_dt.head()

#test_rule_dt['feature2'][2]
def translate(val):
    lock = predict_dt.iloc[0]['category']
    return lock    
def translate(x):
    try:
        n = predict_dt[predict_dt['id']==x]
        return n['category'].values[0]
    except:
        return 11
test_rule_dt['feauture1'] = test_rule_dt.apply(lambda x:  translate(x['feature1']), axis=1)
test_rule_dt['feauture2'] = test_rule_dt.apply(lambda x:  translate(x['feature2']), axis=1)
test_rule_dt['feauture3'] = test_rule_dt.apply(lambda x:  translate(x['feature3']), axis=1)
test_rule_dt.head()
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
        return math.floor(((f2*f3)/9)*11)
    elif(f1 == 8):
        return math.floor((f3*(f3 +1) - f2*(f2-1))/2)
    elif(f1 == 9):
        return 50 + f2
    else:
        return f2 + f3
test_rule_dt['predict'] = test_rule_dt.apply(lambda x: calculate(x['feauture1'] , x['feauture2'] , x['feauture3']) , axis=1)
test_rule_dt
result = test_rule_dt[['id', 'predict']]
result.head(10)
result.to_csv('submit.csv', index=False)