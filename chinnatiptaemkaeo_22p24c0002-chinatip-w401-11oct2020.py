import os
import cv2
import math 
import pandas as pd
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from skimage import feature
from skimage import measure
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
os.listdir('/kaggle/input/thai-mnist-classification')
train_img_path = '/kaggle/input/thai-mnist-classification/train'
train_label_path = '/kaggle/input/thai-mnist-classification/mnist.train.map.csv'
# THAI-MNIST Labelling
train_mnist_df = pd.read_csv('/kaggle/input/thai-mnist-classification/mnist.train.map.csv')

# prepare droplist
drop_df = pd.read_csv('/kaggle/input/thaimnistdroplist/drop_lists.csv')
drop_lists = drop_df['id'].values

# Remove droplist from label sets
filter_df = train_mnist_df[~train_mnist_df['id'].isin(drop_lists)]
filter_df
# Prepare Validation set
test_folder_path = "/kaggle/input/thai-mnist-classification/test"
test_images = [f for f in os.listdir(test_folder_path) if os.path.isfile(os.path.join(test_folder_path, f))]
valid_array = {'id': [], 'category': []}

for i in range(len(test_images)):
    valid_array['id'].append(test_images[i])
    valid_array['category'].append(0)
    # print(test_images[1])

valid_df = pd.DataFrame(valid_array , columns=['id','category'])
valid_df
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
gdt = getdata(filter_df,group='train')
gdt.dataFile.shape
gvalid = getdata(valid_df, group='test')
gvalid.dataFile.shape
from skimage.morphology import convex_hull_image
from skimage.util import invert
temp_img = invert(gdt.get1img(232,'gray'))
# 
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
def convex_resize(img):
    img = invert(img)
    img = convex_crop(img,pad=20)
    img = cv2.resize(img,(32,32))
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
# Thesholding Train Data
temp_img = gdt.get1img(64,'gray')
fig, [ax1,ax2] = plt.subplots(1, 2,figsize=(10,7))
ax1.imshow(convex_resize(temp_img))
ax1.set_title('Without thresholding')
ax2.imshow(thes_resize(temp_img))
ax2.set_title('Thresholding')
# Thesholding Valid Data
temp_valid_img = gvalid.get1img(4,'gray')
fig, [ax1,ax2] = plt.subplots(1, 2,figsize=(10,7))
ax1.imshow(convex_resize(temp_valid_img))
ax1.set_title('Without thresholding')
ax2.imshow(thes_resize(temp_valid_img))
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
y_cat = tf.keras.utils.to_categorical(y)
y_cat.shape
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.25, random_state=1234)
X_train = X_train / 255.
X_test = X_test / 255.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(6, (5,5), input_shape=(32, 32, 1), activation='relu'))
model.add(tf.keras.layers.MaxPool2D()) 
model.add(tf.keras.layers.Conv2D(16, (5,5), activation='relu')) 
model.add(tf.keras.layers.MaxPool2D()) 
model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dense(120, activation='relu'))
model.add(tf.keras.layers.Dense(84, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
model.summary()
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0000001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10,verbose=1)
history = model.fit(X_train, y_train, batch_size=64,validation_data=(X_test,y_test), epochs=200, callbacks=[learning_rate_reduction,early_stop])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
# Prepare validation set
print(len(X_valid))

prediction = model.predict(X_valid)
label_df = valid_df

print(len(prediction))

# Prepare blank object
predict_array = {'id': [], 'category': []}
count_error   = 0

id_val = label_df['id'].values

# Prediction
for index in range(len(prediction)):
    result = np.where(prediction[index] == np.amax(prediction[index]))
    try:
        predict_label = result[0][0]
        predict_id    = id_val[index] # label_df['id'].values
        predict_array['id'].append(predict_id)
        predict_array['category'].append(predict_label)
    except:
        count_error += 1

print(f"found ... {count_error} errors")
### Download predict CSV
predict_df = pd.DataFrame(predict_array , columns=['id','category'])
predict_df.to_csv('val.csv')
# Rule controlling
test_rule_df = pd.read_csv('/kaggle/input/thai-mnist-classification/test.rules.csv')
test_rule_df.head(5)

test_rule_df['feature2'][2]
def translate(val):
    lock = predict_df.iloc[0]['category']
    return lock    
def translate(x):
    try:
        n = predict_df[predict_df['id']==x]
        return n['category'].values[0]
    except:
        return 11
test_rule_df['feauture1_trans'] = test_rule_df.apply(lambda x:  translate(x['feature1']) , axis=1)
test_rule_df['feauture2_trans'] = test_rule_df.apply(lambda x:  translate(x['feature2']) , axis=1)
test_rule_df['feauture3_trans'] = test_rule_df.apply(lambda x:  translate(x['feature3']) , axis=1)
test_rule_df
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
test_rule_df['predict'] = test_rule_df.apply(lambda x: calculate(x['feauture1_trans'] , x['feauture2_trans'] , x['feauture3_trans']) , axis=1)
test_rule_df
result = test_rule_df[['id', 'predict']]
result.head(10)
result.to_csv('predict_submit.csv', index=False)
