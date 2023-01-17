import os
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from skimage.morphology import convex_hull_image
from skimage.util import invert

from sklearn.model_selection import KFold, train_test_split

import tensorflow as tf
from tensorflow.keras import Sequential, metrics
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Flatten, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
DATA_PATH = "../input/thai-mnist-classification/"

TRAIN_PATH = DATA_PATH + "train/"
TEST_PATH = DATA_PATH + "test/"
seed = 42

BATCH_SIZE = 128
IMAGE_SIZE = 32
train_csv = pd.read_csv("../input/thai-mnist-classification/mnist.train.map.csv")
# train_csv.index = train_csv.id
# train_csv = train_csv.drop(['id'], axis=1)
def convex_crop(img,pad=20):
    convex = convex_hull_image(img)
    r,c = np.where(convex)
    while (min(r)-pad < 0) or (max(r)+pad > img.shape[0]) or (min(c)-pad < 0) or (max(c)+pad > img.shape[1]):
        pad = pad - 1
    return img[min(r)-pad:max(r)+pad,min(c)-pad:max(c)+pad]
def convex_resize(img):
    img = invert(img)
    img = convex_crop(img,pad=20)
    img = cv2.resize(img,(32,32))
    return img
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
gdt = getdata(train_csv,group='train')
gdt.dataFile.shape
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
temp_img = gdt.get1img(24,'gray')
fig, [ax1,ax2] = plt.subplots(1, 2,figsize=(10,7))
ax1.imshow(convex_resize(temp_img))
ax1.set_title('Without thresholding')
ax2.imshow(thes_resize(temp_img))
ax2.set_title('Thresholding')
### Apply training set image
X = []

for i , image_path in enumerate(tqdm(gdt.dataFile)):
    X.append(thes_resize(gdt.get1img(i,'gray')))

X = np.array(X)   
y = gdt.label
X = X.reshape((-1,32,32,1))
X.shape,y.shape
y = gdt.label
y = to_categorical(y, num_classes=10)
print(y[0])
dump_img = pickle.dump(X, open( "train_image.p", "wb" ))
load_img = pickle.load(open( "train_image.p", "rb" ))
with tf.device('/device:GPU:0'):                # Initialize process to GPU
    
    def build_model():
        model = Sequential()
        
#         model.add(Conv2D(IMAGE_SIZE, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
#         model.add(MaxPooling2D(pool_size=2))
        
#         #1st Layer
#         model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
#         model.add(BatchNormalization())
#         model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
#         model.add(BatchNormalization())
        
        model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=2))


#         #1st Layer
#         model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
#         model.add(BatchNormalization())
#         model.add(MaxPooling2D(pool_size=2))
#         model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
#         model.add(BatchNormalization())
#         model.add(MaxPooling2D(pool_size=2))
        
        
        model.add(Flatten())
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        
        model.add(Dense(64, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        
        model.add(Dense(32, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        
        model.add(Dense(10, activation=tf.nn.softmax))
       
        return model

    model = build_model()
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    kfold = KFold(n_splits=10, random_state=seed, shuffle=False)
    val_acc = []
    for count, (train_index, valid_index) in enumerate(kfold.split(X)):
    #     x_train = X[train_index]
    #     y_train = Y[train_index]
    #     x_valid = X[valid_index]
    #     y_valid = Y[valid_index]
    #     earlystopper = EarlyStopping(patience=10, mode=max, monitor='val_val_accuracy', verbose=1)
        model.fit(X[train_index], y[train_index], 
                  validation_data=(X[valid_index], y[valid_index]),
                  batch_size=BATCH_SIZE,
                  epochs=20,
                  verbose=0)
        print(count+1)
        print('========= Genearator Model =========')
        results_train = model.evaluate(X[train_index], y[train_index])
        results_valid = model.evaluate(X[valid_index], y[valid_index])
        print('Train loss :', results_train[0])
        print('Train accuracy :', results_train[1])
        print('Valid loss :', results_valid[0])
        print('Valid accuracy :', results_valid[1])
        val_acc.append(results_valid[1])
        print('\n\n')

    model.save('./KFold_basic.h5')
model.summary()
model.evaluate(X,y)
# with tf.device('/device:GPU:0'):                # Initialize process to GPU
    
#     def build_model():
#         model = Sequential()
        
#         model.add(Conv2D(IMAGE_SIZE, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
#         model.add(MaxPooling2D(pool_size=2))
        
# #         #1st Layer
# #         model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
# #         model.add(BatchNormalization())
# #         model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
# #         model.add(BatchNormalization())
        
#         model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform'))
# #         model.add(BatchNormalization())
#         model.add(MaxPooling2D(pool_size=2))
#         model.add(Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_uniform'))
# #         model.add(BatchNormalization())
#         model.add(MaxPooling2D(pool_size=2))


# #         #1st Layer
# #         model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
# #         model.add(BatchNormalization())
# #         model.add(MaxPooling2D(pool_size=2))
# #         model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
# #         model.add(BatchNormalization())
# #         model.add(MaxPooling2D(pool_size=2))
        
        
#         model.add(Flatten())
#         model.add(Dense(128, activation=tf.nn.relu))
# #         model.add(Dropout(0.2))
#         model.add(Dense(10, activation=tf.nn.softmax))
       
#         return model

#     model = build_model()
#     model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
#     X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)
#     val_acc = []
#     model.fit(X_train, y_train, 
#               validation_data=(X_test, y_test),
#               batch_size=BATCH_SIZE,
#               epochs=200,
#               verbose=1)
# #         print(count+1)
# #         print('========= Genearator Model =========')
# #         results_train = model.evaluate(train_img[train_index], label[train_index])
# #         results_valid = model.evaluate(train_img[valid_index], label[valid_index])
# #         print('Train loss :', results_train[0])
# #         print('Train accuracy :', results_train[1])
# #         print('Valid loss :', results_valid[0])
# #         print('Valid accuracy :', results_valid[1])
# #         val_acc.append(results_valid[1])
# #         print('\n\n')

#     model.save('./Split_basic.h5')
split_basic = load_model("./KFold_basic.h5")
split_basic.evaluate(X,y)
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
gvalid = getdata(valid_df,group='test')
gvalid.dataFile.shape
print(gvalid.dataFile)
print(gvalid.group)
print("../input/thai-mnist-classification/test/0003ce53-7b1b-45c6-86e1-9df2ad0c8cb6.png")
### Apply training set image
X_valid = []

for i , image_path in enumerate(tqdm(gvalid.dataFile)):
    X_valid.append(thes_resize(gvalid.get1img(i,'gray')))

X_valid = np.array(X_valid)   
X_valid = X_valid.reshape((-1,32,32,1))
X_valid.shape
# Prepare validation set
print(len(X_valid))

prediction = split_basic.predict(X_valid)
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
def heavenly_predict(c, x ,y):
    re = 0
    if c == 11:
        re = x+y
    elif c == 0:
        re = x*y
    elif c == 1:
        re = abs(x - y)
    elif c == 2:
        re = (x + y)*abs(x - y)
    elif c == 3:
        re = abs(((x * (y+1)) - x*(x-1))/2)
    elif c == 4:
        re = 50+(x - y)
    elif c == 5:
        re = min(x, y)
    elif c == 6:
        re = max(x, y)
    elif c == 7:
        re = ((x * y)%9)*11
    elif c == 8:
        re = ((((x**2)+1)*x) + (y*(y+1)))
        re = re % 99
    elif c == 9:
        re = 50+x

    return int(re)
import math
test_rule_df['predict'] = test_rule_df.apply(lambda x: heavenly_predict(x['feauture1_trans'] , x['feauture2_trans'] , x['feauture3_trans']) , axis=1)
test_rule_df
result = test_rule_df[['id', 'predict']]
result.head(10)
result.to_csv('predict_submission(2).csv', index=False)