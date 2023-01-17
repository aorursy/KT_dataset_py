import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from IPython.display import display
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
df_train = pd.read_csv('../input/thai-mnist-classification/mnist.train.map.csv')
df_train.tail(5)
image_size = (32, 32)
# เช็คตำแหน่งตัวการเขียตัวอักษร
def chk_location(img):
    for i in range(img.shape[0]):
        k = 0
        for l in range(3):
            if np.sum(1 - img[i,:,l]) > 0 :
                k += 1
        if k > 1 :
            h1 = i
            break
    for i in range(img.shape[0] - 1 ,1,-1):
        k = 0
        for l in range(3):
            if np.sum(1- img[i,:,l]) > 0 :
                k += 1
        if k > 1 :
            h2 = i
            break
    for i in range(img.shape[1]):
        k = 0
        for l in range(3):
            if np.sum(1- img[:,i,l]) > 0 :
                k += 1
        if k > 1 :
            w1 = i
            break
    for i in range(img.shape[0]- 1 ,1,-1):
        k = 0
        for l in range(3):
            if np.sum(1- img[:,i,l]) > 0 :
                k += 1
        if k > 1 :
            w2 = i
            break
    h = h2 - h1
    w = w2 - w1
    if h > w:
        if w1 + h <= 620:
            return h1,h1 + h ,w1, w1+h
        elif w2 - h > 1:
            return h1,h1 + h ,w2 - h , w2
        else:
            return 0, 620, 0, 620
    else:
        if h1 + w <= 620:
            return h1, h1 + w ,w1, w1 + w
        elif h2 - w >1:
            return h2 - w, h2, w1 , w1 +w
        else:
            return 0, 620, 0, 620
def get_img(ds):
    ims_X_train = []
    for i,value in enumerate(ds):
        print(i)
        image = tf.keras.preprocessing.image.load_img('../input/thai-mnist-classification/train/'+value,
                                           target_size=(620, 620),
                                           interpolation='nearest')
        image = keras.preprocessing.image.img_to_array(image)
        h1,h2,w1,w2 = chk_location(image/ 255.)
        ims_X_train.append(tf.keras.preprocessing.image.smart_resize(image[h1:h2, w1:w2,:], image_size, interpolation='bilinear')/255.)
    return ims_X_train
y_train_DS = np.array(df_train[['category']])
X_train_DS = np.array(get_img(df_train['id']))
import seaborn as sns
sns.countplot(df_train['category']) 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
vgg = VGG16(include_top=False, weights='imagenet', input_shape=(32, 32, 3))

# fit output
x = layers.Flatten()(vgg.output)
x = layers.Dense(4096,activation="hard_sigmoid")(x)
x = layers.Dense(4096,activation="relu")(x)
x = layers.Dense(4096,activation="sigmoid")(x)
x = layers.Dense(4096,activation="relu")(x)
x = layers.Dense(4096,activation="relu")(x)
x = layers.Dense(4096,activation="relu")(x)
x = layers.Dense(4096,activation="sigmoid")(x)
x = layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(vgg.input, x)
model.summary()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.000001), 
              loss=keras.losses.sparse_categorical_crossentropy)
history = model.fit(X_train_DS, y_train_DS,epochs=200)
Ztest = model.predict(X_train_DS)
np.sum(Ztest.argmax(axis=1) == y_train_DS.reshape(1,-1)[0])/len(Ztest)
Xtest =[]
Xfilename =[]
for dirname, _, filenames in os.walk('../input/thai-mnist-classification/test'):
    for filename in filenames:
        Xfilename.append(filename)
        image = tf.keras.preprocessing.image.load_img('../input/thai-mnist-classification/test/'+filename,
                                          target_size=(620,620),
                                          interpolation='nearest')
        image = keras.preprocessing.image.img_to_array(image)
        h1,h2,w1,w2 = chk_location(image/ 255.)
        Xtest.append(tf.keras.preprocessing.image.smart_resize(image[h1:h2, w1:w2,:], image_size, interpolation='bilinear')/255.)      
Xtest = np.array(Xtest)

ytest = model.predict(Xtest).argmax(axis=1)
ytest
Xfilename = np.array(Xfilename)
Xfilename.shape
test_ds = pd.DataFrame({'id':Xfilename,'category':ytest})
test_ds.to_csv("export008.csv")
test_ds.head
train_map_df = pd.read_csv('../input/thai-mnist-classification/mnist.train.map.csv')
train_rules_df = pd.read_csv('../input/thai-mnist-classification/train.rules.csv')
train_rules_df.head()
for i,value in enumerate(train_rules_df['feature1']):
    temp = np.array(train_map_df[train_map_df['id'] == value]['category'])
    if len(temp) == 1:
        train_rules_df.iloc[i, train_rules_df.columns.get_loc('feature1')] = temp
    else:
        train_rules_df.iloc[i, train_rules_df.columns.get_loc('feature1')] = 999
for i,value in enumerate(train_rules_df['feature2']):
    temp = np.array(train_map_df[train_map_df['id'] == value]['category'])
    if len(temp) == 1:
        train_rules_df.iloc[i, train_rules_df.columns.get_loc('feature2')] = temp
for i,value in enumerate(train_rules_df['feature3']):
    temp = np.array(train_map_df[train_map_df['id'] == value]['category'])
    if len(temp) == 1:
        train_rules_df.iloc[i, train_rules_df.columns.get_loc('feature3')] = temp
train_rules_df['predict-1'] =0
train_rules_df
for i in range(len(train_rules_df)):
    f1 = train_rules_df['feature1'].iloc[i]
    f2 = train_rules_df['feature2'].iloc[i]
    f3 = train_rules_df['feature3'].iloc[i]
    train_rules_df['predict-1'].iloc[i] = f2 + f3
    if f1 == 0:
        train_rules_df['predict-1'].iloc[i] = f2 * f3
    
    elif f1 == 1:
        train_rules_df['predict-1'].iloc[i] = np.abs(f2 - f3)
        
    elif f1 == 2:
        train_rules_df['predict-1'].iloc[i] = (f2 + f3) * np.abs(f2 - f3)
        
    elif f1 == 3:
        train_rules_df['predict-1'].iloc[i] = np.abs((f3 * (f3 + 1) - f2 * (f2 - 1))/2)
        
    elif f1 == 4:
        train_rules_df['predict-1'].iloc[i] = 50 + (f2 - f3)
        
    elif f1 == 5:
        train_rules_df['predict-1'].iloc[i] = min(f2,f3)
        
    elif f1 == 6:
        train_rules_df['predict-1'].iloc[i] = max(f2,f3)
        
    elif f1 == 7:
        train_rules_df['predict-1'].iloc[i] = ((f2 * f3) % 9) *11
        
    elif f1 == 8:
        train_rules_df['predict-1'].iloc[i] = ((((f2 ** 2) + 1) * f2) + (f3 * (f3 + 1)) % 99)%99
        
    elif f1 == 9:
        train_rules_df['predict-1'].iloc[i] = 50 + f2
train_rules_df['predict-1'] = train_rules_df['predict-1'].astype('int64')
train_rules_df
train_rules_df[train_rules_df['predict'] != train_rules_df['predict-1']]
test_rules_ds = pd.read_csv('../input/thai-mnist-classification/test.rules.csv') 
test_rules_ds
test_ds
for i,value in enumerate(test_rules_ds['feature1']):
    temp = np.array(test_ds[test_ds['id'] == value]['category'])
    if len(temp) == 1:
        test_rules_ds.iloc[i, test_rules_ds.columns.get_loc('feature1')] = temp
    else:
        test_rules_ds.iloc[i, test_rules_ds.columns.get_loc('feature1')] = 999
test_rules_ds
for i,value in enumerate(test_rules_ds['feature2']):
    temp = np.array(test_ds[test_ds['id'] == value]['category'])
    if len(temp) == 1:
        test_rules_ds.iloc[i, test_rules_ds.columns.get_loc('feature2')] = temp
test_rules_ds
for i,value in enumerate(test_rules_ds['feature3']):
    temp = np.array(test_ds[test_ds['id'] == value]['category'])
    if len(temp) == 1:
        test_rules_ds.iloc[i, test_rules_ds.columns.get_loc('feature3')] = temp
test_rules_ds['predict'] = 0
test_rules_ds
test_rules_ds
for i in range(len(test_rules_ds)):
    f1 = test_rules_ds['feature1'].iloc[i]
    f2 = test_rules_ds['feature2'].iloc[i]
    f3 = test_rules_ds['feature3'].iloc[i]
    test_rules_ds['predict'].iloc[i] = f2 + f3
    if f1 == 0:
        test_rules_ds['predict'].iloc[i] = f2 * f3
    
    elif f1 == 1:
        test_rules_ds['predict'].iloc[i] = np.abs(f2 - f3)
        
    elif f1 == 2:
        test_rules_ds['predict'].iloc[i] = (f2 + f3) * np.abs(f2 - f3)
        
    elif f1 == 3:
        test_rules_ds['predict'].iloc[i] = np.abs((f3 * (f3 + 1) - f2 * (f2 - 1))/2)
        
    elif f1 == 4:
        test_rules_ds['predict'].iloc[i] = 50 + (f2 - f3)
        
    elif f1 == 5:
        test_rules_ds['predict'].iloc[i] = min(f2,f3)
        
    elif f1 == 6:
        test_rules_ds['predict'].iloc[i] = max(f2,f3)
        
    elif f1 == 7:
        test_rules_ds['predict'].iloc[i] = ((f2 * f3) % 9) *11
        
    elif f1 == 8:
        test_rules_ds['predict'].iloc[i] = ((((f2 ** 2) + 1) * f2) + (f3 * (f3 + 1)) % 99)%99
        
    elif f1 == 9:
        test_rules_ds['predict'].iloc[i] = 50 + f2
test_rules_ds['predict'] = test_rules_ds['predict'].astype('int64')
test_rules_ds
submit_ds  = pd.read_csv('../input/thai-mnist-classification/submit.csv')
submit_ds
for i,value in enumerate(submit_ds['id']):
    temp = np.array(test_rules_ds[test_rules_ds['id'] == value]['predict'])
    if len(temp) == 1:
        submit_ds.iloc[i, submit_ds.columns.get_loc('predict')] = temp
submit_ds['predict'].notnull()
submit_ds.to_csv('submission999.csv', index=False)