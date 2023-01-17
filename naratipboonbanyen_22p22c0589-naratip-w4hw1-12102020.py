# 22p22c0589_Naratip_W4HW1_12102020
# Thai Mnist
# re-write/adapt model crop/thes
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import cv2
from skimage import feature
from skimage import measure
os.listdir('/kaggle/input/thai-mnist-classification')
train_img_path = '/kaggle/input/thai-mnist-classification/train'
train_label_path = '/kaggle/input/thai-mnist-classification/mnist.train.map.csv'
pd.read_csv(train_label_path)
dx = pd.read_csv(train_label_path)
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
gdt = getdata(train_img_path,train_label_path)
gdt.get1img(1,'rgb').shape
plt.gray()
from skimage.morphology import convex_hull_image
from skimage.util import invert
temp_img = invert(gdt.get1img(234,'gray'))
fig, [ax1,ax2] = plt.subplots(1, 2)

ax1.imshow(temp_img)

cvh =  convex_hull_image(temp_img)
ax2.imshow(cvh)
def convex_crop(img,pad=50):
    convex = convex_hull_image(img)
    r,c = np.where(convex)
    while (min(r)-pad < 0) or (max(r)+pad > img.shape[0]) or (min(c)-pad < 0) or (max(c)+pad > img.shape[1]):
        pad = pad - 1
    return img[min(r)-pad:max(r)+pad,min(c)-pad:max(c)+pad]
crop_img = convex_crop(temp_img,pad=100)
plt.imshow(crop_img)
def convex_resize(img):
    img = invert(img)
    img = convex_crop(img,pad=80)
    img = cv2.resize(img,(224,224))
    return img
def thes_resize(img,thes=10):
    img = invert(img)
    img = convex_crop(img,pad=80)
    
    img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 300):
        img = cv2.resize(img,(300,300))
        img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 224):
        img = cv2.resize(img,(224,224))
        img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(224,224))
    img = ((img > thes)*255).astype(np.uint8)
#     img = cv2.resize(img,(50,50))
#     img = ((img > thes)*255).astype(np.uint8)
#     img = cv2.resize(img,(32,32))
#     img = ((img > thes)*255).astype(np.uint8)
    return img
temp_img = gdt.get1img(64,'gray')
fig, [ax1,ax2] = plt.subplots(1, 2,figsize=(10,7))
ax1.imshow(convex_resize(temp_img))
ax1.set_title('Without thresholding')
ax2.imshow(thes_resize(temp_img))
ax2.set_title('Thresholding')
convex_resize(temp_img).shape
fig, ax = plt.subplots(5, 5, figsize=(15,15))
for i in range(5):
    for j in range(5):
        img_index = np.random.randint(0,gdt.n_index)
        ax[i][j].imshow(thes_resize(gdt.get1img(img_index,'gray')))
        ax[i][j].set_title('Class: '+str(gdt.label[img_index]))
        ax[i][j].set_axis_off()
gdt.n_index
X = []
for i in range(gdt.n_index):
    X.append(thes_resize(gdt.get1img(i,'gray')))
    if (i+1) % 100 == 0:
        print(i)
        
X = np.array(X)
y = gdt.label
X = X.reshape((-1,224,224,1))
X.shape,y.shape
import tensorflow as tf
y_cat = tf.keras.utils.to_categorical(y)
y_cat.shape
y_cat
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.33)
X_train = X_train / 255.
X_test = X_test / 255.
len(X_train)
import tensorflow as tf
from tensorflow import keras

resnet = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)
resnet.summary()

IMAGE_WIDTH=224
IMAGE_HEIGHT=224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=1
batch_size = 64
from tensorflow.keras import layers, Model
x_in = layers.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)) # สร้าง input layer
# เอา x_in เข้า conv2D
x = layers.Conv2D(3, 1)(x_in) # เอา conv2D ใส่ filter 3 filter ทำให้ grey scale มี 3 channel และ ขนาด 1*1 เพื่อไม่เปลี่ยนค่า
# x = 32,32,3 ขนาด 32*32 /3 filter
x = resnet(x)
# fit output

x = layers.Flatten()(x) # เอา x เข้า flatten
x = layers.Dense(10, activation='softmax')(x) # เอา x เข้า dense
model = Model(x_in, x)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics='acc')
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(monitor='val_acc', patience=10,verbose=1)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.1, 
                                            min_lr=0.0000001)

callbacks = [earlystop, learning_rate_reduction]
history = model.fit(X_train, 
                    y_train, 
                    batch_size=batch_size,
                    validation_data=(X_test,y_test),
                    epochs=50, 
                    callbacks=callbacks)
# summarize history for accuracy
plt.figure(figsize=(12,6))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.figure(figsize=(12,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model.save_weights("incepResnew.h5")
from keras.models import load_model
 
# load weight model
# โหลดโมเดลที่ train แล้วมาใช้ 

# history = model.load_weights('../input/modelh5/incepResnew.h5')    
# history = model.load_weights('../input/resold/incepRes.h5') 

class gettestdata():
    def __init__(self,data_path,label_list):
        
        self.dataPath = data_path
        self.labelPath = label_list
        self.label_df = label_list
        
        self.dataFile = self.label_df['id'].values
#         self.label = self.label_df['category'].values
        self.n_index = len(self.dataFile)
        
    
    def get1img(self,img_index,mode='rgb',label = False):
        img = cv2.imread(os.path.join(self.dataPath,self.label_df.iloc[img_index]['id']) )
        if mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == 'gray':
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
        if label:
            return img,self.label_df.iloc[img_index]['category']
        
        return img
test_filenames = os.listdir("../input/thai-mnist-classification/test")
test_df = pd.DataFrame({
    'id': test_filenames
})
nb_samples = test_df.shape[0]
test_df
test_img_path = '/kaggle/input/thai-mnist-classification/test'
# test_label_path = '/kaggle/input/thai-mnist-classification/test.rules.csv'
gdt_test = gettestdata(test_img_path, test_df)
test = []
for i in range(gdt_test.n_index): # gdt_test.n_index
    test.append(thes_resize(gdt_test.get1img(i,'gray')))
    if (i+1) % 100 == 0:
        print(i)
test = np.array(test)
test = test/255.
Ztest = model.predict(test)
test_df['category'] = np.argmax(Ztest, axis=-1)
test_df
test_df.to_csv ('./m_val.csv', index = False)
df1 = pd.read_csv("../input/thai-mnist-classification/test.rules.csv")
df1
df_label1 = test_df
df_label1 = df_label1.rename(columns={"id": "feature1", "category": "label1"})

df_label2 = test_df
df_label2 = df_label2.rename(columns={"id": "feature2", "category": "label2"})

df_label3 = test_df
df_label3 = df_label3.rename(columns={"id": "feature3", "category": "label3"})

df_clogic = df1
df_clogic["feature1"] = df_clogic["feature1"].fillna('99')
df_clogic
df_clogic2 = df_clogic.merge(df_label1, on ='feature1', how='left')
df_clogic2 = df_clogic2.merge(df_label2, on ='feature2', how='left')
df_clogic2 = df_clogic2.merge(df_label3, on ='feature3', how='left')
df_clogic2["label1"] = df_clogic2["label1"].fillna(99)
df_clogic2 = df_clogic2[["id","label1","label2","label3","predict"]]

df_clogic2['label1'] = df_clogic2['label1'].astype(int)
df_clogic2['label2'] = df_clogic2['label2'].astype(int)
df_clogic2['label3'] = df_clogic2['label3'].astype(int)
df_clogic2
df_clogic2.info()
import math 

def f_0(f2,f3):
    return f2*f3

def f_1(f2,f3):
    return math.sqrt((f2-f3)**2)
    
def f_2(f2,f3):
    return (f2+f3) * math.sqrt((f2-f3)**2)
    
def f_3(f2,f3):
    return math.sqrt((((f3*(f3+1))-(f2*(f2-1)))/2)**2)
    
def f_4(f2,f3):
    return 50+(f2-f3)
    
def f_5(f2,f3):
    return min(f2,f3)
    
def f_6(f2,f3):
    return max(f2,f3)
    
def f_7(f2,f3):
    return ((f2*f3)%9)*11

def f_8(f2,f3):
    
    result = (((f2**2)+1)*f2)+((f3)*(f3+1))
    result = f_8re(result)
    
    return result
    
def f_8re(result):  
    if result > 99:
        result = result - 99
        result = f_8re(result)
        
    return result

    
def f_9(f2,f3):
    return 50+f2
    
def f_99(f2,f3):
    return f2+f3
    

def controller(f1,f2,f3):
    if f1 == 99:
        return f_99(f2,f3)
    
    elif f1 == 0:
        return f_0(f2,f3)
    
    elif f1 == 1:
        return f_1(f2,f3)
    
    elif f1 == 2:
        return f_2(f2,f3)
    
    elif f1 == 3:
        return f_3(f2,f3)
    
    elif f1 == 4:
        return f_4(f2,f3)
    
    elif f1 == 5:
        return f_5(f2,f3)
    
    elif f1 == 6:
        return f_6(f2,f3)
    
    elif f1 == 7:
        return f_7(f2,f3)
    
    elif f1 == 8:
        return f_8(f2,f3)
    
    elif f1 == 9:
        return f_9(f2,f3)
    

df_clogic2.info()
ans = []
for i in range(0,len(df_clogic2)):
    ans.append(int(controller(df_clogic2["label1"][i],df_clogic2["label2"][i],df_clogic2["label3"][i])))
df_clogic2["ans"] = ans
submit = pd.read_csv("../input/thai-mnist-classification/submit.csv")
submit
val = df_clogic2[['id','ans']]
val = val.rename(columns={"ans": "predict"})
val
print("min: {0}".format(min(val["predict"])))
print("max: {0}".format(max(val["predict"])))
val.to_csv ('./submit.csv', index = False)
