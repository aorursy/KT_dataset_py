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
class getdata():
    def __init__(self,data_path,label_path=None):
        self.dataPath = data_path
        self.dataFile = os.listdir(self.dataPath)
        self.n_index = len(self.dataFile)
        if label_path is not None:
            self.labelPath = label_path
            self.label_dict = pd.read_csv(train_label_path).set_index('id').to_dict()['category']
            self.label = [self.label_dict[x] for x in self.dataFile]
        
        
    
    def get1img(self,img_index,mode='rgb',label = False):
        img = cv2.imread( os.path.join(self.dataPath,self.dataFile[img_index]) )
        if mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == 'gray':
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if label:
            return img,self.label[img_index]
        return img
gdt = getdata(train_img_path,train_label_path)
plt.gray()
from skimage.morphology import convex_hull_image
from skimage.util import invert
temp_img = invert(gdt.get1img(1112,'gray'))
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
temp_img = gdt.get1img(234,'gray')
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
from tqdm import tqdm
X = []
for i in tqdm(range(gdt.n_index)):
    X.append(thes_resize(gdt.get1img(i,'gray')))
X = np.array(X)
y =np.array(gdt.label)
X = X.reshape((-1,32,32,1))
X.shape,y.shape
import tensorflow as tf
y_cat = tf.keras.utils.to_categorical(y)
y_cat.shape
X = X/255.
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
kf.get_n_splits(X)

def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(6, (3,3), input_shape=(32, 32, 1), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D()) 
    model.add(tf.keras.layers.Conv2D(16, (5,5), activation='relu')) 
    model.add(tf.keras.layers.MaxPool2D()) 
    model.add(tf.keras.layers.Flatten()) 
    model.add(tf.keras.layers.Dropout(0.5)) #Add dropout for prevent overfit
    model.add(tf.keras.layers.Dense(120, activation='relu'))
    model.add(tf.keras.layers.Dense(84, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
    return model
create_model().summary()
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0000001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=15,verbose=1)
models = []
for i,(train_index, test_index) in enumerate(kf.split(X)):
    print("Fold number ",i)
    model = create_model()
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_cat[train_index], y_cat[test_index]
    model = create_model()
    history = model.fit(X_train, y_train, batch_size=64,validation_data=(X_test,y_test), 
                    epochs=100, callbacks=[learning_rate_reduction,early_stop],verbose=0)
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    models.append(model)
tdt = getdata('../input/thai-mnist-classification/test')
xtest = []
for i in tqdm(range(tdt.n_index)):
    xtest.append(thes_resize(tdt.get1img(i,'gray')))
xtest = np.array(xtest)
xtest = xtest / 255.
xtest = xtest.reshape((-1,32,32,1))
xtest.shape
preds = []
for model in models :
    pred = model.predict(xtest)
    pred = np.argmax(pred,axis=1)
    preds.append(pred)
preds = np.array(preds)
from scipy import stats
pred = stats.mode(preds,axis=0)[0].reshape(-1)
test_df = pd.DataFrame()
test_df["id"] = tdt.dataFile
test_df["category"] = pred
test_df.head()
map_dict = test_df.set_index('id').to_dict()['category']
test_rule = pd.read_csv('/kaggle/input/thai-mnist-classification/test.rules.csv')
test_rule.head()
def value_search(fn):
    if isinstance(fn,str):
        return map_dict[fn]
    else :
        return fn
test_rule['feature1'] = test_rule['feature1'].map(value_search)
test_rule['feature2'] = test_rule['feature2'].map(value_search)
test_rule['feature3'] = test_rule['feature3'].map(value_search)
test_rule.head()
def pred_map(feature):
    f1,f2,f3 = feature[0],feature[1],feature[2]
    if np.isnan(f1) :
        return (f2+f3)%99
    if f1 == 0 :
        return (f2*f3)%99
    if f1 == 1:
        return np.abs(f2-f3)%99
    if f1 ==2:
        return ((f2+f3)*np.abs(f2-f3))%99
    if f1 == 3:
        return (np.abs((f3*(f3+1)) - (f2*(f2-1))) / 2) % 99
    if f1 == 4:
        return (50+f2-f3) % 99
    if f1 == 5:
        return min(f2,f3)%99
    if f1 == 6 :
        return max(f2,f3) % 99
    if f1 == 7 :
        return ( ((f2*f3) % 9) *11 ) % 99
    if f1 == 8 :
        return( ((f2**2)+1)*f2 + f3*(f3+1)) % 99
    if f1 ==9 :
        return (f2+50) % 99
test_rule['predict'] = test_rule[['feature1','feature2','feature3']].apply(pred_map,axis=1)
test_rule.head()
test_rule.tail()
out_df = test_rule[['id','predict']]
out_df.head()
out_df.to_csv('out_submit.csv',index=False)
for i,model in enumerate(models):
    model.save('model_'+str(i)+'.h5')
import pickle
pickle.dump([[X,y],xtest], open('thaiMnist_processed.pickle','wb'))
