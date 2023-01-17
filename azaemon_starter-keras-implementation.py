from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import os
import cv2
import keras
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.models import load_model  
from keras.utils import plot_model
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Softmax,Activation,Dense,Dropout
from keras.callbacks import Callback,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,auc
#from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
import pickle
from skimage import measure
from skimage import morphology
from skimage.transform import resize
from sklearn.cluster import KMeans
def split_target_dir(target_dir,output_dir):
    target_list=[target_dir+os.sep+file for file in os.listdir(target_dir)]
    for target in target_list:
        img_split=split_lung_parenchyma(target,15599,-96)
        dst=target.replace(target_dir,output_dir)
        dst_dir=os.path.split(dst)[0]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        cv2.imencode('.jpg', img_split)[1].tofile(dst)
    print(f'Target list done with {len(target_list)} items')
    
def split_lung_parenchyma(target,size,thr):
    img=cv2.imdecode(np.fromfile(target,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    try:
        img_thr= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,size,thr).astype(np.uint8)
    except:
        img_thr= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,999,thr).astype(np.uint8)
    img_thr=255-img_thr
    img_test=measure.label(img_thr, connectivity = 1)
    props = measure.regionprops(img_test)
    img_test.max()
    areas=[prop.area for prop in props]
    ind_max_area=np.argmax(areas)+1
    del_array = np.zeros(img_test.max()+1)
    del_array[ind_max_area]=1
    del_mask=del_array[img_test]
    img_new = img_thr*del_mask
    mask_fill=fill_water(img_new)
    img_new[mask_fill==1]=255
    img_out=img*~img_new.astype(bool)
    return img_out

def fill_water(img):
    copyimg = img.copy()
    copyimg.astype(np.float32)
    height, width = img.shape
    img_exp=np.zeros((height+20,width+20))
    height_exp, width_exp = img_exp.shape
    img_exp[10:-10, 10:-10]=copyimg
    mask1 = np.zeros([height+22, width+22],np.uint8)   
    mask2 = mask1.copy()
    mask3 = mask1.copy()
    mask4 = mask1.copy()
    cv2.floodFill(np.float32(img_exp), mask1, (0, 0), 1) 
    cv2.floodFill(np.float32(img_exp), mask2, (height_exp-1, width_exp-1), 1) 
    cv2.floodFill(np.float32(img_exp), mask3, (height_exp-1, 0), 1) 
    cv2.floodFill(np.float32(img_exp), mask4, (0, width_exp-1), 1)
    mask = mask1 | mask2 | mask3 | mask4
    output = mask[1:-1, 1:-1][10:-10, 10:-10]
    return output
def normal(X):
  norm= np.linalg.norm(X)
  n= X/norm
  return n
def read_image(target_dir):
    x = cv2.imread(target_dir)
    x = cv2.resize(x,(200,200))
    global i
    print(i)
    i=i+1
    return x
i=1;
def predict_comparision(y_predict,y_test):
    tp,tn,fp,fn = 0,0,0,0
    y_predict_index = np.argmax(y_predict,axis = 1)
    y_test_index = np.argmax(y_test,axis = 1)
    m = len(y_predict_index)
    for i in range(m):
        if y_predict_index[i] == 0:
            if y_test_index[i]==0:
                tp +=1    
            else:
                fp +=1
        else:
            if y_test_index[i]==1:
                tn +=1
            else:
                fn += 1 
    return tp,tn,fp,fn
def VGG_Simple():
    model=Sequential()
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(200,200,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',kernel_initializer='uniform',activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(16,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(16,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3,activation='softmax'))
    return model
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=()):
        super(Callback, self).__init__()
        self.x_val,self.y_val = validation_data
    def on_epoch_end(self, epoch, log={}):
        y_pred = self.model.predict(self.x_val)
        AUC1 = roc_auc_score(self.y_val[:,0], y_pred[:,0])
        AUC2 = roc_auc_score(self.y_val[:,1], y_pred[:,1])
        AUC3 = roc_auc_score(self.y_val[:,2], y_pred[:,2])
        print('val_AUC NiCT epoch:%d: %.6f' % (epoch+1, AUC1))
        print('val_AUC pCT epoch:%d: %.6f' % (epoch+1, AUC2))
        print('val_AUC nCT epoch:%d: %.6f' % (epoch+1, AUC3))
        print()
tr1='../input/preprocessed-ct-scans-for-covid19/CT scans/NiCT/'
tr2='../input/preprocessed-ct-scans-for-covid19/CT scans/pCT/'
tr3='../input/preprocessed-ct-scans-for-covid19/CT scans/nCT/'
tl1=[tr1+file for file in os.listdir(tr1)]
tl2=[tr2+file for file in os.listdir(tr2)]
tl3=[tr3+file for file in os.listdir(tr3)]
target_list= tl1+tl2+tl3
print("The number of non-informative images: ",len(tl1))
print("The number of positive images: ",len(tl2))
print("The number of negative images: ",len(tl3))
fig= plt.figure(figsize=(20,10))
index= tl1[0]
a= fig.add_subplot(1,3,1)
a.set_title('Non-informative Image')
plt.imshow(plt.imread(index))

index= tl2[0]
a= fig.add_subplot(1,3,2)
a.set_title('Positive Image')
plt.imshow(plt.imread(index))

index= tl3[0]
a= fig.add_subplot(1,3,3)
a.set_title("Negative Image")
plt.imshow(plt.imread(index))
##Skip this cell if you want to use the pickle object for data loading

y_list=to_categorical(np.concatenate(np.array([[0]*len(tl1),
                                               [1]*len(tl2),
                                               [2]*len(tl3)])),3)
X=np.array([read_image(file) for file in target_list])
f= '../input/pickle-file-of-ct-scans/train_lung.pickle'
with open(f, 'rb') as file: 
    X_train,X_val,y_train,y_val=pickle.load(file)

print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)

X_train= normal(X_train)
X_val= normal(X_val)
#X_train, X_val, y_train, y_val = train_test_split(X, y_list, test_size=0.2, stratify=y_list)
checkpoint = ModelCheckpoint('vggnormal.model',save_weights_only = False, monitor='val_loss', verbose=1,save_best_only=True,mode='auto',period=1)
RocAuc = RocAucEvaluation(validation_data=(X_val,y_val))
model=VGG_Simple()
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val),batch_size=64, callbacks=[checkpoint,RocAuc],verbose=1)
