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
import os,shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import pydicom as dicom
from skimage.transform import resize
import cv2
import seaborn as sns
sns.set_style('darkgrid')
os.listdir('/kaggle/input/rsna-pneumonia-detection-challenge')
df=pd.read_csv('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
df.head()
df['path']='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/'+df['patientId'].astype(str)+'.dcm'
df['path'][0]
df.info()
negative=df[df['Target']==0]
print(len(negative))
negative.head()
positive=df[df['Target']==1]
unique_positive=positive[['path','patientId']]
path=unique_positive['path'].unique()
patientId=unique_positive['patientId'].unique()
unique_positive=pd.DataFrame({'path':path,'patientId':patientId})
len(unique_positive)
os.mkdir('/kaggle/working/data')

os.mkdir('/kaggle/working/data/positive')

os.mkdir('/kaggle/working/data/negative')
os.chdir('/kaggle/working')
for _,row in tqdm(unique_positive.iterrows()):
    img=dicom.read_file(row['path']).pixel_array
    img=resize(img,(256,256))
    plt.imsave('data/positive/'+row['patientId']+'.jpg',img,cmap='gray')
for _,row in tqdm(negative.iterrows()):
    img=dicom.read_file(row['path']).pixel_array
    img=resize(img,(256,256))
    plt.imsave('data/negative/'+row['patientId']+'.jpg',img,cmap='gray')
plt.figure(figsize=(30,20))
for j,img in enumerate(os.listdir('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images')):
    path=os.path.join('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images',img)
    tar=df[df['path']==path]['Target'].values[0]
    img=dicom.read_file(path).pixel_array
    plt.subplot(4,4,j+1)
    plt.axis('off')
    if tar==0:
        plt.title('Negative')
    else:
        plt.title('Positive')
        
        s=df[df['path']==path]
        
        for _,row in s.iterrows():
            x=int(row['x'])
            y=int(row['y'])
            w=int(row['width'])
            h=int(row['height'])
            cv2.rectangle(img,(x,y),(x+h,y+h),(255,255,0),5)
    plt.imshow(img,cmap='gray')
    if(j==15):
        break
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True,horizontal_flip=True,
                          width_shift_range=0.05,rescale=1/255,fill_mode='nearest',height_shift_range=0.05,
                           preprocessing_function=preprocess_input,validation_split=0.3,
                          )
train=datagen.flow_from_directory('data',color_mode='rgb',batch_size=128,class_mode='binary',subset='training')
test=datagen.flow_from_directory('data',color_mode='rgb',batch_size=32,class_mode='binary',subset='validation')
train.class_indices
pre_trained_model = VGG16(input_shape = (256,256,3), 
                                include_top = False, 
                                weights = 'imagenet')

for layer in pre_trained_model.layers:
  layer.trainable = False

# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('block5_pool')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization,ReLU,GaussianDropout

model = Flatten()(last_output)
model = Dense(1024)(model)
model=ReLU(0.1)(model)
model=Dropout(0.25)(model)
model=BatchNormalization()(model)
model = Dense(1024)(model)
model=ReLU(0.1)(model)
model=Dropout(0.25)(model)
model=BatchNormalization()(model)
model = Dense(1, activation='sigmoid')(model)
from tensorflow.keras.models import Model


fmodel = Model( pre_trained_model.input, model) 

fmodel.compile(optimizer = 'adam', 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])
fmodel.summary()
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau


early=EarlyStopping(monitor='accuracy',patience=3,mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=2, verbose=1,cooldown=0, mode='auto',min_delta=0.0001, min_lr=1e-5)
class_weight={0:1,1:3.3}
fmodel.fit(train,epochs=20,callbacks=[reduce_lr],steps_per_epoch=100,validation_data=test,class_weight=class_weight)
fmodel.save('/kaggle/working/model_vgg16.h5')
# Plot Accuracy
plt.figure(figsize=(30,20))
val_acc=np.asarray(fmodel.history.history['val_accuracy'])*100
acc=np.asarray(fmodel.history.history['accuracy'])*100
acc=pd.DataFrame({'val_acc':val_acc,'acc':acc})
acc.plot(figsize=(20,10),yticks=range(50,100,5))
# Plot loss
loss=fmodel.history.history['loss']
val_loss=fmodel.history.history['val_loss']
loss=pd.DataFrame({'val_loss':val_loss,'loss':loss})
loss.plot(figsize=(20,10))
y=[]

test.reset()

for i in tqdm(range(84)):
    _,tar=test.__getitem__(i)
    for j in tar:
        y.append(j)
test.reset()
y_pred=fmodel.predict(test)
# define iou or jaccard loss function
def iou_loss(y_true, y_pred):
    #print(y_true)
    y_true=tf.cast(y_true, tf.float32)
    y_pred=tf.cast(y_pred, tf.float32)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
   
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score

# combine bce loss and iou loss
def iou_bce_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)

# mean iou as a metric
def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))

pred=[]
for i in y_pred:
    if i[0]>=0.5:
        pred.append(1)
    else:
        pred.append(0)
from sklearn.metrics import roc_curve,auc,precision_recall_curve,classification_report
# Classification Report
# print(classification_report(y,pred))  --> error for Found input variables with inconsistent numbers of samples: [2688, 8004]
print(classification_report(y,pred[:len(y)]))

plt.figure(figsize=(30,20))
fpr,tpr,_=roc_curve(y,y_pred[:len(y)])
area_under_curve=auc(fpr,tpr)
print('The area under the curve is:',area_under_curve)
# Plot area under curve
plt.plot(fpr,tpr,'b.-')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.plot(fpr,fpr,linestyle='--',color='black')