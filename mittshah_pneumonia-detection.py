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
plt.figure(figsize=(30,20))

heatmap=np.zeros((1024,1024))

for _,row in df.iterrows():
    if row['Target']==1:
        x=int(row['x'])
        y=int(row['y'])
        w=int(row['width'])
        h=int(row['height'])
        heatmap[y:y+h,x:x+w]+=1

plt.axis('off')

plt.imshow(heatmap,cmap='hot')
    

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
from tensorflow.keras.applications.vgg19 import VGG19,preprocess_input
datagen=ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True,horizontal_flip=True,
                          width_shift_range=0.05,rescale=1/255,fill_mode='nearest',height_shift_range=0.05,
                           preprocessing_function=preprocess_input,validation_split=0.1,
                          )
train=datagen.flow_from_directory('data',color_mode='rgb',batch_size=128,class_mode='binary',subset='training')
test=datagen.flow_from_directory('data',color_mode='rgb',batch_size=32,class_mode='binary',subset='validation')
train.class_indices
pre_trained_model = VGG19(input_shape = (256,256,3), 
                                include_top = False, 
                                weights = 'imagenet')

for layer in pre_trained_model.layers:
  layer.trainable = False

# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('block5_pool')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization,LeakyReLU,GaussianDropout

model = Flatten()(last_output)
model = Dense(1024)(model)
model=LeakyReLU(0.1)(model)
model=Dropout(0.25)(model)
model=BatchNormalization()(model)
model = Dense(1024)(model)
model=LeakyReLU(0.1)(model)
model=Dropout(0.25)(model)
model=BatchNormalization()(model)
model = Dense(1, activation='sigmoid')(model)
from tensorflow.keras.models import Model


fmodel = Model( pre_trained_model.input, model) 

fmodel.compile(optimizer = 'adam', 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])
# fmodel.summary()
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau


early=EarlyStopping(monitor='accuracy',patience=3,mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=2, verbose=1,cooldown=0, mode='auto',min_delta=0.0001, min_lr=0)
class_weight={0:1,1:3.3}
fmodel.fit(train,epochs=13,callbacks=[reduce_lr],steps_per_epoch=100,validation_data=test,class_weight=class_weight)
fmodel.save('/kaggle/working/model.h5')
plt.figure(figsize=(30,20))
val_acc=np.asarray(fmodel.history.history['val_accuracy'])*100
acc=np.asarray(fmodel.history.history['accuracy'])*100
acc=pd.DataFrame({'val_acc':val_acc,'acc':acc})
acc.plot(figsize=(20,10),yticks=range(50,100,5))
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
pred=[]
for i in y_pred:
    if i[0]>=0.5:
        pred.append(1)
    else:
        pred.append(0)
from sklearn.metrics import roc_curve,auc,precision_recall_curve,classification_report
print(classification_report(y,pred))
plt.figure(figsize=(30,20))
fpr,tpr,_=roc_curve(y,y_pred)
area_under_curve=auc(fpr,tpr)
print('The area under the curve is:',area_under_curve)
plt.plot(fpr,tpr,'b.-')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.plot(fpr,fpr,linestyle='--',color='black')