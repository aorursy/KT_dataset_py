# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import math
from keras.layers import Dense,Input,Conv2D,MaxPooling2D,Dropout,BatchNormalization,Flatten,GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,CSVLogger,Callback
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import keras.backend as K
from keras.utils import to_categorical
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix
import gc
import tqdm
from sklearn.metrics import confusion_matrix,accuracy_score
from keras.preprocessing.image import load_img
from skimage.transform import resize
from skimage.io import imread
from sklearn.metrics import mean_absolute_error



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
np.set_printoptions(suppress=True)

PATH = '/kaggle/input/'
TRAIN_PATH = PATH+'distorted_images/'
BATCH_SIZE = 16
lr = 0.001
W,H = 254,254
EPOCH = 15
df = pd.read_csv(PATH+'mos_with_names.txt',sep='\s',header=None,engine='python') 
# Read data details from official page to know the reason
df.rename({0:'Score',1:'Image Name'},axis=1,inplace=True)
df = df[['Image Name','Score']]
# df = df.sample(frac=1.0)
df.head()
df['Score'].nunique()
df['Round Class'] = df['Score'].apply(lambda x: round(x))
df['Ceil Class'] = df['Score'].apply(lambda x: math.ceil(x))
df.describe()
y = to_categorical(df['Round Class']) # because mean is close to original and std is close to ceil
# y.shape
df['Label'] = df['Round Class'].apply(lambda x: str(x)) 
#df['Label'] = to_categorical(df['Round Class'])

# because ImageDataGenerator accepts strings as categorical
f,ax = plt.subplots(3,3,figsize=(15,9))
ax = ax.ravel()
sample_df = df.sample(9)

for i,index in enumerate(sample_df.index):
    ax[i].imshow(plt.imread(TRAIN_PATH+sample_df.loc[index,'Image Name']))
    ax[i].set_title(f'Score: {sample_df.loc[index,"Score"]}')
    ax[i].axis('off')
    
plt.savefig('random_images.png')

f,ax = plt.subplots(1,2,figsize=(10,5))
ax = ax.ravel()
max_min = [df['Score'].idxmax(),df['Score'].idxmin()]
for i,index in enumerate(max_min):
    ax[i].imshow(plt.imread(TRAIN_PATH+df.iloc[index,0]))
    ax[i].set_title(f'Score: {df.iloc[index,1]}')
    ax[i].axis('off')

plt.savefig('min_max_score_images.png')
def earth_mover_loss(y_true, y_pred):
    cdf_true = K.clip(K.cumsum(y_true, axis=-1), 0,1)
    cdf_pred = K.clip(K.cumsum(y_pred, axis=-1), 0,1)
    emd = K.mean(K.square(cdf_true - cdf_pred), axis=-1)
    return K.mean(emd)
base_model = InceptionResNetV2(input_shape=(W,H, 3),include_top=False,pooling='avg',weights='imagenet')
# Do not train any of the layers so that weights are optimized as they were in imagenet compitition
for layer in base_model.layers: 
    layer.trainable = False

# x = GlobalAveragePooling2D(basemodel.output) # we have used pooling='avg' so no need   
x = Dropout(0.77)(base_model.output)
out = Dense(y.shape[1],activation='softmax')(x)

model = Model(base_model.input,out)
# model.summary()
split = int(len(df)*0.89)

df = df.sample(frac=1.0)

train_df = df.iloc[:split,:]
test_df = df.iloc[split:,:]


gen=ImageDataGenerator(validation_split=0.11,preprocessing_function=preprocess_input)

train = gen.flow_from_dataframe(train_df,TRAIN_PATH,x_col='Image Name',y_col='Label',subset='training',
                                target_size=(W,H),batch_size=BATCH_SIZE,class_mode='categorical')

val = gen.flow_from_dataframe(train_df,TRAIN_PATH,x_col='Image Name',y_col='Label',subset='validation',
                              target_size=(W,H),batch_size=BATCH_SIZE,class_mode='categorical')
optimizer = Adam(lr=0.001)
model.compile(optimizer,loss=earth_mover_loss,metrics=['acc'])
es = EarlyStopping(monitor='val_loss',min_delta=0.01,mode='min',patience=5,verbose=1,restore_best_weights=True)

rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1, min_delta=0.01,min_lr=0.0001,
                        mode='min',cooldown=1)

csv_logger = CSVLogger('training.log')

callbacks = [es,rlp,csv_logger]
History = model.fit_generator(train,steps_per_epoch=train.n//BATCH_SIZE,epochs=EPOCH,callbacks=callbacks,
                              validation_data=val,validation_steps=val.n//BATCH_SIZE,)
def open_preprocess_image(filepath,W=W,H=H):
    image = imread(filepath)
    image = resize(image,(W,H,3))
    image = preprocess_input(image)
    
    return image


def get_image_batch(test_df):
    images = []
    for i in range(len(test_df)):    
        filepath = TRAIN_PATH+test_df.iloc[i,0]
        image = open_preprocess_image(filepath)
        images.append(image)
        
    return np.array(images)


def get_predictions(images):
    pred_scores = []
    lis = range(y.shape[1])
    
    class_probs = model.predict(images)
    for prob in class_probs:
        score = sum([prob[i]*i for i in lis])
        pred_scores.append(score)
    
    return pred_scores
images = get_image_batch(test_df)
pred_scores = get_predictions(images)
test_df['Pred Score'] = pred_scores
print(f'Model has MAE of: {mean_absolute_error(test_df["Score"],pred_scores)}')
f,ax = plt.subplots(3,3,figsize=(15,9))
ax = ax.ravel()
sample_df = test_df.sample(9)

for i,index in enumerate(sample_df.index):
    ax[i].imshow(plt.imread(TRAIN_PATH+sample_df.loc[index,'Image Name']))
    ax[i].set_title(f'Actual: {round(sample_df.loc[index,"Score"],2)}  Predicted: {round(sample_df.loc[index,"Pred Score"],2)}')
    ax[i].axis('off')
    
plt.savefig('Actual vs Predicted Scores.png')

plot_df = test_df[abs(test_df['Score']-test_df['Pred Score'])<=0.09]

f,ax = plt.subplots(3,3,figsize=(15,9))
ax = ax.ravel()
sample_df = plot_df.sample(9)

for i,index in enumerate(sample_df.index):
    ax[i].imshow(plt.imread(TRAIN_PATH+sample_df.loc[index,'Image Name']))
    ax[i].set_title(f'Actual: {round(sample_df.loc[index,"Score"],2)}  Predicted: {round(sample_df.loc[index,"Pred Score"],2)}')
    ax[i].axis('off')
    
plt.savefig('Actual vs Predicted Scores.png')

