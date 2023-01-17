
import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os
import cv2
from glob import glob
import random
from keras.layers import *
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.utils import to_categorical
IMG_ROW=IMG_COL=224
BASE_IMG_DIR='../input/HAM10000_metadata.csv'
IMG_CHANNEL=3
base_skin_dir = os.path.join('..', 'input')
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
train_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))
train_df['path'] = train_df['image_id'].map(imageid_path_dict.get)
train_df.sample(3)
labelnum=train_df.groupby('dx').size()
labelnum.plot(kind='barh')
le=LabelEncoder()
le.fit(train_df['dx'])
labeldf=le.transform(train_df['dx'])

labeldf=to_categorical(labeldf)
print(labeldf[:10])
train_img_path,valid_img_path,train_label,valid_label=train_test_split(train_df['path'],labeldf,
                                                                      test_size=0.2)
def read_img(imgpath):
    img=cv2.imread(imgpath)
    img=cv2.resize(img,(IMG_ROW,IMG_COL))
    return img
def train_gen(batch_size=64):
    while(True):
        imglist=np.zeros((batch_size,IMG_ROW,IMG_COL,IMG_CHANNEL))
        labellist=np.zeros((batch_size,len(labelnum)))
        for i in range(batch_size):
            rndid=random.randint(0,len(train_img_path)-1)
            imgpath=train_img_path[train_img_path.index[rndid]]
            img=read_img(imgpath)
            label=train_label[rndid]
            imglist[i]=img
            labellist[i]=label
        yield (imglist,labellist)
def valid_gen():
    imglist=np.zeros((len(valid_img_path),IMG_ROW,IMG_COL,IMG_CHANNEL))
    labellist=np.zeros((len(valid_img_path),len(labelnum)))
    for i,imgpath in enumerate(valid_img_path):
        img=read_img(imgpath)
        label=valid_label[i]
        imglist[i]=img
        labellist[i]=label
    return (imglist,labellist)
validdata=valid_gen()
def build_model():
    # load pre-trained model graph, don't add final layer
    model = InceptionResNetV2(include_top=False, input_shape = (IMG_ROW,IMG_COL,IMG_CHANNEL),
                                          weights='imagenet')
    new_output =GlobalAveragePooling2D()(model.output)
    new_output=Dense(256,activation='relu')(new_output)
    new_output = Dense(len(labelnum), activation='softmax')(new_output)
    model = Model(model.inputs, new_output)
    return model
model=build_model()
model.compile(metrics=['accuracy'],loss='categorical_crossentropy',
             optimizer='Adam')
callbacks=[
    ModelCheckpoint('model.h5',monitor='val_loss',save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss',patience=5)
]
history=model.fit_generator(train_gen(),epochs=50,
                            steps_per_epoch=80,
                            validation_data=validdata,
                           callbacks=callbacks)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','valid'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['train','valid'])