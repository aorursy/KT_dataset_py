import os
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import keras.models as models
import keras.layers as layers
from keras import backend
train_df=pd.read_csv('/kaggle/input/planet-understanding-the-amazon-from-space/train_v2.csv/train_v2.csv')
train_df.head()
train_df['tags'].values
test_df=pd.read_csv('/kaggle/input/planet-understanding-the-amazon-from-space/test_v2_file_mapping.csv/test_v2_file_mapping.csv')
train_df.tail()
def tag_mapping(data):
    labels=set()
    for i in range(len(data)):
        tags=data['tags'][i].split(' ')
        labels.update(tags)
    labels=list(labels)
    labels.sort()
    labels_dict={labels[i]:i for i in range(len(labels))}
    inv_labels={i:labels[i] for i in range(len(labels))}
    return labels_dict,inv_labels
label_map,invmap=tag_mapping(train_df)
def file_mapping(data):
    mapping={}
    for i in range(len(data)):
        name,tags=train_df['image_name'][i],train_df['tags'][i]
        mapping[name]=tags.split(' ')
    return mapping
def one_hot_encode(tags, mapping):
    encoding = np.zeros(len(mapping), dtype='uint8')
    for tag in tags:
        encoding[mapping[tag]] = 1
    return encoding
def load_dataset(path,file_mapping,tag_mapping):
    photos,targets=list(),list()
    for filename in os.listdir(path):
        photo=load_img(path+filename,target_size=(75,75))
        photo=img_to_array(photo,dtype='uint8')
        tags=file_mapping[filename[:-4]]
        target=one_hot_encode(tags,tag_mapping)
        photos.append(photo)
        targets.append(target)
    X=np.asarray(photos,dtype='uint8')
    y=np.asarray(targets,dtype='uint8')
    return X,y
tags_mapping,_=tag_mapping(train_df)
files_mapping=file_mapping(train_df)
path='/kaggle/input/planets-dataset/planet/planet/train-jpg/'
X,y=load_dataset(path,files_mapping,tags_mapping)
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.optimizers import RMSprop
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
def fbeta(y_true, y_pred, beta=2):
    y_pred = backend.clip(y_pred, 0, 1)

    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
    p = tp / (tp + fp + backend.epsilon())
    r = tp / (tp + fn + backend.epsilon())
    bb = beta ** 2
    fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
    return fbeta_score
from keras.applications import InceptionV3
model=InceptionV3(input_shape=(75,75,3),include_top=False)
for layer in model.layers:
    layers.trainable=False
last_layer=model.get_layer('mixed7')
last_output=last_layer.output
    
x=layers.Flatten()(last_output)
x=layers.Dense(1024,activation='relu')(x)
x=layers.Dense(512,activation='relu')(x)
x=layers.Dropout(0.2)(x)
x=layers.Dense(17,activation='sigmoid')(x)
model=models.Model(model.inputs,x)
model.compile(optimizer=RMSprop(lr=0.0001),loss='binary_crossentropy',metrics=[fbeta])
train_datagen=ImageDataGenerator(rescale=1.0/255.0,horizontal_flip=True, vertical_flip=True, rotation_range=90)
test_datagen=ImageDataGenerator(rescale=1.0/255.0)
train_gen=train_datagen.flow(X_train,y_train,batch_size=64)
test_gen=test_datagen.flow(X_test,y_test,batch_size=64)
history = model.fit(train_gen,steps_per_epoch=506,validation_data=test_gen, validation_steps=127, epochs=250, verbose=0)
loss, fbeta =model.evaluate_generator(test_gen, steps=8, verbose=0)
print('> loss=%.3f, fbeta=%.3f' % (loss, fbeta))
history.history['fbeta']
test_path_1='/kaggle/input/planets-dataset/planet/planet/test-jpg/'
test_path_2='/kaggle/input/planets-dataset/test-jpg-additional/test-jpg-additional/'
submission_df=pd.read_csv('/kaggle/input/planet-understanding-the-amazon-from-space/sample_submission_v2.csv/sample_submission_v2.csv')
photo_test=[]
for filename in submission_df['image_name']:
    if filename[:1]=='t':
        img=load_img(test_path_1+filename+'.jpg',target_size=(75,75))
    elif filename[:1]=='f':
        img=load_img(test_path_2+filename+'.jpg',target_size=(75,75))
    ph=img_to_array(img,dtype='uint8')
    photo_test.append(ph)
test_x=np.asarray(photo_test,dtype='uint8')
image_gen_test=ImageDataGenerator(rescale=1/255.0)
test_data_gen=image_gen_test.flow(test_x,shuffle=False,batch_size=64)
result=model.predict(test_data_gen)
new_df=pd.DataFrame(result,columns=tags_mapping.keys())
tags=new_df.columns
pred_tags=new_df.apply(lambda x: ' '.join(tags[x>0.5]),axis=1)
pred_tag=pd.DataFrame(pred_tags,columns=['tags'])
submission_df['tags']=pred_tag['tags']
submission_df.to_csv('attempt_4.csv',index=False)
submission_df.head()