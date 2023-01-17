import pandas as pd
import numpy as np
import os, shutil, keras, glob, warnings
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings(action='ignore')
labels={'bee':'c0', 'wasp':'c1', 'insect':'c2', 'other':'c3'}
batch_size=8

df=pd.read_csv('../input/bee-vs-wasp/kaggle_bee_vs_wasp/labels.csv')
df['label']=df['label'].map(lambda x: labels[x])
df['path']=df['path'].map(lambda x: x.replace("\\", "/"))

trn_df=df[df['is_final_validation']==0].reset_index(drop=True)
tst_df=df[df['is_final_validation']==1].reset_index(drop=True)

tst_df['file_name']=tst_df['path'].map(lambda x: os.path.basename(x))
tst_df.head()
def remove_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)

data_path='../input/bee-vs-wasp/kaggle_bee_vs_wasp'
trn_path='../working/trn/'
vld_path='../working/vld/'

remove_path(trn_path)
remove_path(vld_path)
os.mkdir(trn_path)
os.mkdir(vld_path)

tst_path='../working/tst/'
remove_path(tst_path)
os.mkdir(tst_path)
os.mkdir(os.path.join(tst_path, 'imgs'))

for tst_index in tst_df.index:        
    shutil.copy(os.path.join(data_path, tst_df.loc[tst_index, 'path']),
                os.path.join(tst_path, 'imgs'))

print('My Folders.......')
print(os.listdir('../working'))
print('Sample test imgs......')
print(glob.glob(os.path.join(tst_path, 'imgs','*.jpg'))[:5])
trn_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,    
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

tst_datagen=ImageDataGenerator(rescale=1./255)

tst_gen=tst_datagen.flow_from_directory(tst_path, target_size=(224,224),
                                        batch_size=1, class_mode=None,
                                        shuffle=False)
def get_model():
    model=keras.models.Sequential()    
    base_model=keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                              input_shape=(224,224,3))
    model.add(base_model)
    model.add(Flatten())    
    model.add(Dense(units=2048, activation='relu', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(units=2048, activation='relu', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.5))    
    model.add(Dense(units=4, activation='softmax'))    
    opt=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)        
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    return model
def trn_vld_split(trn_index, vld_index):
    for label in labels.values():
        remove_path(os.path.join(trn_path, label))
        remove_path(os.path.join(vld_path, label))
        os.mkdir(os.path.join(trn_path, label))
        os.mkdir(os.path.join(vld_path, label))
    
    for index in trn_index:
        shutil.copy(os.path.join(data_path, trn_df.loc[index, 'path']),
                    os.path.join(trn_path, trn_df.loc[index, 'label']))
    
    for index in vld_index:
        shutil.copy(os.path.join(data_path, trn_df.loc[index, 'path']),
                    os.path.join(vld_path, trn_df.loc[index, 'label']))
skf=StratifiedKFold(n_splits=5)

history_list=[]
preds_list=[]

for n, (trn_index, vld_index) in enumerate(skf.split(trn_df.drop(columns='label'), trn_df['label'])):
    print('='*100)
    print('{}/5 StratifiedKFold starting..................'.format(str(n+1)))
    print('='*100)
    trn_vld_split(trn_index, vld_index)
    
    trn_gen=trn_datagen.flow_from_directory(trn_path, target_size=(224,224),
                                            batch_size=batch_size, class_mode='categorical',
                                            shuffle=True)
    
    vld_gen=trn_datagen.flow_from_directory(vld_path, target_size=(224,224),
                                            batch_size=batch_size, class_mode='categorical',
                                            shuffle=True)
    
    model=get_model()
    
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)]               
    
    history=model.fit_generator(trn_gen, steps_per_epoch=len(trn_index)/batch_size, epochs=100,
                                validation_data=vld_gen, validation_steps=len(vld_index)/batch_size,
                                shuffle=True, verbose=1, callbacks=callbacks)
    
    print("Making Predictions..................")
    preds_list.append(model.predict_generator(tst_gen, verbose=1))
    history_list.append(history)
preds_enb=0
for preds in preds_list:
    preds_enb+=preds
preds_enb/=len(preds_list)
df=pd.DataFrame(data={'file_name':tst_gen.filenames, 'pred':preds_enb.argmax(axis=1)})
df['pred']=df['pred'].map(lambda x: 'c'+str(x))
df['file_name']=df['file_name'].map(lambda x: os.path.basename(x))
df=pd.merge(df, tst_df[['file_name', 'label']])
df.head()
df['result']=df.apply(lambda x: 1 if x['preds']==x['label'] else 0, axis=1)
acc=format(df['result'].sum()/df.shape[0], '.1%')
print('StratifiedKFold 5 ensemble acc : {}'.format(acc))
