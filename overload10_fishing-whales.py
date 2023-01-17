# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import shutil

import time

import gc

from tqdm import tqdm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.



'''from sklearn.model_selection import train_test_split

from keras import models,callbacks,layers,utils,optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import resnet50'''



from fastai.vision import *

from fastai.metrics import error_rate
df=pd.read_csv('../input/train.csv')

df.head()
df.describe()
df['total_count'] = df.groupby('Id')['Id'].transform('count')
df.head()
df.shape
df_grouped = df.groupby('Id').apply(lambda x : x.sample(frac=0.2,random_state=22))

df_grouped.shape
df_grouped.describe()
df_grouped.tail(10)
df_merged = pd.merge(left=df,right=df_grouped,how='left',on='Image',suffixes=('','_y'))
df_merged['is_valid'] = df_merged.Id_y.isnull() != True

df_merged.tail(10)
df_merged.drop(columns=['Id_y','total_count_y'],inplace=True)

df_merged.tail(10)
del df_grouped

del df

gc.collect()
data_src=(ImageItemList.from_df(df=df_merged,path='../input/',folder='train')

       #Where to find the data? 

       .split_from_df(col='is_valid')

       #How to split in train/valid?

       .label_from_df(cols='Id'))         
data= (data_src       

       .add_test_folder(test_folder='../input/test/')

       #Optionally add a test set

       .transform(get_transforms(do_flip=False),size=224)

       #Data augmentation?           

       .databunch(num_workers=0)

       #Finally? -> use the defaults for conversion to ImageDataBunch

      .normalize(imagenet_stats))

      

        

       
print('Batch size:',data.batch_size, '\nTotal Classes:',data.c,'\n')
data.show_batch(rows=3,figsize=(10,10))
learn = create_cnn(data=data,arch=models.resnet50,metrics=error_rate,model_dir='.',path='../working/tmp/',pretrained=True)

#learn = create_cnn(data=data,arch=models.resnet18,metrics=error_rate,model_dir='.',callback_fns=ShowGraph)

learn.model
learn.lr_find()

learn.recorder.plot()
# learn.fit_one_cycle(4)

learn.fit_one_cycle(7,max_lr=slice(1e-04,1e-02))
learn.save('stage-1_res18')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(7, max_lr=slice(1e-9,1e-6))
learn.save('stage-2_res18')
# data= (data_src       

#        .add_test_folder(test_folder='../input/test/')

#        #Optionally add a test set

#        .transform(get_transforms(do_flip=False),size=320)

#        #Data augmentation?           

#        .databunch(num_workers=0)

#        #Finally? -> use the defaults for conversion to ImageDataBunch

#       .normalize(imagenet_stats))                 
# learn.save('stage-Final_res18')
os.listdir('../working/tmp/')
learn.recorder.plot_losses()
learn.recorder.plot_lr()
interp = ClassificationInterpretation.from_learner(learn)
#interp.plot_confusion_matrix()
losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))


#interp.most_confused()
# img=learn.data.train_ds[0][0]

# learn.predict(img)
#preds,y = learn.TTA()
#preds_t,y_t = learn.TTA(ds_type=learn.data.train_ds[:100],is_test=True)
# #preds_t = np.stack(preds, axis=-1)

# preds_t = np.exp(preds_t)

# preds_t = preds_t.mean(axis=-1)
# preds_t.shape
# data.train_ds
# preds=learn.pred_batch(ds_type=data.test_ds)
# preds[0].shape,preds[1].shape
len(learn.data.train_ds),len(learn.data.valid_ds),len(learn.data.test_ds)
len(data.train_ds),len(data.valid_ds), len(data.test_ds)
# preds = learn.predict()
#LRFinder??

len(data.train_ds),len(data.valid_ds), len(data.test_ds)
# learn.show_results(ds_type=DatasetType.Train)
 #learn.show_results(ds_type=DatasetType.Test)

#pred = learn.predict(DatasetType.Test)
pred_2=learn.TTA(ds_type=DatasetType.Test)
best_5_pred = torch.flip(np.argsort(pred_2[0])[:,-5:],dims=[1])
#Generate mapping of each index to an Id on the format

pred_label = []

for i in best_5_pred:

    temp=[]

    temp_str=''

    for name in i:

        temp.append(data.classes[name])         

    temp_str =' '.join(temp)

    pred_label.append(temp_str)    
submission = pd.DataFrame({"Image": os.listdir('../input/test/'), "Id": pred_label})

submission.to_csv("submission.csv", index = False)

submission.head(10)
submission.shape
#train_labels = df[~(df.Id == 'new_whale')]
#train,val = train_test_split(df.Id,df.index,stratify=df.index)
#train.shape, val.shape
#df.head()
#train_labels.head()
#train_labels.shape
# Define data pre-processing 

#train_image_gen = ImageDataGenerator(rescale=1/255,horizontal_flip=True,validation_split=val_split,)



#train_image_gen = ImageDataGenerator(rotation_range=0.3, width_shift_range=0.3, height_shift_range=0.3,shear_range=0.3, horizontal_flip=True, rescale=1/255, preprocessing_function=resnet50.preprocess_input, validation_split=val_split)
#train_generator = train_image_gen.flow_from_directory(train_dir,target_size=(Image_width,Image_height),batch_size=batch_size,seed=42,subset='training',shuffle=True,class_mode='categorical')

#val_generator = train_image_gen.flow_from_directory(train_dir,target_size=(Image_width,Image_height),batch_size=batch_size,seed=42,subset='validation',shuffle=True,class_mode='categorical')

#train_generator=train_image_gen.flow_from_dataframe(dataframe=train_labels,directory='../input/train/',x_col='Image',y_col='Id',seed=42,has_ext=True,class_mode='categorical',target_size=(Image_width,Image_height),subset='training')

#val_generator=train_image_gen.flow_from_dataframe(dataframe=train_labels,directory='../input/train/',x_col='Image',y_col='Id',seed=42,has_ext=True,class_mode='categorical',target_size=(Image_width,Image_height),subset='validation')
#reset50_base_model = resnet50.ResNet50(include_top=False, weights='imagenet')
# res = reset50_base_model.output

# res_pool = layers.GlobalAveragePooling2D()(res)

# res_dense = layers.Dense(units=Number_FC_Neurons,activation='relu')(res_pool)

# final_pred = layers.Dense(num_classes,activation='softmax')(res_dense)

# model = models.Model(inputs=reset50_base_model.input,output=final_pred)

# model.summary()
# cb_checkpoint = callbacks.ModelCheckpoint(filepath='../working/best.hd5',monitor='val_loss',save_best_only=True,mode=min)

# cb_stopping = callbacks.EarlyStopping(monitor='val_loss',mode=min,patience=2,restore_best_weights=True)

# my_callback = [cb_checkpoint,cb_stopping]
# layers_to_freeze=130

# for layer in model.layers[:layers_to_freeze]:

#     layer.trainable=False



# for layer in model.layers[layers_to_freeze:]:

#     layer.trainable=True

    

# #sgd = optimizers.SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)

# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#train_step = train_generator.n//train_generator.batch_size

#val_step= val_generator.n//val_generator.batch_size
#learnings = model.fit_generator(train_generator,epochs=2,steps_per_epoch=train_step,validation_data=val_generator,validation_steps=val_step,verbose=1,callbacks=my_callback,class_weight='auto')
#test_dir='../data/'
# Define data pre-processing 

#test_image_gen = ImageDataGenerator(rescale=1./255)

#test_generator = test_image_gen.flow_from_directory(test_dir,target_size=(Image_width,Image_height),batch_size=1,seed=42,class_mode=None,shuffle=False)
#test_generator.reset()

#y_pred = model.predict_generator(generator=test_generator,verbose=1,)
#best_5_pred = np.flip(axis=1,m=np.argsort(y_pred)[:,-5:])
#Generate mapping of each index to an Id on the format

#mapIndex = dict((v,k) for (k,v) in train_generator.class_indices.items())
#submission.loc[submission['Image']=='000dcf7d8.jpg']