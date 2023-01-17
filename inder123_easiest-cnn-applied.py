import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random 
import os
print(os.listdir("../input/dogs-vs-cats/test/test"))
print(len(os.listdir('../input/dogs-vs-cats/train/train')))
print(len(os.listdir('../input/dogs-vs-cats/test/test')))
print(os.listdir('../input/dogs-vs-cats/train/train'))
filenames=os.listdir("../input/dogs-vs-cats/train/train/")
categories=[]
for filename in filenames:
    category=filename.split('.')[0]
    if category=='dog':
        categories.append(1)
    else:
        categories.append(0)
        
df=pd.DataFrame({
    'filename':filenames,
    'category':categories
})
df.head()
df.tail()
df['category'].value_counts().plot.bar()
sample=random.choice(filenames)
image=load_img("train/" +sample)
plt.imshow(image)
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Activation

image_width=128
image_height=128
image_size=(image_width,image_height)
image_channels=3
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(image_width,image_height,image_channels)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))              #model.add(Dense(2,activation='softmax'))  

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.summary()

from keras.callbacks import EarlyStopping,ReduceLROnPlateau
earlystop=EarlyStopping(patience=10)
learning_rate_reduce=ReduceLROnPlateau(monitor='val_acc',min_lr=0.0001)
callbacks=[earlystop,learning_rate_reduce]
df['category']=df['category'].replace({0:'cat',1:'dog'})
train_df,validate_df=train_test_split(df,test_size=0.20)
train_df=train_df.reset_index(drop=True)
validate_df=validate_df.reset_index(drop=True)
train_df['category'].value_counts().plot.bar()
validate_df['category'].value_counts().plot.bar()
total_train=train_df.shape[0]
total_validate=validate_df.shape[0]

print(total_train,total_validate)
train_datagen=ImageDataGenerator(
rotation_range=15,
rescale=1/255,
zoom_range=0.1,
horizontal_flip=True,width_shift_range=0.1,height_shift_range=0.1)

train_generator=train_datagen.flow_from_dataframe(train_df,"../input/dogs-vs-cats/train/train/",x_col='filename',y_col='category',target_size=image_size,class_mode='categorical',batch_size=64)
validation_datagen=ImageDataGenerator(
rotation_range=15,
rescale=1/255,
zoom_range=0.1,
horizontal_flip=True,width_shift_range=0.1,height_shift_range=0.1)

valid_generator=validation_datagen.flow_from_dataframe(validate_df,"../input/dogs-vs-cats/train/train/",x_col='filename',y_col='category',target_size=image_size,class_mode='categorical',batch_size=64)
example_df=train_df.sample(n=1).reset_index(drop=True)
example_generator=train_datagen.flow_from_dataframe(
example_df,"../input/dogs-vs-cats/train/train/",x_col='filename',y_col='category',target_size=image_size,class_mode='categorical'


)
batch_size=250
history=model.fit_generator(train_generator,epochs=20,validation_data=valid_generator,validation_steps=total_validate/batch_size,steps_per_epoch=total_train/batch_size,callbacks=callbacks)
plt.plot(history.history['val_loss'],color='r')
plt.plot(history.history['accuracy'],color='black')
plt.plot(history.history['loss'],color='black')
plt.plot(history.history['val_accuracy'],color='yellow')
plt.show()

test_filenames=os.listdir('../input/dogs-vs-cats/test/test')

test_df=pd.DataFrame({
    'filename':test_filenames
})
print(len(os.listdir('../input/dogs-vs-cats/test/test')))
test_gen=ImageDataGenerator(rescale=1/255)

test_generator=test_gen.flow_from_dataframe(
test_df,'../input/dogs-vs-cats/test/test/',x_col='filename',y_col=None,class_mode=None,target_size=image_size,batch_size=64,shuffle=False)
predict=model.predict_generator(test_generator)
test_df['category']=np.argmax(predict,axis=-1)
label_map=dict((v,k) for k,v in train_generator.class_indices.items())

test_df['category']=test_df['category'].replace(label_map)
test_df['category']=test_df['category'].replace({'dog':1,'cat':0})
test_df['category'].value_counts().plot.bar()
sample_test=test_df.head(15)
sample_test.head()
plt.figure(figsize=(15,25))
for index,row in sample_test.iterrows():
    filename=row['filename']
    category=row['category']
    img=load_img('../input/dogs-vs-cats/test/test/'+ filename,target_size=image_size)
    plt.subplot(6,3,index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
    
plt.show()





submission_df=test_df.copy()
submission_df['id']=submission_df['filename'].str.split('.').str[0]
submission_df['label']=submission_df['category']
submission_df.drop(['filename','category'],axis=1,inplace=True)
submission_df.to_csv('submission.csv',index=False)
