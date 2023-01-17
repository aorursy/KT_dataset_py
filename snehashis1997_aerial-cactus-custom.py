import numpy as np

seed = 42

np.random.seed(seed)
import cv2

from glob import glob

import pandas as pd 

import numpy as np

from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import cv2

import copy

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Conv2D,BatchNormalization,Convolution2D,MaxPooling2D

from keras.layers import Flatten,Activation

from keras.layers import Dropout

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

from keras import initializers

import numpy as np

from keras import regularizers



from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import auc,roc_curve,roc_auc_score



from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from glob import glob

import pandas as pd

import os
pngs=glob(r'../input/train/train/*.jpg')

len(pngs)
df=pd.read_csv(r'../input/train.csv')
height=32



width=32



batchsize=32



channel=1



ch=0
dataset=[]



y_true=[]



for i in range(len(pngs)):

    

    name=r'../input/train/train/' + str(df['id'][i])

    

    y_true.append(df['has_cactus'][i])

    

    img=cv2.imread(name,ch)

    

    dataset.append(img)
dataset=np.array(dataset)



y_true=np.array(y_true)
dataset = dataset.reshape(-1,height,width,channel)
y_true
dataset.shape
img=dataset[1]

img.shape



#plt.imshow(img)

#plt.show()
type(dataset)
x_train,x_val,y_train,y_val=train_test_split(dataset,y_true,shuffle=True,test_size=0.5)



print('okay')
earlystop = EarlyStopping(monitor = 'val_loss', 

                          min_delta = 0, 

                          patience = 6,

                          verbose = 1,mode='min',

                          restore_best_weights = True)



reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', mode='min',factor = 0.2, patience = 1, verbose = 1, min_delta = 0.0001)



# we put our call backs into a callback list

callbacks = [earlystop,reduce_lr]
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=15,

                                    width_shift_range=0.1,

                                    height_shift_range=0.1,

                                    shear_range=0.2,

                                    zoom_range=0.2,

                                    horizontal_flip=True,

                                    fill_mode='nearest')



test_datagen=ImageDataGenerator(rescale=1./255)
model=Sequential()

#model.add(GaussianNoise(0.1))

model.add(Convolution2D(8,kernel_size=(3,3),

                        activation='relu',

                        kernel_regularizer=regularizers.l2(0.00001),

                        input_shape=(height,width,channel)))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.5))



model.add(Convolution2D(8,kernel_size=(3,3),

                        activation='relu',

                        kernel_regularizer=regularizers.l2(0.00001)))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.5))



model.add(Convolution2D(16,kernel_size=(5,5),

                        activation='relu',

                        kernel_regularizer=regularizers.l2(0.00001)))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(32*4,

                activation='relu',

                kernel_regularizer=regularizers.l2(0.00001)))



model.add(BatchNormalization())

model.add(Dropout(0.8))



model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['acc'])

va = EarlyStopping(monitor='val_loss',verbose=1, patience=50)
output=model.fit_generator(train_datagen.flow(x=x_train, y=y_train, batch_size=batchsize),

                             epochs=40, verbose=1,

                             validation_data=test_datagen.flow(x_val,y_val,batch_size=batchsize), 

                             shuffle=False, steps_per_epoch=x_train.shape[0]//batchsize,

                             validation_steps=x_val.shape[0])
plt.plot(output.history['acc'])

plt.plot(output.history['val_acc'])

plt.title('multiclass classifier 40X accuracy for  view')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(output.history['loss'])

plt.plot(output.history['val_loss'])

plt.title('multiclass classifier 40X loss for  view')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
#test=pd.read_csv(r'../input/test.csv')
x_test=[]



y_test=[]



pngs_test=glob(r'../input/test/test/*.jpg')

#len(pngs)



for i in range(len(pngs_test)):

    

    img=cv2.imread(pngs_test[i],ch)

    

    x_test.append(img)
x_test=np.array(x_test)

x_test=x_test.reshape(-1,height,width,channel)



x_test=x_test/np.max(x_test)

x_test.shape
test_err=model.evaluate_generator(test_datagen.flow(x_val,y_val,batch_size=batchsize),steps=20)



print('Loss: ',test_err[0])

print('Accuracy: ',test_err[1])
req=x_val/np.max(x_val)

predicted=model.predict_classes(req)
predicted
cm_binary=confusion_matrix(y_val,predicted)
def accuracy(confusion_matrix):

    diagonal_sum = confusion_matrix.trace()

    sum_of_all_elements = confusion_matrix.sum()

    return diagonal_sum / sum_of_all_elements 
fpr, tpr, _ = roc_curve(y_val,predicted)

auc = roc_auc_score(y_val,predicted)
cm_binary
accuracy(cm_binary)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.title('for binary classification')

plt.legend(loc=4)

plt.show()
pred=model.predict_classes(x_test)

ids=[]

test_path='../input/test/test'

label=[]

a=0

for i in tqdm(os.listdir(test_path)):

    id=i

    ids.append(id)

    label.append(pred[a])

    a=a+1



label=np.array(label,dtype='float64')

out=pd.DataFrame({'id': ids,'has_cactus':label[:,0]})



out.to_csv('cactus_identifier_net.csv',index=False,header=True)