import numpy as np

seed = 42

np.random.seed(seed)
import cv2

from glob import glob

import pandas as pd 

import numpy as np

from tqdm import tqdm



from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical
#from sklearn.metrics import confusion_matrix

#import cv2

#import copy

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import BatchNormalization,Convolution2D,MaxPooling2D

from keras.layers import Flatten,Activation

from keras.layers import Dropout

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

from keras import initializers

#import numpy as np

from keras import regularizers



#from sklearn.preprocessing import LabelEncoder

#from keras.utils.np_utils import to_categorical

#from sklearn.metrics import confusion_matrix,classification_report

#from sklearn.metrics import auc,roc_curve,roc_auc_score



from sklearn.model_selection import train_test_split

#from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

#from glob import glob

#import pandas as pd

import os
pngs=glob(r'../input/train_images/*.png')

len(pngs)
#len(aspect_ratio)
#set(aspect_ratio)
ratio=0.75



height=int(128*.75)



width=128



batchsize=4



channel=1



ch=0
height
#del aspect_ratio,height_list,width_list
#del aspect
df=pd.read_csv(r'../input/train.csv')
set(df['diagnosis'])
df[5:10]
dataset=[]



y_true=[]



for i in range(len(pngs)):

    

    #print(i)

    name=r'../input/train_images/' + str(df['id_code'][i]) + '.png'

    

    y_true.append(df['diagnosis'][i])

    

    img=cv2.imread(name,ch)

    

    img=cv2.resize(img,(width,height),cv2.INTER_AREA)

    

    dataset.append(img)

    

    del img
dataset=np.array(dataset)



y_true=np.array(y_true)
encoder = LabelEncoder()

encoder.fit(y_true)

y_true = encoder.transform(y_true)

y_true = to_categorical(y_true)
dataset = dataset.reshape(-1,height,width,channel)
dataset.shape
img=dataset[1]

img.shape
type(dataset)
x_train,x_val,y_train,y_val=train_test_split(dataset,y_true,shuffle=True,test_size=0.4)



print('okay')
del dataset,y_true
earlystop = EarlyStopping(monitor = 'val_loss', 

                          min_delta = 0, 

                          patience = 8,

                          verbose = 1,mode='min',

                          restore_best_weights = True)



reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', mode='min',factor = 0.2, patience = 1, verbose = 1, min_delta = 0.0001)



# we put our call backs into a callback list

callbacks = [earlystop,reduce_lr]
train_datagen = ImageDataGenerator(rescale=1./255,

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

#model.add(Dropout(0.5))



model.add(Convolution2D(8,kernel_size=(3,3),

                        activation='relu',

                        kernel_regularizer=regularizers.l2(0.00001)))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.5))



model.add(Convolution2D(32,kernel_size=(5,5),

                        activation='relu',

                        kernel_regularizer=regularizers.l2(0.00001)))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(32*5,

                activation='relu',

                kernel_regularizer=regularizers.l2(0.00001)))



model.add(BatchNormalization())

model.add(Dropout(0.8))



model.add(Dense(5,activation='softmax'))
model.compile(optimizer=Adam(0.00001), loss='categorical_crossentropy', metrics=['acc'])

#va = EarlyStopping(monitor='val_loss',verbose=1, patience=50)
x_train=x_train.reshape(-1,height,width,channel)



x_val=x_val.reshape(-1,height,width,channel)
output=model.fit_generator(train_datagen.flow(x=x_train, y=y_train, batch_size=batchsize),

                             epochs=3, verbose=1,callbacks=callbacks,

                             validation_data=test_datagen.flow(x_val,y_val,batch_size=batchsize), 

                             shuffle=False, steps_per_epoch=x_train.shape[0]//batchsize,

                             validation_steps=x_val.shape[0])
plt.plot(output.history['acc'])

plt.plot(output.history['val_acc'])

plt.title('multiclass classifier accuracy for  view')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(output.history['loss'])

plt.plot(output.history['val_loss'])

plt.title('multiclass classifier loss for  view')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
submission_df = pd.read_csv('../input/sample_submission.csv')
#submission_df
x_test=[]



for i in range(1928):

    

    #print(i)

    

    name=r'../input/test_images/' + str(submission_df['id_code'][i]) + '.png'

    

    #y_true.append(df['diagnosis'][i])

    

    img=cv2.imread(name,ch)

    

    img=cv2.resize(img,(width,height),cv2.INTER_AREA)

    

    x_test.append(img)
x_test=np.array(x_test)



x_test=x_test.reshape(-1,height,width,channel)



x_test=x_test/np.max(x_test)



x_test.shape
pred=model.predict_classes(x_test)
pred.shape
pred[0]
def accuracy(confusion_matrix):

    diagonal_sum = confusion_matrix.trace()

    sum_of_all_elements = confusion_matrix.sum()

    return diagonal_sum / sum_of_all_elements 
req=x_train/np.max(x_train)

predicted=model.predict_classes(req)
from sklearn.metrics import confusion_matrix
def one_hot_to_indices(data):

    indices = []

    for el in data:

        indices.append(list(el).index(1))

    return indices
y_train = one_hot_to_indices(y_train)
y_train
cm=confusion_matrix(y_train,predicted)
accuracy(cm)
del x_test,x_train,x_val
import os

ids=[]

test_path='../input/test_images'

label=[]

a=0

for i in range(len(os.listdir(test_path))):

    

    #idx=submission_df['id_code'][a]

    #ids.append(idx)

    submission_df['diagnosis'][a]=int(pred[a])

    #label.append(int(pred[a]))

    a=a+1



#label=np.array(label,dtype='uint16')



#out=pd.DataFrame({'id_code': ids,'diagnosis':label[:]})



submission_df.to_csv('submission.csv', index=False)
submission_df