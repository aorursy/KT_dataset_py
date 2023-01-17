import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import os,cv2


#train_data_preprocessing
train="../input/train/"
data_dir_list = os.listdir(train)

img_data_list=[]
for dataset in data_dir_list:
    img_list=os.listdir(train+'/'+ dataset)
    for img in img_list:
        input_img=cv2.imread(train + '/'+ dataset + '/'+ img )
        ##(for greyscale)input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        img_resize=cv2.resize(input_img,(32,32))
        img_data_list.append(img_resize)

train_x=np.array(img_data_list)
train_x=train_x.astype('float32')
train_x/=255
print(train_x.shape)


#test_data_preprocessing
test='../input/test/'
data_dir_list=os.listdir(test)

img_data_list1=[]
for dataset in data_dir_list:
    img_list=os.listdir(test+'/'+ dataset)
    for img in img_list:
        input_img=cv2.imread(test+'/'+ dataset+'/'+ img )
        #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        img_resize=cv2.resize(input_img,(32,32))
        img_data_list1.append(img_resize)
        
test_img_data=np.array(img_data_list1)
test_img_data=test_img_data.astype('float32')
test_img_data/=255
test_img_data.shape

traind=pd.read_csv("../input/train.csv")
#.value_counts(normalize=False, sort=True)
testd=pd.read_csv("../input/test.csv")
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train_y=le.fit_transform(traind.Class)
print(train_y)
train_y=np_utils.to_categorical(train_y)
train_y


model = Sequential()
model.add(Conv2D(32, 5,5,border_mode="same",input_shape=(32,32,3),activation="relu"))
model.add(Conv2D(32, 3,3,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, 3, 3,activation="relu"))
model.add(Conv2D(64, 3, 3,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(3,activation="softmax"))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=["accuracy"])
model.summary()
model.fit(train_x,train_y,epochs=30,batch_size=100,verbose=1,validation_split=0.2)
#plt.plot(epochs,train_acc,'r', label='train_acc')
#plt.plot(epochs,val_acc,'b', label='val_acc')
#plt.title('train_acc vs val_acc')
#plt.legend()
#plt.figure()

p=model.predict(test_img_data)
print(p.shape)
p=le.inverse_transform(p.all())
p.any()
submit=pd.DataFrame({"CLASS":p.any(),"ID":testd.ID})
submit.sort_values('ID', inplace=True)  
submit.to_csv("s.csv", index=False)
submit
