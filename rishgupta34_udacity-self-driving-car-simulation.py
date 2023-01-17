import cv2
import numpy as np
import pandas as pd
from keras.models import Model,Sequential,save_model
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,Flatten,Dropout,Activation,AveragePooling2D,GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
N_Channels=3
Final_length=70
Final_breadth=200
def read_image(path):
    image=cv2.imread(path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image
def resize(image,Final_length,Final_breadth):
    image=cv2.resize(image,(Final_breadth,Final_length))
    return image
def flip(image,angle):
    r=np.random.randint(1,3)
    if r==2:
        image=np.flip(image,1)
        angle=angle*(-1)
    return image,angle
def process_image(image_path,angle):
    image=read_image(image_path)
    image=image[60:-20,:,:]
    flipped_image,angle=flip(image,angle)
    resized_image=resize(flipped_image,Final_length,Final_breadth)
    return resized_image,angle
def process(image):
    image=image[60:-20,:,:]
    resized_image=resize(image,Final_length,Final_breadth)
    return resized_image
def choose_image(images,angle):
    r=np.random.randint(0,3)
    image_path="../input/img/"+images[r].split('\\')[-1]
    if r==2:
        angle=angle-0.2
    elif r==1:                      ##left
        angle=angle+0.2
    return image_path,angle
def import_dataset():
    dataset=pd.read_csv("../input/driving_log.csv",header=None,names=['center','left','right','steering','throttle','2','speed'])
    images=dataset[['center','left','right']].values
    angles=dataset['steering'].values
    throttle=dataset['throttle'].values
    X_train=np.zeros((images.shape[0],Final_length,Final_breadth,N_Channels))
    Y_train_angle=np.zeros((images.shape[0]))
    Y_train_throttle=np.zeros((images.shape[0]))    
    for i in range(images.shape[0]):
        image_path,angle=choose_image(images[i],angles[i])
        image,angle=process_image(image_path,angle)
        X_train[i]=image
        Y_train_angle[i]=angle
        Y_train_throttle[i]=throttle[i]
    return X_train,Y_train_angle,Y_train_throttle
INPUT_SHAPE=(Final_length,Final_breadth,N_Channels)
data_x,data_y_angle,data_y_throttle=import_dataset()
print(data_x.shape,data_y_angle.shape,data_y_throttle.shape)
data_x,data_y_angle,data_y_throttle=shuffle(data_x,data_y_angle,data_y_throttle)
split=int(0.85*data_x.shape[0])
X_train=data_x[:split,:,:,:]
Y_train_a=data_y_angle[:split]
Y_train_t=data_y_throttle[:split]
X_test=data_x[split:,:,:,:]
Y_test_a=data_y_angle[split:]
Y_test_t=data_y_throttle[split:]
X_test/=255.0
X_train/=255.0
inp=Input(shape=INPUT_SHAPE)
c1=Conv2D(filters=128,strides=(2,2),input_shape=INPUT_SHAPE,kernel_size=(3,3),activation="relu")
c2=Conv2D(filters=64,kernel_size=(3,3),activation="relu")
c19=AveragePooling2D()
c3=Conv2D(filters=48,kernel_size=(3,3),activation="relu")
c9=AveragePooling2D()
c4=Conv2D(filters=40,kernel_size=(3,3),activation="relu")
c17=AveragePooling2D()
# c5=Conv2D(filters=32,kernel_size=(3,3),activation="relu")
c6=Dropout(0.6)
c7=Flatten()
c8=Dense(500,activation="relu")
c10=Dense(250,activation="relu")
c11=Dense(100,activation="relu")
c12=Dense(50,activation="relu")
c13=Dense(25,activation="relu")
c14=Dense(10,activation="relu")
c15=Dense(1)
c16=Dense(1)
out1=c15(c14(c13(c12(c11(c10(c8(c7(c6(c17(c4(c9(c3(c19(c2(c1(inp))))))))))))))))
out2=c16(c14(c13(c12(c11(c10(c8(c7(c6(c17(c4(c9(c3(c19(c2(c1(inp))))))))))))))))
model=Model(inputs=[inp],outputs=[out1,out2])
model.summary()
model.compile(optimizer=Adam(lr=0.0001),loss="mse")
checkpoint = ModelCheckpoint('modelv12_{val_loss:.4f}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
model.fit(X_train,[Y_train_a,Y_train_t],batch_size=16,epochs=35,validation_data=[X_test,[Y_test_a,Y_test_t]],callbacks=[checkpoint],shuffle=True)

