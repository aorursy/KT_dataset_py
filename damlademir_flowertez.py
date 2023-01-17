import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

%matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf
import random as rn

import cv2                  
from tqdm import tqdm
import os                   
from random import shuffle  
from PIL import Image
print(os.listdir("../input"))
from keras.preprocessing.image import ImageDataGenerator

X=[]
Y=[]
IMG_SIZE=200
FLOWER_DAISY_DIR='../input/flowers-recognition/flowers/flowers/daisy'
FLOWER_SUNFLOWER_DIR='../input/flowers-recognition/flowers/flowers/sunflower'
FLOWER_TULIP_DIR='../input/flowers-recognition/flowers/flowers/tulip'
FLOWER_DANDI_DIR='../input/flowers-recognition/flowers/flowers/dandelion'
FLOWER_ROSE_DIR='../input/flowers-recognition/flowers/flowers/rose'
def assign_label(img,flower_type):
    return flower_type
#Fotoğrafların yüklenmesi
def make_train_data(flower_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,flower_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            X.append(np.array(img))
            Y.append(str(label))
        else:
            print("image not loaded")

make_train_data('Daisy',FLOWER_DAISY_DIR)
print(len(X))
make_train_data('Sunflower',FLOWER_SUNFLOWER_DIR)
print(len(X))
make_train_data('Tulip',FLOWER_TULIP_DIR)
print(len(X))
make_train_data('Dandelion',FLOWER_DANDI_DIR)
print(len(X))
make_train_data('Rose',FLOWER_ROSE_DIR)
print(len(X))
#Veri setinden random olarak görselleştirme
fig,ax=plt.subplots(5,2)
fig.set_size_inches(17,17)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Y))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Flower: '+Y[l])        
plt.tight_layout()
X=np.array(X)
X=X/255
Y = np.asarray(Y)
pd.unique(Y)#arrayın içinde kaç farklı değer var
#Veri setini train data, validation data olarak bölmek
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42)

#Y dizisi için Label Encoding (i.e. Daisy->0, Rose->1 etc...)
le=LabelEncoder()
y_test=le.fit_transform(y_test)
y_train=le.fit_transform(y_train)
#OneHotEncoding
y_train_binary=to_categorical(y_train,5)
y_test_binary=to_categorical(y_test,5)
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',activation ='relu', input_shape = (IMG_SIZE,IMG_SIZE,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
 
model.add(Conv2D(filters =64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters =128, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(5, activation = "softmax"))
batch_size=80
epochs=200
# Ezberlemeyi önlemek için Data Augmentation 
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
#train data ile fit ve validation data ile predict
History = model.fit_generator(datagen.flow(x_train,y_train_binary, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test_binary),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
# model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data = (x_test,y_test))
#modeli kaydetme
model.save('flowerTez_model.h5')
#sonuçları confusion matrix ile bastırma
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
pred = model.predict_classes(x_test)
cm = confusion_matrix(y_test,pred)

f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm, annot=True, linewidths=0.01,cmap="Purples",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
crlr1 =classification_report(y_test,pred)
print(crlr1)
acclr1 =accuracy_score(y_test,pred) 
print("Accuracy Score:",+acclr1)
#validaiton acc ve loss değerleri grafikleri
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
plot_model_history(History)

# validation setini tahmin etme
pred=model.predict(x_test)
pred_digits=np.argmax(pred,axis=1)

#Doğru ve yanlış tahmin sonuçları
true=0
false=0
for i in range(len(pred)):
   if(pred_digits[i]==y_test[i]):
    true=true+1
   else:
    false=false+1
print("Doğru Tahmin Sayısı:",+true)
print("Yanlış Tahmin Sayısı:",+false)
i=0
prop_class=[]
mis_class=[]

for i in range(len(y_test)):
    if(y_test[i]==pred_digits[i]):
        prop_class.append(i)
    if(len(prop_class)==8):
        break

i=0
for i in range(len(y_test)):
    if(not y_test[i]==pred_digits[i]):
        mis_class.append(i)
    if(len(mis_class)==8):
        break
#Doğru tahminlerin görselleştirilmesi
count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(17,17)
for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[prop_class[count]])
        ax[i,j].set_title("Predicted Flower : "+str(le.inverse_transform([pred_digits[prop_class[count]]]))+"\n"+"Actual Flower : "+str(le.inverse_transform([y_test[prop_class[count]]])))
        plt.tight_layout()
        count+=1
#Yanlış tahminlerin görselleştirilmesi
count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(17,17)
for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[mis_class[count]])
        ax[i,j].set_title("Predicted Flower : "+str(le.inverse_transform([pred_digits[mis_class[count]]]))+"\n"+"Actual Flower : "+str(le.inverse_transform([y_test[mis_class[count]]])))
        plt.tight_layout()
        count+=1

#Yeni çiçek resmi ile predict işlemi
from keras.preprocessing import image
test_image = image.load_img('../input/flowerstest/aycicegi9.jpg', target_size = (IMG_SIZE, IMG_SIZE))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(result)
if result[0][0] == 1:
    prediction = 'Daisy'
elif result[0][1]==1:
    prediction = 'Dandelion'
elif result[0][2]==1:
    prediction = 'Rose'
elif result[0][3]==1:
    prediction='Sunflower'
elif result[0][4]==1:
    prediction='Tulip'
print('so the flower is %s' %prediction)