#Kütüphanelerin tanımlanması

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

#dataset konumu

TSHIRTS_DIR='../input/datasets/datasets/tshirt' 

TIES_DIR='../input/datasets/datasets/tie'

PANTS_DIR='../input/datasets/datasets/pants'

SOCKS_DIR='../input/datasets/datasets/sock'

SHIRTS_DIR='../input/datasets/datasets/shirt'
#Labellama işlemi

def assign_label(img, cloth_type):

    return cloth_type
#Fotoğrafların yüklenmesi

def make_train_data(cloth_type,DIR):

    for img in tqdm(os.listdir(DIR)):

        label=assign_label(img,cloth_type)

        path = os.path.join(DIR,img)

        img = cv2.imread(path,cv2.IMREAD_COLOR)

        if img is not None:

            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

            X.append(np.array(img))

            Y.append(str(label))

        else:

            print("image not loaded")
make_train_data('Tshirt',TSHIRTS_DIR)

print(len(X))
make_train_data('Tie',TIES_DIR)

print(len(X))
make_train_data('Pant',PANTS_DIR)

print(len(X))
make_train_data('Sock',SOCKS_DIR)

print(len(X))
make_train_data('Shirt',SHIRTS_DIR)

print(len(X))
#Veri setinden random olarak görselleştirme

fig,ax=plt.subplots(5,2)

fig.set_size_inches(17,17)

for i in range(5):

    for j in range (2):

        l=rn.randint(0,len(Y))

        ax[i,j].imshow(X[l])

        ax[i,j].set_title('Cloth is: '+Y[l])        

plt.tight_layout()
X=np.array(X)

X=X/255

Y = np.asarray(Y)
pd.unique(Y) #arrayın içinde kaç farklı değer var
#Veri setini train data, validation data olarak bölmek

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42)
#Y dizisi için Label Encoding (i.e. Pants->0, T-shirts->1 etc...)

le=LabelEncoder()

y_test=le.fit_transform(y_test)

y_train=le.fit_transform(y_train)
#OneHotEncoding

y_train_binary=to_categorical(y_train,5)

y_test_binary=to_categorical(y_test,5)
#Dropout add

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

        featurewise_center=False, 

        samplewise_center=False,  

        featurewise_std_normalization=False,

        samplewise_std_normalization=False, 

        zca_whitening=False,  

        rotation_range=10,  

        zoom_range = 0.1,

        width_shift_range=0.2, 

        height_shift_range=0.2,

        horizontal_flip=True, 

        vertical_flip=False)  

datagen.fit(x_train)
#ACC metriğine göre compile etmesi için 'Adam' optimizerini kullanıyorum

model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
#train data ile fit ve validation data ile predict

History = model.fit_generator(datagen.flow(x_train,y_train_binary, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_test,y_test_binary),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
#Oluşan modelin kaydedilmesi

model.save('clothRecognize.h5')
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

pred = model.predict_classes(x_test)

cm = confusion_matrix(y_test,pred)
#Acc değerinin yazılması

crlr1 =classification_report(y_test,pred)

print(crlr1)

acclr1 =accuracy_score(y_test,pred) 

print("Accuracy Score:",+acclr1)
#validaiton ve train accuracy ve loss değerleri grafikleri

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
#doğru ve yanlış tahminleri arraylere append etme 

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

        ax[i,j].set_title("Predicted Cloth : "+str(le.inverse_transform([pred_digits[prop_class[count]]]))+"\n"+"Actual Cloth : "+str(le.inverse_transform([y_test[prop_class[count]]])))

        plt.tight_layout()

        count+=1
#Yanlış tahminlerin görselleştirilmesi

count=0

fig,ax=plt.subplots(2,2)

fig.set_size_inches(17,17)

for i in range (2):

    for j in range (2):

        ax[i,j].imshow(x_test[mis_class[count]])

        ax[i,j].set_title("Predicted Cloth : "+str(le.inverse_transform([pred_digits[mis_class[count]]]))+"\n"+"Actual Cloth : "+str(le.inverse_transform([y_test[mis_class[count]]])))

        plt.tight_layout()

        count+=1
#Yeni giysi resmi ile predict işlemi

print('Test Image is tshirt')

from keras.preprocessing import image

test_image = image.load_img('../input/datasets/datasets/test_tshirt/test.JPG', target_size = (IMG_SIZE, IMG_SIZE))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = model.predict(test_image)

print(result)

if result[0][0] == 1:

    prediction = 'Tshirt'

elif result[0][1]==1:

    prediction = 'Tie'

elif result[0][2]==1:

    prediction = 'Pant'

elif result[0][3]==1:

    prediction='Sock'

elif result[0][4]==1:

    prediction='Shirt'

print('Predicted cloth is %s' %prediction)