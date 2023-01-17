!wget -O "dog_breed_classification_ai_challenge-dataset.zip" "https://dockship-job-models.s3.ap-south-1.amazonaws.com/5d1d683b041da2669eed8b591fba65ac?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIDOPTEUZ2LEOQEGQ%2F20200913%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20200913T161029Z&X-Amz-Expires=1800&X-Amz-Signature=27494dbb404c6781e32e42641d149b5a1960d686093cda869675d8872553a17c&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22dog_breed_classification_ai_challenge-dataset.zip%22"
!unzip ./dog_breed_classification_ai_challenge-dataset.zip
import pandas as pd
import numpy as np
df=pd.read_csv('./dataset/train.csv')
df.head()
img_with_path=[]
for i in list(df["Filename"].values):
  i='./dataset/train/'+i
  img_with_path.append(i)


from keras.preprocessing import image
img=image.load_img(img_with_path[0],target_size=(224,224))
img=image.img_to_array(img)/255
import matplotlib.pyplot as plt

plt.imshow(img)
label=df['Labels'].unique()
len(label)
dic={}
rev_dic={}
s=0
for i in label:
  dic[i]=s
  rev_dic[s]=i
  s+=1
y=[]
for i in list(df["Labels"].values):
  y.append(dic[i])
samples=[]
for i,j in zip(img_with_path,y):
  samples.append([i,j])
len(samples)
from sklearn.model_selection import train_test_split
train_sample,test_sample=train_test_split(samples,test_size=0.1, random_state=42)
from sklearn.utils import shuffle
def data_generator(samples,batch_size=32,shuffle_data=True,img_size=229):
    num_samples=len(samples)
    while True:
        samples=shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples=samples[offset:offset+batch_size]
            
            X_train=[]
            y_train=[]
            
            for batch_sample in batch_samples:
                img=image.load_img(batch_sample[0],target_size=(img_size,img_size))
                img=image.img_to_array(img)    
                img=img/255
                X_train.append(img)
                label=batch_sample[1]
                label=to_categorical(label, num_classes=120)
                y_train.append(label)
            
            X_train=np.array(X_train)
            y_train=np.array(y_train)
            
            yield X_train,y_train
    
    
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input,decode_predictions
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input,decode_predictions
from keras.preprocessing import image
inception=InceptionV3(include_top=True,weights="imagenet")
x=inception.layers[-2].output
fc1=Dense(120,activation='softmax')(x)
my_model=Model(inputs=inception.input,outputs=fc1)
from keras.optimizers import Adam
adam=Adam(learning_rate=1e-4)
for l in my_model.layers[:-2]:
    #print(l)
    l.trainable = False
my_model.compile(optimizer=adam,loss = "categorical_crossentropy",metrics=["accuracy"])
train_datagen=data_generator(train_sample)
val_datagen=data_generator(test_sample)
hist=my_model.fit_generator(train_datagen,epochs=15,steps_per_epoch=len(train_sample)//32,validation_data=val_datagen,validation_steps=len(test_sample)//32)
import matplotlib.pyplot as plt
plt.plot(my_model.history.history['accuracy'])
plt.plot(my_model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(my_model.history.history['loss'])
plt.plot(my_model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
