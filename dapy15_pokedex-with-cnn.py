!wget https://www.dropbox.com/s/xmopr2altgp8f0a/dataset.zip?dl=0 -O dataset.zip
!unzip dataset
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.layers import *
from keras.models import *
from keras import losses
import os
import shutil
folder = os.listdir("Train")
print(folder)
if not os.path.isdir("Val"):
  os.mkdir("Val")
!ls
for c in folder:
  p=os.path.join("Val",c)
  if not os.path.isdir(p):
    os.mkdir(p)
for f in folder:
  path="Train/"+f
  print(f+ " "+ str(len(os.listdir(path))))
split = 0.9
for f in os.listdir("Train"):
    path = "Train/"+f
    imgs=os.listdir(path)
    split_size= int(split*len(imgs))
    file_to_move=imgs[split_size:]
  
    for img_f in file_to_move:
        src=os.path.join(path,img_f)
        dest=os.path.join("Val/"+f,img_f)
        #print(src)
        #print(dest)
        shutil.move(src,dest)
print("For training data:- ")
for f in folder:
    path="Train/"+f
    print(f+ " "+ str(len(os.listdir(path))))

print("\n For vaidation data:- ")
for f in folder:
    path="Val/"+f
    print(f+ " "+ str(len(os.listdir(path))))
# Image visualisation
def ImgVis(Path):
    img=image.load_img(Path)
    x=image.img_to_array(img)/255.0
    print(x.shape)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
path="Train/Charmander/00000002.jpg"
ImgVis(path)
path = "Train/Bulbasaur/00000013.png"
ImgVis(path)
train_gen = image.ImageDataGenerator(rescale=1.0/255,
                                    horizontal_flip=True,
                                    shear_range=0.2,
                                    zoom_range=0.2,)

val_gen = image.ImageDataGenerator(rescale=1/255.0)

train_generator = train_gen.flow_from_directory("Train/",
                                               target_size=(224,224),
                                               batch_size=32,
                                               class_mode='categorical')

val_generator = val_gen.flow_from_directory("Val/",
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='categorical')
for x,y in train_generator:
    print(x.shape)
    print(y.shape)
    break
print(train_generator.class_indices)
print("-"*15)
print(val_generator.class_indices)
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(loss=losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
hist = model.fit(train_generator,
                          steps_per_epoch=47,
                          epochs=20,
                          validation_data=val_generator,
                          validation_steps=6)
# Visualising the accuracy
plt.style.use("seaborn")

plt.plot(hist.history['accuracy'],label="training acc",c='red')
plt.plot(hist.history['val_accuracy'],label="validation acc",c='blue')
plt.legend()
plt.show()
path="Test/"
y_df=pd.read_csv(path+"sample_submission.csv")
y_df.shape
y_df.head(7)
y_df.drop(['Class'],inplace=True,axis=1)
y_df.head(7)
y_df=y_df.values.reshape((-1,))
print(y_df.shape)
from pathlib import Path
pi_test=Path("Test/images/")

image_data_test=[]
label_test=[]

for image_path in pi_test.glob("*"):
  #label=(str(image_path).split("\\")[-1]) this is not woring in goole colab
  label=(str(image_path).split("/")[-1])

  img=image.load_img(image_path,target_size=(224,224,3))
  image_array=image.img_to_array(img)/255.0
  image_data_test.append(image_array)
  label_test.append(label)
image_data_test=np.array(image_data_test)
print(label_test[:5])
print(label_test[0])
print(image_data_test.shape)
print(len(label_test))
y_predicted=model.predict_classes(image_data_test)
y_predicted
y_pre=[]
for i in range(image_data_test.shape[0]):
  index=label_test.index(y_df[i])
  y=y_predicted[index]
  y_pre.append((y_df[i],y))
y_pre=np.array(y_pre)
df_pred=pd.DataFrame(data=y_pre,columns=['Name','Class'])
df_pred.head(7)
dict_pred = dict(df_pred.values.tolist())
print(dict_pred)
label_pok = {
    0 : "Aerodactyl",  
    1 : "Bulbasaur",  
    2 : "Charmander", 
    3 : "Dratini",  
    4 : "Fearow",  
    5 : "Mewtwo",  
    6 : "Pikachu",  
    7 : "Psyduck",  
    8 : "Spearow",  
    9 : "Squirtle"
}
path = "/kaggle/working/Test/images/test_32.jpg"
ImgVis(path)
print(dict_pred['test_32.jpg'])
print(label_pok[int(dict_pred['test_32.jpg'])])
path = "/kaggle/working/Test/images/test_9.jpg"
ImgVis(path)
print(dict_pred['test_9.jpg'])
print(label_pok[int(dict_pred['test_9.jpg'])])
