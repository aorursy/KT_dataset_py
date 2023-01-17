import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



from keras.preprocessing.image import load_img, save_img, img_to_array

import matplotlib.pyplot as plt

from keras.models import Model,Sequential

from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Dense,Dropout,Flatten

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


def preprocess_image(image_path):

    

    from keras.applications import vgg19

    img = load_img(image_path)

    new_width  = 100

    new_height = 100

    img = img.resize((new_width, new_height))

    img = img_to_array(img)

    img=img.astype(int)

    return img



Path = '../input/images/images/'

plt.figure()

plt.title("First Image",fontsize=20)

img1 = load_img(Path+'Paul_Gauguin/Paul_Gauguin_3.jpg')

new_width  = 200

new_height = 200

img1 = img1.resize((new_width, new_height))

img1 = img_to_array(img1)

plt.imshow(img1.astype(int))

all_paint=[]

artists=os.listdir(Path)

for artist in artists:

    paint=os.listdir(Path+artist)

    for paints in paint:

        all_paint.append(preprocess_image(Path+artist+'/'+paints))



len(all_paint)
all_paint=np.stack(all_paint)

all_paint=all_paint.astype('float32')/255

all_paint.shape
noise_factor=0.5

x_train_noisy=all_paint+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=all_paint.shape)

x_train_noisy=np.clip(x_train_noisy,0.,1.)
n=17

plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(all_paint[i])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()



plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(x_train_noisy[i])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()



x_test_noisy=x_train_noisy[round(len(x_train_noisy)*80/100):,:,:,:]

x_test=all_paint[round(len(all_paint)*80/100):,:,:,:]

x_train_noisy=x_train_noisy[:round(len(x_train_noisy)*80/100),:,:,:]

all_paint=all_paint[:round(len(all_paint)*80/100),:,:,:]
model=Sequential()



model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(100,100,3)))

model.add(MaxPooling2D((2,2),padding='same'))

model.add(Conv2D(32,(3,3),activation='relu',padding='same'))

model.add(MaxPooling2D((2,2),padding='same'))





model.add(Conv2D(32,(3,3),activation='relu',padding='same'))

model.add(UpSampling2D((2,2)))

model.add(Conv2D(32,(3,3),activation='relu',padding='same'))

model.add(UpSampling2D((2,2)))

model.add(Conv2D(3,(3,3),activation='sigmoid',padding='same'))

model.compile(optimizer='adam',loss='binary_crossentropy')

model.fit(x_train_noisy,all_paint,epochs=30,batch_size=64,shuffle=True,validation_data=(x_test_noisy,x_test))

essai=model.predict(x_test_noisy)

n=17

plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(x_test_noisy[i])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()





plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(essai[i])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()



plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(x_test[i])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()







x_train_noisy=all_paint.copy()

x_test_noisy=x_test.copy()
size=15

for element in x_train_noisy:

    number=np.random.randint(low=1, high=85, size=1)

    number=number[0]

    element[number:number+size,number:number+size,:]=1.0

for element in x_test_noisy:

    number=np.random.randint(low=1, high=85, size=1)

    number=number[0]

    element[number:number+size,number:number+size,:]=1.0


plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(all_paint[i])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()



plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(x_train_noisy[i])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
model_blank=Sequential()



model_blank.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(100,100,3)))

model_blank.add(MaxPooling2D((2,2),padding='same'))

model_blank.add(Conv2D(32,(3,3),activation='relu',padding='same'))

model_blank.add(MaxPooling2D((2,2),padding='same'))





model_blank.add(Conv2D(32,(3,3),activation='relu',padding='same'))

model_blank.add(UpSampling2D((2,2)))

model_blank.add(Conv2D(32,(3,3),activation='relu',padding='same'))

model_blank.add(UpSampling2D((2,2)))

model_blank.add(Conv2D(3,(3,3),activation='sigmoid',padding='same'))



model_blank.compile(optimizer='adam',loss='binary_crossentropy')

model_blank.fit(x_train_noisy,all_paint,epochs=30,batch_size=64,shuffle=True,validation_data=(x_test_noisy,x_test))
essai=model_blank.predict(x_test_noisy)

n=17

plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(x_test_noisy[i+2])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()





plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(essai[i+2])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()



plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(x_test[i+2])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
x_train_noisy=all_paint.copy()

x_test_noisy=x_test.copy()
size=4

for element in x_train_noisy:

    number=np.random.randint(low=1, high=100, size=1)

    number=number[0]

    element[number:number+size,:,:]=1.0

for element in x_test_noisy:

    number=np.random.randint(low=1, high=100, size=1)

    number=number[0]

    element[number:number+size,:,:]=1.0
plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(all_paint[i])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()



plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(x_train_noisy[i])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
model_line=Sequential()



model_line.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(100,100,3)))

model_line.add(MaxPooling2D((2,2),padding='same'))

model_line.add(Conv2D(32,(3,3),activation='relu',padding='same'))

model_line.add(MaxPooling2D((2,2),padding='same'))





model_line.add(Conv2D(32,(3,3),activation='relu',padding='same'))

model_line.add(UpSampling2D((2,2)))

model_line.add(Conv2D(32,(3,3),activation='relu',padding='same'))

model_line.add(UpSampling2D((2,2)))

model_line.add(Conv2D(3,(3,3),activation='sigmoid',padding='same'))



model_line.compile(optimizer='adam',loss='binary_crossentropy')

model_line.fit(x_train_noisy,all_paint,epochs=30,batch_size=64,shuffle=True,validation_data=(x_test_noisy,x_test))
essai=model_line.predict(x_test_noisy)

n=17

plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(x_test_noisy[i+2])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()





plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(essai[i+2])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()



plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(x_test[i+2])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
x_train_noisy=all_paint.copy()

x_test_noisy=x_test.copy()
size=15

for element in x_train_noisy:

    number=np.random.randint(low=1, high=85, size=1)

    number=number[0]

    element[number:number+size,number:number+size,:]=0.0

for element in x_test_noisy:

    number=np.random.randint(low=1, high=85, size=1)

    number=number[0]

    element[number:number+size,number:number+size,:]=0.0
model_black=Sequential()



model_black.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(100,100,3)))

model_black.add(MaxPooling2D((2,2),padding='same'))

model_black.add(Conv2D(32,(3,3),activation='relu',padding='same'))

model_black.add(MaxPooling2D((2,2),padding='same'))





model_black.add(Conv2D(32,(3,3),activation='relu',padding='same'))

model_black.add(UpSampling2D((2,2)))

model_black.add(Conv2D(32,(3,3),activation='relu',padding='same'))

model_black.add(UpSampling2D((2,2)))

model_black.add(Conv2D(3,(3,3),activation='sigmoid',padding='same'))



model_black.compile(optimizer='adam',loss='binary_crossentropy')

model_black.fit(x_train_noisy,all_paint,epochs=30,batch_size=64,shuffle=True,validation_data=(x_test_noisy,x_test))
essai=model_black.predict(x_test_noisy)

n=17

plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(x_test_noisy[i+2])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()





plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(essai[i+2])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()



plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(x_test[i+2])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
essai=model_blank.predict(x_test_noisy)

n=17

plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(x_test_noisy[i+2])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()





plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(essai[i+2])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()



plt.figure(figsize=(100,4))

for i in range (12,n):

    i=i+1

    ax=plt.subplot(1,n,i)

    plt.imshow(x_test[i+2])

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()