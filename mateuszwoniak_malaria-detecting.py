import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import cv2 
infected = os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized")
uninfected = os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected")


data = []
labels = []

#obrobka i dodadanie zakazonych komorek
for i in infected:
    try:
            img = cv2.imread("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/"+i)
            img = cv2.resize(img,(50,50))
            img_array = np.array(img)
            img_array = img_array/255
            data.append(img_array)
            labels.append(1)
    except:
        print("!")
print(len(data))


#zaladownie zdrowych komorek
for i in uninfected:
    try:
        img = cv2.imread("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/"+i)
        img = cv2.resize(img,(50,50))
        img_array = np.array(img)
        img_array = img_array/255
        data.append(img_array)
        labels.append(0)
    except:
        print("!")
print(len(data))

#zrezygnowałem z augmentacji danych, z powodu prostego problemu i dużej dostępności zdjęć. 
data = np.array(data)
print(data.shape)
labels = np.array(labels)
print(labels.shape)
from sklearn.model_selection import train_test_split

#podzial na zbiory testowe i ewaluacyjne, i treningowe

train_x , x , train_y , y = train_test_split(data , labels , 
                                            test_size = 0.2 ,
                                            random_state = 15)

eval_x , test_x , eval_y , test_y = train_test_split(x , y , 
                                                    test_size = 0.5 , 
                                                    random_state = 15)
print(f"train: {train_x.shape}, val: {eval_x.shape}, test: {test_x.shape}")
plt.figure(1 , figsize = (15 , 9))
n = 0 
for i in range(25):
    n += 1 
    r = np.random.randint(0 , train_x.shape[0] , 1)
    plt.subplot(5 , 5 , n)
    plt.subplots_adjust(hspace = 0.3 , wspace = 0.3)
    plt.imshow(train_x[r[0]])
    plt.title(f"{'Infected' if train_y[r[0]] == 1 else 'Unifected'} : {train_y[r[0]]}")
    plt.xticks([]) , plt.yticks([])
    
plt.show()
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation ='relu', input_shape = (50,50,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation ='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation ='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))





model.summary()

model.compile(loss ="binary_crossentropy", optimizer='rmsprop', metrics=['acc'])
history = model.fit(train_x, train_y, epochs=20, batch_size=16, validation_data=(eval_x,eval_y))
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs, acc, 'bo', label='Dokladnosc trenowania')
plt.plot(epochs,val_acc, 'b', label='Dokladnosc walidacji')

plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Strata trenowania')
plt.plot(epochs,val_loss, 'b', label='Strata walidacji')

plt.legend()

plt.show()

_, test_acc = model.evaluate(test_x, test_y)
print(test_acc)

