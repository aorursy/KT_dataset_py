from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

sns.set_style({'xtick.bottom':False,
               'ytick.left':False,
               'axes.spines.bottom': False,
               'axes.spines.left': False,
               'axes.spines.right': False,
               'axes.spines.top': False})
im_list = [162+i for i in range(9)]
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15))
for i in enumerate(im_list):
    img = plt.imread("../input/cell_images/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_"+str(i[1])+".png")
    ax=axes[i[0]//3,i[0]%3]
    ax.imshow(img)           
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,15))
img1 = plt.imread("../input/cell_images/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_128.png")
img2 = plt.imread("../input/cell_images/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_131.png")
img3 = plt.imread("../input/cell_images/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_21.png")
img4 = plt.imread("../input/cell_images/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_34.png")

ax = axes[0,0]
ax1 = axes[0,1]
ax2 = axes[1,0]
ax3 = axes[1,1]

ax.imshow(img1)
ax1.imshow(img2)
ax2.imshow(img3)
ax3.imshow(img4)       
datagen = ImageDataGenerator(rescale=1./255,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      validation_split=0.2)
train_data = datagen.flow_from_directory('../input/cell_images/cell_images',
                                                     target_size=(128,128),
                                                     batch_size=32,
                                                     class_mode = 'binary',
                                                     subset = 'training')

validation_data = datagen.flow_from_directory('../input/cell_images/cell_images',
                                                     target_size=(128,128),
                                                     batch_size=32,
                                                     class_mode = 'binary',
                                                     subset = 'validation')
accuracies_ =[]

def train(train_data,validation_data,optimizer,name,epochs=30):
    classifier = Sequential([Conv2D(16,(3,3),input_shape=(128,128,3),activation='relu'),
                        MaxPool2D(2,2),
                        #2nd conv
                        Conv2D(32,(3,3),activation='relu'),
                        MaxPool2D(2,2),
                        #Dropout(0.1),
                        #3rd conv
                        Conv2D(64,(3,3),activation='relu'),
                        MaxPool2D(2,2),
                        #Dropout(0.1),
                        #4th conv
                        Conv2D(128,(3,3),activation='relu'),
                        MaxPool2D(2,2),
                        #Dropout(0.1),
                        
                        Flatten(),
                        Dense(1024,activation='relu'),
                        Dropout(0.2),
                        Dense(512,activation='relu'),
                        Dense(1,activation='sigmoid')])

    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    
    accuracies = classifier.fit(train_data,
                         steps_per_epoch = 100,
                         epochs = epochs,
                         validation_data=validation_data,
                         validation_steps = 10,
                         verbose=1)
    
    acc = pd.DataFrame.from_dict(accuracies.history)
    acc = pd.concat([pd.Series(range(0,30),name='epochs'),acc],axis=1)
    
    fig,(ax,ax1) = plt.subplots(nrows=2,ncols=1,figsize=(16,16))
    sns.lineplot(x='epochs',y='acc',data=acc,ax=ax,color='m')
    sns.lineplot(x='epochs',y='val_acc',data=acc,ax=ax,color='c')
    sns.lineplot(x='epochs',y='loss',data=acc,ax=ax1,color='m')
    sns.lineplot(x='epochs',y='val_loss',data=acc,ax=ax1,color='c')
    ax.legend(labels=['Test Accuracy','Training Accuracy'])
    ax1.legend(labels=['Test Loss','Training Loss'])
    plt.show()
    
    accuracies_.append((name,("Validation Accuracy",accuracies.history['val_acc'][epochs-1]),("Training Accuracy",accuracies.history['acc'][epochs-1])))
    
    return classifier
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
#Adam


adam_classifier = train(train_data,validation_data,Adam(),name='Adam',epochs=40)
# SGD

sgd_classifier = train(train_data,validation_data,SGD(nesterov=True,momentum=0.02),name="SGD",epochs=40)

#RMSprop

rms_classifier = train(train_data,validation_data,RMSprop(),name="RMSprop",epochs=40)
accuracies_
test_img = validation_data[0][0][0]
plt.imshow(test_img)
validation_data[0][1]
(rms_classifier.predict_classes(validation_data[0][0]))
