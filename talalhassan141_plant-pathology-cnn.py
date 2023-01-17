import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import cv2
import re 

import matplotlib.pyplot as plot
import matplotlib.gridspec as gridspec
import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics

from keras.utils.vis_utils import model_to_dot
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint

from glob import glob
df_train = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv") # read train data

df_test = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv") # read test data
#sort image name on the basis of numerical value in the name

def sort_names( l ): 
#     """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
def load_images_from_folder(folder,file_type):
    images = []
   
    filenames = glob(folder + "*.jpg")
    filenames = sort_names(filenames)
    
    for filename in filenames:
        
        filename = filename.split('images/')[1]
        
        if(file_type == filename[:5]):
            
            img = cv2.imread(folder + filename)
            img = cv2.resize(img,(150,150))
                        
            if img is not None:                
                images.append(img)
                print("Reading Image ",filename)
                
    return images
#healthy = 0,multiple_diseases = 1, rust = 2, scab = 3

def convert_multiclass_label(dataset):
    label = []
    for i in range(len(dataset)):
        if (dataset[i:i+1]["healthy"][i] == 1):
            label.append(0)
        elif (dataset[i:i+1]["multiple_diseases"][i] == 1):
            label.append(1)
        elif (dataset[i:i+1]["rust"][i] == 1):
            label.append(2)
        elif (dataset[i:i+1]["scab"][i] == 1):
            label.append(3)
        
    return label
#healthy = 0,multiple_diseases = 1, rust = 2, scab = 3
#Encode multiple columns of labels to one column i.e. 0,1,2,3

#FOR TRAIN
train_label = pd.DataFrame(convert_multiclass_label(df_train)).rename(columns={0: "label"})
train_temp = pd.concat([df_train,train_label],axis = 1)

train_temp.head(3)
#load train images
train_images = load_images_from_folder("../input/plant-pathology-2020-fgvc7/images/","Train")
#load test images
test_images = load_images_from_folder("../input/plant-pathology-2020-fgvc7/images/","Test_")
#Convert Train image data to np array

X_train = np.array(train_images)
Y_train = np.array(train_temp["label"])

#Convert Test image data to np array

X_test = np.array(test_images)
print("Shape of Train Images:",X_train.shape)
print("Shape of Train Labels:",Y_train.shape)

print("Shape of Test Images:",X_test.shape)
# Class : healthy = 0,multiple_diseases = 1, rust = 2, scab = 3
#Plot sample images

f,ax = plot.subplots(5,5) 
f.subplots_adjust(0,0,3,3)
for i in range(0,5,1):
    for j in range(0,5,1):
        rnd_number = randint(0,len(X_train))
        ax[i,j].imshow(X_train[rnd_number])
        ax[i,j].set_title("Train_" + str(rnd_number) + ".jpg \nClass " + str(Y_train[rnd_number]))
        ax[i,j].axis('off')
#Build CNN model with layers of different activation function

model = Models.Sequential()

model.add(Layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))
model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(140,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(100,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(50,kernel_size=(3,3),activation='relu'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Flatten())
model.add(Layers.Dense(180,activation='relu'))
model.add(Layers.Dense(100,activation='relu'))
model.add(Layers.Dense(50,activation='relu'))
model.add(Layers.Dropout(rate=0.5))
model.add(Layers.Dense(4,activation='softmax'))

model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary() #define model summary
#train ccn model with train data 
cnn_model_trained = model.fit(X_train,Y_train,epochs=40,validation_split=0.10)
#plot accuracy and loss during epochs 

plot.plot(cnn_model_trained.history['accuracy'])  #train accuracy
plot.plot(cnn_model_trained.history['val_accuracy']) #Validation accuracy
plot.title('Model accuracy')
plot.ylabel('Accuracy')
plot.xlabel('Epoch')
plot.legend(['Train', 'Validation'], loc='upper left')
plot.show()

plot.plot(cnn_model_trained.history['loss'])
plot.plot(cnn_model_trained.history['val_loss'])
plot.title('Model loss')
plot.ylabel('Loss')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()
#prediction classes of unseen test images having no label

pred_class = model.predict_classes(X_test)

pred_labels_df = pd.DataFrame(pred_class).rename(columns={0: "label"})

Prediction = pd.concat([df_test,pred_labels_df],axis = 1)

Prediction.head(5)
#predict probabilities of images for each class

pred_prob_array = model.predict(X_test)
pred_prob_array = pd.DataFrame(pred_prob_array).rename(columns={0: "healthy",1:"multiple_diseases",2:"rust",3:"scab"})
Prediction_prob = pd.concat([df_test,pred_prob_array],axis = 1)
Prediction_prob.head(5)
Prediction_prob.to_csv("plant_pathology_cnn_prob.csv")
#healthy = 0,multiple_diseases = 1, rust = 2, scab = 3
#plot result images 

fig = plot.figure(figsize=(30, 30))
outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)

for i in range(25):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.1)
    rnd_number = randint(0,len(X_test))
    pred_image = np.array([X_test[rnd_number]])
    pred_class = model.predict_classes(pred_image)[0]
    pred_prob = model.predict(pred_image).reshape(4)
    for j in range(2):
        if (j%2) == 0:
            ax = plot.Subplot(fig, inner[j])
            ax.imshow(pred_image[0])
            ax.set_title("Test_" + str(rnd_number) + ".jpg\n Class " + str(pred_class))
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
        else:
            ax = plot.Subplot(fig, inner[j])
            ax.bar([0,1,2,3],pred_prob)
            fig.add_subplot(ax)


fig.show()