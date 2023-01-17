 
import os
import pandas as pd
 
from tensorflow.keras import layers
from tensorflow.keras import Model
 
from tensorflow.keras.applications.inception_v3 import InceptionV3
 
 
from tensorflow.keras.models import Model,Sequential
 
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.optimizers import RMSprop , Adam , Adamax , Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
local_weights_file = "../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
pretrained_model = InceptionV3(input_shape=(224,224, 3), 
                              include_top=False,
                              weights = None)
pretrained_model.load_weights(local_weights_file)
for layer in pretrained_model.layers:
    layer.trainable = False
pretrained_model.summary()
last_layer = pretrained_model.get_layer('mixed7')
last_output = last_layer.output
x = layers.Flatten()(last_output)
x = layers.Dropout(0.5)(x)
#x = layers.BatchNormalization()(x)
x  = layers.Dense(512 , activation='relu')(x)
x = layers.Dropout(0.5)(x)
 
#x = layers.BatchNormalization()(x)
   
x = layers.Dense(6, activation='softmax')(x)
model = Model(pretrained_model.input, x)
model.summary()
model.compile(optimizer = Adam(lr=0.0001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])
train_dir = '../input/intel-image-classification/seg_train/seg_train'
train_datagen = ImageDataGenerator(
         
        rescale=1./255,
        #featurewise_std_normalization = True , 
        #featurewise_center=True,
         
        fill_mode = 'nearest',  
        rotation_range=40,  
        #zoom_range = 0.4,  
        #shear_range = 0.4,
         
        
        horizontal_flip=0.5 
         
        
       
        )
valid_dir = r"../input/intel-image-classification/seg_test/seg_test"
valid_datagen = ImageDataGenerator(
         
        rescale=1./255,
        #featurewise_std_normalization = True , 
        #featurewise_center=True,
         
        fill_mode = 'nearest',  
        rotation_range=40,  
        #zoom_range = 0.4,  
        #shear_range = 0.4,
         
        
        horizontal_flip=0.5 
         
        
       
        )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=128,
                                                   target_size=(224, 224),
                                                    shuffle = True, 
                                                    
                                                   class_mode='categorical' 
                                                    )
valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                    batch_size=128,
                                                   target_size=(224, 224),
                                                    shuffle = True, 
                                                    
                                                   class_mode='categorical' 
                                                    )
history = model.fit(train_generator,
                   epochs=10,
                    validation_data = valid_generator , 
                    batch_size = 128
                  
                   
                  )
plt.plot(history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
plt.plot(history.history['accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
path = r"../input/intel-image-classification/seg_pred/seg_pred/10038.jpg"
img = image.load_img(path, target_size=(224, 224))
x = image.img_to_array(img)/255
x = np.expand_dims(x, axis=0)
classes = model.predict([x])
pred_labels = np.argmax(classes, axis = 1)
labels = {
    0 : "buildings" , 
    1 : "forest" , 
    2 : "glacier" , 
    3 : "mountain" , 
    4 : "sea" , 
    5 : "street"
}
for i in pred_labels:
    print(labels[i])
path = "../input/intel-image-classification/seg_pred/seg_pred/"
images = []

x = os.listdir("../input/intel-image-classification/seg_pred/seg_pred")

for i in x:
    img = path + i
    images.append(img)



outputs = []
ids = []
for i in images:
    img = image.load_img(i , target_size = (224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img , axis=0)
    c = model.predict([img])
    pred_labels = np.argmax(c, axis = 1)
    la = labels[pred_labels[0]]
    
    outputs.append(la)
    ids.append(i)
    
    
data = pd.DataFrame({
    "Image" : ids , 
    "Label" : outputs
})
data.head(10)
model.save("Intel_Image_classification.h5")