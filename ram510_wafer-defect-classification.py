import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

from matplotlib.image import imread



from sklearn.metrics import classification_report,confusion_matrix



from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPool2D

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam



import warnings

warnings.filterwarnings('ignore')
%%time

b_d=os.listdir('/kaggle/input/wafermap/WaferMap/balanced')

for i in b_d:

   print( i,len(os.listdir('/kaggle/input/wafermap/WaferMap/balanced/'+i)))

print("==================================================================")

test_d=os.listdir('/kaggle/input/wafertestdata/B_Test')

for i in test_d:

   print( i,len(os.listdir('/kaggle/input/wafertestdata/B_Test/'+i)))
b_path='/kaggle/input/wafermap/WaferMap/balanced/'

sample_wafer=[]

for i in b_d:

    sample_wafer.append(b_path+i+'/'+os.listdir(b_path+i+'/')[0])

sample_wafer

   
# plt.figure(figsize=(24,12))

f, axarr = plt.subplots(3,3,figsize=(24,12))

m=0

for i in range(3):

    for j in range(3):

        axarr[i,j].imshow(imread(sample_wafer[m]))

        axarr[i,j].set_title(os.path.basename(sample_wafer[m])) 

        m+=1

 
def dimension(path,dim1,dim2):

    for image_filename in os.listdir(path): 

        image=imread(path+image_filename)

        d1,d2,channels=image.shape

        dim1.append(d1)

        dim2.append(d2)

#         print(channels)

    return dim1,dim2



loc_dim1=[]

loc_dim2=[]

loc_dim1,loc_dim2=dimension('/kaggle/input/wafermap/WaferMap/balanced/Loc/',loc_dim1,loc_dim2)



edgeRing_dim1=[]

edgeRing_dim2=[]

edgeRing_dim1,edgeRing_dim2=dimension('/kaggle/input/wafermap/WaferMap/balanced/Edge-ring/',edgeRing_dim1,edgeRing_dim2)



edgeLoc_dim1=[]

edgeLoc_dim2=[]

edgeLoc_dim1,edgeLoc_dim2=dimension('/kaggle/input/wafermap/WaferMap/balanced/Edge-loc/',edgeLoc_dim1,edgeLoc_dim2)



center_dim1=[]

center_dim2=[]

center_dim1,center_dim2=dimension('/kaggle/input/wafermap/WaferMap/balanced/Center/',center_dim1,center_dim2)



random_dim1=[]

random_dim2=[]

random_dim1,random_dim2=dimension('/kaggle/input/wafermap/WaferMap/balanced/Random/',random_dim1,random_dim2)



scratch_dim1=[]

scratch_dim2=[]

scratch_dim1,scratch_dim2=dimension('/kaggle/input/wafermap/WaferMap/balanced/Scratch/',scratch_dim1,scratch_dim2)



nearFull_dim1=[]

nearFull_dim2=[]

nearFull_dim1,nearFull_dim2=dimension('/kaggle/input/wafermap/WaferMap/balanced/Near-Full/',nearFull_dim1,nearFull_dim2)



donut_dim1=[]

donut_dim2=[]

donut_dim1,donut_dim2=dimension('/kaggle/input/wafermap/WaferMap/balanced/Donut/',donut_dim1,donut_dim2)



none_dim1=[]

none_dim2=[]

none_dim1,none_dim2=dimension('/kaggle/input/wafermap/WaferMap/balanced/None/',donut_dim1,donut_dim2)

    



np.mean(scratch_dim1)


# img_gen=ImageDataGenerator()

image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees

                               width_shift_range=0.10, # Shift the pic width by a max of 5%

                               height_shift_range=0.10, # Shift the pic height by a max of 5%

                               rescale=1/255, # Rescale the image by normalzing it.

                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)

                               zoom_range=0.1, # Zoom in by 10% max

                               horizontal_flip=True, # Allo horizontal flipping

                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value

                              )
plt.imshow(imread(sample_wafer[4]))
plt.imshow(image_gen.random_transform(imread(sample_wafer[4])))
train_path='/kaggle/input/wafermap/WaferMap/balanced'

test_path='/kaggle/input/wafertestdata/B_Test'

# image_gen.flow_from_directory(train_path)

# image_gen.flow_from_directory(test_path)
batch_size = 16

img_shape=(64,65,4)
model = Sequential()



## FIRST SET OF LAYERS

# CONVOLUTIONAL LAYER

# POOLING LAYER

model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=img_shape, activation='relu',))

model.add(MaxPool2D(pool_size=(2, 2)))



## SECOND SET OF LAYERS

# CONVOLUTIONAL LAYER

# POOLING LAYER

model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=img_shape, activation='relu',))

model.add(MaxPool2D(pool_size=(2, 2)))



# model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=img_shape, activation='relu',))

# model.add(MaxPool2D(pool_size=(2, 2)))





# FLATTEN IMAGES FROM 64 by 65 to 4160 BEFORE FINAL LAYER

model.add(Flatten())



# 256 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.3))

# model.add(Dense(128, activation='sigmoid'))

# model.add(Dropout(0.5))

# LAST LAYER IS THE CLASSIFIER, THUS 9 POSSIBLE CLASSES

model.add(Dense(9, activation='softmax'))





model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.summary()


early_stop = EarlyStopping(monitor='val_loss',patience=5)
train_image_gen = image_gen.flow_from_directory(train_path,

                                               target_size=img_shape[:2],

                                                color_mode='rgba',

                                               batch_size=batch_size,

                                               class_mode='categorical')
test_image_gen = image_gen.flow_from_directory(test_path,

                                               target_size=img_shape[:2],

                                                color_mode='rgba',

                                               batch_size=batch_size,

                                               class_mode='categorical',

                                              shuffle=False)
test_image_gen.class_indices.keys()
train_image_gen.class_indices
results = model.fit_generator(train_image_gen,epochs=50,

                              validation_data=test_image_gen,

                             callbacks=[early_stop])
target_names=['Center', 'Donut', 'Edge-loc', 'Edge-ring', 'Loc', 'Near-Full', 'None', 'Random', 'Scratch']

Y_pred=model.predict_generator(test_image_gen,855)

y_pred=np.argmax(Y_pred,axis=1)

con_matrix=confusion_matrix(test_image_gen.classes,y_pred)

plt.figure(figsize=(12, 12))

ax = sns.heatmap(con_matrix, cmap=plt.cm.Greens, annot=True, square=True, xticklabels=target_names, yticklabels=target_names)

ax.set_ylabel('Actual', fontsize=20)

ax.set_xlabel('Predicted', fontsize=20)
report=classification_report(test_image_gen.classes,y_pred,target_names=target_names,output_dict=True)

pd.DataFrame(report).transpose()
# model.save('to_deploy.h5')
losses = pd.DataFrame(model.history.history)

losses.plot()
from tensorflow.keras.preprocessing import image



plt.imshow(imread(sample_wafer[6]))
eval_image = image.load_img(sample_wafer[6],target_size=img_shape,color_mode='rgba')

eval_image = image.img_to_array(eval_image)

eval_image = np.expand_dims(eval_image, axis=0)

l=model.predict(eval_image)

keys=list(test_image_gen.class_indices.keys())

print('wafer defect classifed as '+ str(keys[l.argmax()]))