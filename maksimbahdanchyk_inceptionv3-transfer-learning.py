import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
label = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
train_dir = '/kaggle/input/dog-breed-identification/train/'

test_dir  = '/kaggle/input/dog-breed-identification/test/'
train_img = os.listdir(train_dir)

test_img  = os.listdir(test_dir)



print('Number of train_img',len(train_img))

print('Number of test_img' ,len(test_img))

print(len(set(train_img)))

print(len(set(test_img)))
label['path'] = train_dir+label.id+'.jpg'



fig,axes = plt.subplots(2,5,figsize = (30,10))



for ax in axes.reshape(-1,):

    rnd_idx = np.random.randint(label.index[0],label.index[-1])

    arr = plt.imread(label.loc[rnd_idx,'path'])

    ax.imshow(arr)

    ax.set_title(label.loc[rnd_idx,'breed']+'\n'+str(arr.shape))

    ax.axis('off')



# as we can see pictures have different size.

# lets extract this feature and write to label frame
# there should be more straitforward way



sizes_info = label.path.apply(lambda x: plt.imread(x).shape).values.tolist()

meta_df    = pd.DataFrame(sizes_info,columns = ['height','width','channels'])

label      = pd.merge(label,meta_df,how = 'left',left_index = True,right_index=True)
import seaborn as sns



fig,axes = plt.subplots(1,3,figsize = (20,5))



for ax,col in zip(axes,['height','width','channels']):

    label[[col]].hist(bins = 100,ax = ax)



# most of the pictures have height = 300 and width 500

# make sense to rescale pictures
label['id1'] = label.id+'.jpg'

breed = label.breed.unique()
rnd_row = np.random.randint(label.index[0],label.index[-1],2000)

shorter_df = label.loc[rnd_row,:].reset_index(drop = True)
import tensorflow as tf

datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    rescale          =1./255,

    #rotation_range   = 45,

    #horizontal_flip  = True,

    #shear_range      = 0.2,

    validation_split = 0.25

)



train_generator=datagen.flow_from_dataframe(

    dataframe = label,

    directory="/kaggle/input/dog-breed-identification/train/",

    x_col="id1",

    y_col="breed",

    color_mode = 'rgb',

    subset = 'training',

    batch_size=32,

    seed=42,

    shuffle=False,

    class_mode="categorical",

    target_size=(300,300)

)



val_generator=datagen.flow_from_dataframe(

    dataframe = label,

    directory="/kaggle/input/dog-breed-identification/train/",

    x_col="id1",

    y_col="breed",

    color_mode = 'rgb',

    subset = 'validation',

    batch_size=32,

    seed=42,

    shuffle=False,

    class_mode="categorical",

    target_size=(300,300)

)



classes = train_generator.class_indices
classes = {x:y for x,y in classes.items()}
mnv = tf.keras.applications.inception_v3.InceptionV3(include_top = False,

                                           weights      = 'imagenet',

                                           input_shape = (300,300,3))



mnv.trainbale = False

for layer in mnv.layers:

    layer.trainable = False

    



model = tf.keras.Sequential([

    mnv,

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512,activation ='relu'),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(256,activation ='relu'),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(120,activation = 'softmax')

])
model.compile(loss = 'categorical_crossentropy',

              optimizer = tf.keras.optimizers.Adam(),

              metrics   = ['accuracy'])



hist = model.fit_generator(generator = train_generator,

                           epochs = 3,

                           steps_per_epoch  = train_generator.n//train_generator.batch_size,

                           validation_data  = val_generator,

                           validation_steps = val_generator.n//val_generator.batch_size)

fig, ax = plt.subplots(figsize = (5,5))

ax.plot(range(1,len(hist.history['accuracy'])+1),hist.history['accuracy'])

ax.plot(range(1,len(hist.history['val_accuracy'])+1),hist.history['val_accuracy'])
val_pred = np.argmax(model.predict(val_generator),axis=-1)

classes = {x:y for y,x in classes.items()}



fig,axes = plt.subplots(2,5,figsize = (30,10))



for ax in axes.reshape(-1,):

    rnd_idx = np.random.randint(0,len(val_generator.filepaths))

    arr = plt.imread(val_generator.filepaths[rnd_idx])

    ax.imshow(arr)

    breed_true = classes[val_generator.classes[rnd_idx]]

    breed_pred = classes[val_pred[rnd_idx]]

    

    if breed_true != classes[val_pred[rnd_idx]]:

        ax.set_title(breed_true+'\n'+ breed_pred,color = 'red',fontsize = 15)

    else:

        ax.set_title(breed_true+'\n'+ breed_pred,color = 'black',fontsize = 15)



    ax.axis('off')
from sklearn.metrics import confusion_matrix



confusion_matrix(val_generator.classes,val_pred)