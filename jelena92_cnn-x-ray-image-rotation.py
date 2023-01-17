import os

import numpy as np 

import pandas as pd 

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

from PIL import Image



from keras.models import Sequential

from keras_preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization,Conv2D, MaxPooling2D

from keras import regularizers, optimizers
file = os.listdir("../input/train/train")

columns=["normal", "right", "down", "left"]

normal=[]

right=[]

down=[]

left=[]

for filename in file:

    category = filename.split('_')[0]

    for i in range(len(columns)):

        if category == columns[i]: #if it is "down" do this

            zero=np.zeros(4) #vector zero=[0,0,0,0]

            zero[i]=1  #put 1 on third spot zero=[0,0,1,0]

            normal.append(zero[0])

            right.append(zero[1])

            down.append(zero[2])

            left.append(zero[3])



df = pd.DataFrame({

    "filename": file,

    "normal": normal,

    "right":right,

    "down":down,

    "left":left,

})
df.head()
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
datagen=ImageDataGenerator(rescale=1./255.)

train_generator=datagen.flow_from_dataframe(

    train_df,

    directory="../input/train/train",

    x_col="filename",

    y_col=columns,

    batch_size=32,

    seed=42,

    shuffle=True,

    class_mode="other",

    target_size=(128,128))



test_datagen=ImageDataGenerator(rescale=1./255.)

valid_generator=test_datagen.flow_from_dataframe(

    validate_df,

    directory="../input/train/train",

    x_col="filename",

    y_col=columns,

    batch_size=32,

    seed=42,

    shuffle=True,

    class_mode="other",

    target_size=(128,128))





test_filenames = os.listdir("../input/test/test")

test_df = pd.DataFrame({

    'filename': test_filenames

})

nb_samples = test_df.shape[0]



test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    directory="../input/test/test", 

    x_col='filename',

    y_col=None,

    batch_size=1,

    seed=42,

    shuffle=False,

    class_mode=None,

    target_size=(128,128))
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape=(128,128,3)))

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(4, activation='sigmoid'))

model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])

model.summary()
from keras.utils.vis_utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

plt.show()
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

history=model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=20

)
loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,color='red',label='Training loss')

plt.plot(epochs,val_loss,color='green',label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()



acc = history.history['acc']

val_acc = history.history['val_acc']

plt.plot(epochs,acc,color='red',label='Training acc')

plt.plot(epochs,val_acc,color='green',label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
test_generator.reset()

pred=model.predict_generator(test_generator,

    steps=STEP_SIZE_TEST,

    verbose=1)
pred
#column "rotation_prediction" will take max probability to belong to some class 

rotation_prediction=[]

for i in pred:

    maximum=list(i).index(max(i))

    rotation_prediction.append(maximum)



#columns "normal","right","down","left" choose class if probability of belonging to class is greater than 0.5 

# pred_bool = (pred >0.5)

# predictions = pred_bool.astype(int)

# results=pd.DataFrame(predictions, columns=columns)

# results["filename"]=test_generator.filenames

# ordered_cols=["filename"]+columns

# results=results[ordered_cols] #To get the same column order



results=pd.DataFrame()

results["filename"]=test_generator.filenames

results["rotation_prediction"]=rotation_prediction
results.to_csv("results.csv",header=True,index=False)

results.head()
test_categories = []

for filename in results["filename"]:

    category = filename.split('_')[0]

    if category == 'normal':

        test_categories.append(0)

    elif category == 'right':

        test_categories.append(1)

    elif category == 'down':

        test_categories.append(2)

    else:

        test_categories.append(3)

        

print("Acc:",accuracy_score(test_categories, results["rotation_prediction"]))
cm=confusion_matrix(test_categories, results["rotation_prediction"])

sns.heatmap(cm, annot=True, fmt="d")

#rotate images

not_rotated_image=[] #original images

rotated_image=[] #rotated images

path = '../input/test/test'

for i in range(results.shape[0]):

    img_name=results["filename"][i]

    img=Image.open(os.path.join(path,img_name))

    not_rotated_image.append(img)

    rotation=results["rotation_prediction"][i]  #how many times to rotate

    img=img.rotate(rotation*90) #obrnuto od kazaljke na satu

    rotated_image.append(img)

    #name=img_name[len(img_name.split('_')[0])+1:] # not to be right_232_HAND_PA_RIGHT_145.jpg but just 232_HAND_PA_RIGHT_145.jpg
#show first 15 images 

for i in range(15):

    print("Name of image: ",results["filename"][i])

    plt.figure()

    plt.title("Not rotated")

    plt.imshow(not_rotated_image[i],cmap='gray', vmin=0, vmax=255) 

    plt.show()

    plt.title("Rotated")

    plt.imshow(rotated_image[i],cmap='gray', vmin=0, vmax=255) 

    plt.show()
index=[]

for i in range(len(test_categories)):

    if test_categories[i]!=list(results["rotation_prediction"])[i]:

        index.append(i)

badlyRotated=results.iloc[index]

badlyRotated.head()
for i in badlyRotated.index:

    plt.figure()

    plt.imshow(not_rotated_image[i],cmap='gray', vmin=0, vmax=255) 

    plt.show()