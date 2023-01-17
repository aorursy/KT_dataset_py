import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from ipykernel import kernelapp as app
import os


# Deep Learning libraries 
import tensorflow as tf
import PIL as  pil 
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
# from keras.preprocessing.image import ImageDataGenerator, load_img
# from keras_preprocessing.image import ImageDataGenerator, load_img
!wget 'https://raw.githubusercontent.com/keras-team/keras-preprocessing/d3d58f5c6e2ef8b6270301415738ecb6deee2042/keras_preprocessing/image.py'
from image import ImageDataGenerator

import os
# print(os.listdir("../input"))
print(os.listdir("/kaggle/input"))


# Any results you write to the current directory are saved as output.
keras.__version__
names = pd.read_csv("/kaggle/input/names.csv")
names.sample(5)
# names.describe()
anno_train = pd.read_csv("/kaggle/input/anno_train.csv")
anno_train.sample(5)
# anno_train.describe()
# Folder containng all the files for the test set. 
car_test= "/kaggle/input/car_data/car_data/test/"

# Folder containing all the training data.
car_train = "/kaggle/input/car_data/car_data/train/"
print('There are', len(os.listdir(car_train)),'folders in the training dataset')
# os.listdir(car_train)
car_test = "/kaggle/input/car_data/car_data/test/"
os.listdir(car_test)
# I am setting up a function here that takes the training and test set data and transforms it into a pandas dataframe

def pd_images(folder, is_training = True):
    data = list()
    for labels in os.listdir(car_train):
        for label in os.listdir(car_train+labels):
            if is_training == True:
                car_add = car_train + labels + '/' + label
            else:
                car_add = car_test+ labels + '/' + label
            car_value= (labels, car_add)                
            if car_value not in data:
                data.append(car_value)   

    pd_images = pd.DataFrame(np.array(data).reshape(8144,2), columns= ["car", "image path"])
    
    return pd_images


train_df = pd_images(car_train, is_training = True)
train_df.sample(10)
# t_i = train_df.loc[3300]['image path']

# a = plt.imread(t_i)
# plt.imshow(a)
train_df.describe()
test_df = pd_images(car_test, is_training=False)
test_df.sample(10)
t_b = test_df.loc[3300]['image path']
print(t_b)

os.listdir('/kaggle/input/car_data/car_data/test/Chevrolet Traverse SUV 2012/')
# b = plt.imread(t_b)
# plt.imshow(b)

def cars_to_label(df):
#     good_train = train_df[train_df['car'].str.contains('Audi|BMW|Mercedes')]
    df = df[df['car'].str.contains('Audi|BMW|Mercedes')]
    df = df[df['car'].notnull()].copy()
#     df = df['car'].str.split(' ').str[0]
    df['car'] = df['car'].str.split(' ').str[0]
    df['car label'] = df.car.astype("category").cat.codes

    return df


train_img = cars_to_label(train_df)
# train_img.sample(5)
# print(train_img.iloc[4,:]['image path'])
# train_img.isnull().values.any()
# print('The type of this column is: ', type(train_img.iloc[4,:]['image path']))
train_img_df =  train_img[['image path', 'car label']].copy()
train_img_df.sample(5)

test_img = cars_to_label(test_df)
# print(test_img['image path'].head().str[1][1])
# print(test_img.iloc[4,:]['image path'])
test_img.sample(10)
# let's build the CNN model

model = Sequential()

#Convolution
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))

#Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))

# 2nd Convolution
model.add(Conv2D(32, (3, 3), activation="relu"))

# 2nd Pooling layer
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

#3rd Convolution
model.add(Conv2D(32, (3, 3), activation="relu"))

#Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

# Flatten the layer
model.add(Flatten())

# Fully Connected Layers
model.add(Dense(activation = 'relu', units = 128))
model.add(Dense(activation = 'sigmoid', units = 3))

# Compile the Neural network
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

datagen = ImageDataGenerator(rescale=1./255.,validation_split=0.25)
# print(directory)
train_generator=datagen.flow_from_dataframe(
# dataframe=train_img,
dataframe = train_img_df,
# directory="/kaggle/input/car_data/car_data/train",
directory = None,
x_col="image path",
y_col="car label",
has_ext=True,                                     
subset="training",
batch_size=34,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(32,32))



valid_generator=datagen.flow_from_dataframe(
dataframe=train_img,
# directory="../input,/car_data/car_data/train/",
# directory = train_img_df,
directory = None, 
x_col="image path",
y_col="car label",
has_ext=True,
subset="validation",
batch_size=15,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(32,32))

test_datagen=ImageDataGenerator(rescale=1./255.)


test_generator=test_datagen.flow_from_dataframe(
dataframe=train_img,
# directory="../input/car_data/car_data/test",
directory = None,
x_col="image path",
y_col="car label",
has_ext=True,
batch_size=32,
seed=42,
shuffle=False,
class_mode=None,
target_size=(32,32))
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
# STEP_SIZE_VALID = 23
# STEP_SIZE_TRAIN = 30
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)
model.evaluate_generator(generator=valid_generator
)
test_generator.reset()
pred=model.predict_generator(test_generator,verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
print(predicted_class_indices)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)
os.listdir()
results = pd.read_csv('results.csv')

# results.sample(10)
results['Predictions'].unique
# type(results['Predictions'].loc[1])
# results.head()
results.sample(5)
results.sample(10)

#Total number of mercedes images tested= 261

merc_df = results[results['Filename'].str.contains('Mercedes')]
merc_df.describe()
# merc_df.value_counts
merc_df['Filename'].count()
# merc_df
print('The total number of Mercedes classified correctly are:', (merc_df['Predictions'] ==2).sum()) # 151 
print('The percentage of Mercedes classified correctly is:',((merc_df['Predictions'] ==2).sum())/(merc_df['Filename'].count())*100, 
     '%')
merc_correct = merc_df.loc[merc_df['Predictions']==2, 'Filename']
mc_img = merc_correct.iloc[np.random.randint(1,len(merc_correct))]
img_mc = Image.open(mc_img)

merc_wrong =  merc_df.loc[merc_df['Predictions']!=2, 'Filename']
# merc_wrong 
mw_img = merc_wrong.iloc[np.random.randint(1,len(merc_wrong))]
img_mw = Image.open(mw_img)

f = plt.figure(figsize=(10,10))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(img_mc)
a1.set_title('Correct')

a2 = f.add_subplot(1,2,2)
img_plot = plt.imshow(img_mw)
a2.set_title('Wrong')
bmw_df = results[results['Filename'].str.contains('BMW')]
print(bmw_df['Filename'].count(), 'images were provided for evaluation as BMW or not')
print('The total number of BMW classified correctly are:', (bmw_df['Predictions'] ==1).sum()) # 151 
print('The percentage of BMW classified correctly is:',((bmw_df['Predictions'] ==1).sum())/(bmw_df['Filename'].count())*100, 
     '%')
bmw_correct = bmw_df.loc[bmw_df['Predictions']==1, 'Filename']
bmw_img = bmw_correct.iloc[np.random.randint(1,len(bmw_correct))]
bmw_correct = Image.open(bmw_img)

bmw_wrong =  bmw_df.loc[bmw_df['Predictions']!=1, 'Filename']
# merc_wrong 
not_bmw = bmw_wrong.iloc[np.random.randint(1,len(bmw_wrong))]
not_bmw_img = Image.open(not_bmw)

f = plt.figure(figsize=(10,10))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(bmw_correct)
a1.set_title('Correct')

a2 = f.add_subplot(1,2,2)
img_plot = plt.imshow(not_bmw_img)
a2.set_title('Wrong')
audi = results[results['Filename'].str.contains('Audi')]
print(audi['Filename'].count(), 'images were provided for evaluation as Audi or not')
print('The total number of Audi classified correctly are:', (audi['Predictions'] ==0).sum()) # 151 
print('The percentage of BMW classified correctly is:',((audi['Predictions'] ==0).sum())/(audi['Filename'].count())*100, 
     '%')
audi_correct = audi.loc[audi['Predictions']==0, 'Filename']
audi_img = audi_correct.iloc[np.random.randint(1,len(audi_correct))]
audi_c_img = Image.open(audi_img)

audi_wrong =  audi.loc[audi['Predictions']!=0, 'Filename']
# merc_wrong 
not_audi = audi_wrong.iloc[np.random.randint(1,len(audi_wrong))]
audi_w_img = Image.open(not_audi)

f = plt.figure(figsize=(10,10))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(audi_c_img)
a1.set_title('Correct')

a2 = f.add_subplot(1,2,2)
img_plot = plt.imshow(audi_w_img)
a2.set_title('Wrong')