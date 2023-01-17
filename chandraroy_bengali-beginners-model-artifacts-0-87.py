import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

import matplotlib.pyplot as plt

import gc

from PIL import Image



import dask.dataframe as dd

from tqdm.auto import tqdm

import cv2

print(os.listdir("../input/bengaliai-cv19/"))
df_train_labels = pd.read_csv("../input/bengaliai-cv19/train.csv")

df_train_labels.head()
len(df_train_labels.grapheme_root.unique())
len(df_train_labels.consonant_diacritic.unique())
len(df_train_labels.vowel_diacritic.unique())
df_train_labels.shape
df_train_labels = df_train_labels.drop(['grapheme'], axis=1)
df_train_labels.head()
#df_train_labels.image_id.stack()

df_tmp = pd.melt(df_train_labels, id_vars=['image_id'], value_vars=['grapheme_root',	'vowel_diacritic',	'consonant_diacritic'])
df_tmp.head()
df_tmp[df_tmp['image_id']=='Train_0']
df_tmp['row_id'] = df_tmp['image_id']+'_'+df_tmp['variable']
df_tmp.head()
df_tmp= df_tmp.rename(columns={"variable": "component"}, errors="raise")
df_test_labels = pd.read_csv("../input/bengaliai-cv19/test.csv")

df_test_labels.head()
df_consonant = df_tmp[df_tmp['component'] =='consonant_diacritic']

df_grapheme = df_tmp[df_tmp['component'] =='grapheme_root']

df_vowel = df_tmp[df_tmp['component'] =='vowel_diacritic']
print(df_consonant.shape)

print(df_grapheme.shape)

print(df_vowel.shape)
df_consonant.head()
sns.catplot(x="vowel_diacritic", data=df_train_labels, kind="count")
sns.catplot(x="consonant_diacritic", data=df_train_labels, kind="count")
sns.catplot(x="grapheme_root", data=df_train_labels, kind="count")
HEIGHT = 137

WIDTH = 236



def load_as_npa(file):

    df = pd.read_parquet(file)

    return df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)
#images0 = load_as_npa('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')

#images1 = load_as_npa('/kaggle/input/bengaliai-cv19/train_image_data_1.parquet')

#images2 = load_as_npa('/kaggle/input/bengaliai-cv19/train_image_data_2.parquet')

#images3 = load_as_npa('/kaggle/input/bengaliai-cv19/train_image_data_3.parquet')

#f, ax = plt.subplots(4, 4, figsize=(12, 8))

#ax = ax.flatten()



#for i in range(16):

#    ax[i].imshow(images0[i], cmap='Greys')
#final_train_images = np.concatenate((images0, images1, images2, images3), axis=0)
#del [[images0, images1, images2, images3, final_train_images]]

#del [[final_train_images]]

#gc.collect()
#final_train_images.shape
#import pyarrow.parquet as pq
#able = pq.read_table(file_path, nthreads=4)

#df_image_0 = pq.read_table('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')
def resize(df, size=46, need_progress_bar=True):

    resized = {}

    if need_progress_bar:

        for i in tqdm(range(df.shape[0])):

            image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

            resized[df.index[i]] = image.reshape(-1)

    else:

        for i in range(df.shape[0]):

            image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

            resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T

    return resized
df_image_0 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')
df_image_0.shape
type(df_image_0)
df_image_0.head()
df_image_0 = df_image_0.iloc[:,1:]
df_image_0 = resize(df_image_0)/255

#X_train = resize(X_train)/255
X_image_0 = df_image_0.to_numpy() # Convert the dataframe to matrix 
X_image_0.shape
del [[df_image_0]]

gc.collect()
df_image_1 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_1.parquet')

df_image_1= df_image_1.iloc[:,1:]
df_image_1 = resize(df_image_1)/255
X_image_1 = df_image_1.to_numpy() # Convert the dataframe to matrix 
X_image_1.shape
#del [[df_image_1]]

del df_image_1

gc.collect()
df_image_2 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_2.parquet')

df_image_2= df_image_2.iloc[:,1:]
df_image_2 = resize(df_image_2)/255
X_image_2 = df_image_2.to_numpy() # Convert the dataframe to matrix 
X_image_2.shape
#del [[df_image_2]]

del df_image_2

gc.collect()
df_image_3 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_3.parquet')

df_image_3= df_image_3.iloc[:,1:]
df_image_3 = resize(df_image_3)/255
X_image_3 = df_image_3.to_numpy() # Convert the dataframe to matrix 
X_image_3.shape
del [[df_image_3]]

gc.collect()
#final_train_images
#image_size = 137 * 236

#final_train_images.reshape(image_size)
#X = final_train_images/255
#X = pd.merge([X_image_0, X_image_1, X_image_2, X_image_3])

X= np.concatenate((X_image_0, X_image_1, X_image_2, X_image_3), axis=0)
len(X)
del X_image_0

del X_image_1

del X_image_2

del X_image_3

gc.collect()
type(X)
#from tempfile import TemporaryFile

#train_all_image_file = TemporaryFile()





#from joblib import dump

#dump(X, 'all_image_4_train.joblib', compress=3)

#import pickle

#f=open('all_image_4_train','w')

#pickle.dump(X, f, protocol=4)

#f.close()
# Download form the output folder of thsi kernel

df_consonant.to_csv("target_4_consonant.csv")

df_grapheme.to_csv("target_4_grapheme.csv")

df_vowel.to_csv("target_4_vowel.csv")
#X.shape
X= X.reshape(-1,46, 46,1)
n_classes = 7
y= df_consonant.value
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, n_classes)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
#X = np.divide(X, 255)

#import dask.array as da



#X = np.arange(1000)  #arange is used to create array on values from 0 to 1000

#y = da.from_array(X, chunks=(100))  #converting numpy array to dask array



#y.div(255).compute()  #computing mean of the array
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization

from keras import Sequential
model_consonant = Sequential()



model_consonant .add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(46, 46,1)))



model_consonant .add(Conv2D(32, kernel_size=(3,3), activation='relu'))

model_consonant .add(MaxPooling2D(pool_size=(2,2)))

model_consonant .add(Dropout(0.25))



model_consonant .add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model_consonant .add(MaxPooling2D(pool_size=(2,2)))

model_consonant .add(Dropout(0.25))



model_consonant .add(Flatten())



model_consonant .add(Dense(128, activation='relu'))

#model.add(Dense(128, activation='relu'))

model_consonant.add(Dropout(0.5))



model_consonant .add(Dense(n_classes, activation='softmax'))
model_consonant.summary()
model_consonant.compile(loss='categorical_crossentropy',

             optimizer='nadam',

             metrics=['accuracy'])
model_consonant.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))
#history = model.fit(X_train, 

#                    y_train, 

#                    batch_size=128, 

#                    epochs=100,

#                    verbose=1,

#                    validation_data=(X_test, y_test)

#                   )

#history = model_consonant.fit(X, 

#                    y, 

#                    batch_size=128, 

#                    epochs=1,

#                    verbose=1

#                    #validation_data=(X_test, y_test)

#                   )
# Save the weights

model_consonant.save_weights('model_consonant_weight.h5')



# Save the model architecture

with open('model_consonant_architecture.json', 'w') as f:

    f.write(model_consonant.to_json())

    

## READ weight and architecture

#from keras.models import model_from_json



## Model reconstruction from JSON file

#with open('model_consonant_architecture.json', 'r') as f:

#    model = model_from_json(f.read())



## Load weights into the new model

#model.load_weights('model_consonant_weight.h5')
predictions_consonant = model_consonant.predict(X_test)

predictions_consonant = np.argmax(predictions_consonant, axis=1) 
# calculate accuracy

#from sklearn import metrics

#print(metrics.accuracy_score(y_test, predictions_consonant))

#print(metrics.confusion_matrix(y_test, predictions_consonant))
del X_train

del X_test

del y_train

del y_test

gc.collect()
n_classes = 168

y= df_grapheme.value

y = to_categorical(y, n_classes)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
model_grapheme = Sequential()



model_grapheme.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(46, 46,1)))



model_grapheme.add(Conv2D(32, kernel_size=(3,3), activation='relu'))

model_grapheme.add(MaxPooling2D(pool_size=(2,2)))

model_grapheme.add(Dropout(0.25))



model_grapheme.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model_grapheme.add(MaxPooling2D(pool_size=(2,2)))

model_grapheme.add(Dropout(0.25))



model_grapheme.add(Flatten())



model_grapheme.add(Dense(128, activation='relu'))

#model.add(Dense(128, activation='relu'))

model_grapheme.add(Dropout(0.5))



model_grapheme.add(Dense(n_classes, activation='softmax'))
model_grapheme.summary()
model_grapheme.compile(loss='categorical_crossentropy',

             optimizer='nadam',

             metrics=['accuracy'])
model_grapheme.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))
predictions_grapheme = model_grapheme.predict(X_test)
# Save the weights

model_grapheme.save_weights('model_grapheme_weight.h5')



# Save the model architecture

with open('model_grapheme_architecture.json', 'w') as f:

    f.write(model_grapheme.to_json())

    

## READ weight and architecture

#from keras.models import model_from_json



## Model reconstruction from JSON file

#with open('model_consonant_architecture.json', 'r') as f:

#    model = model_from_json(f.read())



## Load weights into the new model

#model.load_weights('model_consonant_weight.h5')
del X_train

del X_test

del y_train

del y_test

gc.collect()
#n_classes = 11

#y= df_vowel.value

#y = to_categorical(y, n_classes)
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
#model_vowel = Sequential()



#model_vowel.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(46, 46,1)))



#model_vowel.add(Conv2D(32, kernel_size=(3,3), activation='relu'))

#model_vowel.add(MaxPooling2D(pool_size=(2,2)))

#model_vowel.add(Dropout(0.25))



#model_vowel.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

#model_vowel.add(MaxPooling2D(pool_size=(2,2)))

#model_vowel.add(Dropout(0.25))



#model_vowel.add(Flatten())



#model_vowel.add(Dense(128, activation='relu'))

##model.add(Dense(128, activation='relu'))

#model_vowel.add(Dropout(0.5))



#model_vowel.add(Dense(n_classes, activation='softmax'))
#model_vowel.summary()
#model_vowel.compile(loss='categorical_crossentropy',

#             optimizer='nadam',

#             metrics=['accuracy'])
#model_vowel.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))
#predictions_vowel = model_vowel.predict(X_test)
## Save the weights

#model_vowel.save_weights('model_vowel_weight.h5')



## Save the model architecture

#with open('model_vowel_architecture.json', 'w') as f:

#    f.write(model_vowel.to_json())

    

## READ weight and architecture

#from keras.models import model_from_json



## Model reconstruction from JSON file

#with open('model_consonant_architecture.json', 'r') as f:

#    model = model_from_json(f.read())



## Load weights into the new model

#model.load_weights('model_consonant_weight.h5')
#del X_train

#del X_test

#del y_train

#del y_test

#gc.collect()
#components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

#target=[] # model predictions placeholder

#row_id=[] # row_id place holder

#n_cls = [7,168,11] # number of classes in each of the 3 targets

#IMG_SIZE = 46

#IMG_SIZE= 46

#N_CHANNELS = 1

#for i in range(4):

#    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

#    df_test_img.set_index('image_id', inplace=True)



#    X_test = resize(df_test_img)/255

#    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)



#    for pred in preds_dict:

#        preds_dict[pred]=np.argmax(model_dict[pred].predict(X_test), axis=1)



#    for k,id in enumerate(df_test_img.index.values):  

#        for i,comp in enumerate(components):

#            id_sample=id+'_'+comp

#            row_id.append(id_sample)

#            target.append(preds_dict[comp][k])



#df_sample = pd.DataFrame(

#    {'row_id': row_id,

#    'target':target

#    },

#    columns =['row_id','target'] 

#)

#df_sample.to_csv('submission.csv',index=False)