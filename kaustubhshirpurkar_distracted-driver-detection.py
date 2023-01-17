# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import os

import pandas as pd

import pickle

import numpy as np

import seaborn as sns

from sklearn.datasets import load_files

from keras.utils import np_utils

import matplotlib.pyplot as plt

# Pretty display for notebooks

%matplotlib inline

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, Flatten, Dense

from keras.models import Sequential

from keras.utils.vis_utils import plot_model

from keras.callbacks import ModelCheckpoint

from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix

from keras.preprocessing import image                  

from tqdm import tqdm



import seaborn as sns

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

TEST_DIR = os.path.join(os.getcwd(),"/kaggle/input/state-farm-distracted-driver-detection/imgs","test")

TRAIN_DIR = os.path.join(os.getcwd(),"/kaggle/input/state-farm-distracted-driver-detection/imgs","train")

MODEL_PATH = os.path.join(os.getcwd(),"model","self_trained")

PICKLE_DIR = os.path.join(os.getcwd(),"pickle_files")

if not os.path.exists(TEST_DIR):

    print("Testing data does not exists")

if not os.path.exists(TRAIN_DIR):

    print("Training data does not exists")

if not os.path.exists(MODEL_PATH):

    print("Model path does not exists")

    os.makedirs(MODEL_PATH)

    print("Model path created")

if not os.path.exists(PICKLE_DIR):

    os.makedirs(PICKLE_DIR)
def create_csv(DATA_DIR,filename):

    class_names = os.listdir(DATA_DIR)

    data = list()

    if(os.path.isdir(os.path.join(DATA_DIR,class_names[0]))):

        for class_name in class_names:

            file_names = os.listdir(os.path.join(DATA_DIR,class_name))

            for file in file_names:

                data.append({

                    "Filename":os.path.join(DATA_DIR,class_name,file),

                    "ClassName":class_name

                })

    else:

        class_name = "test"

        file_names = os.listdir(DATA_DIR)

        for file in file_names:

            data.append(({

                "FileName":os.path.join(DATA_DIR,file),

                "ClassName":class_name

            }))

    data = pd.DataFrame(data)

    data.to_csv(os.path.join(os.getcwd(),"csv_files",filename),index=False)

CSV_FILES_DIR = os.path.join(os.getcwd(),"csv_files")

if not os.path.exists(CSV_FILES_DIR):

    os.makedirs(CSV_FILES_DIR)

    

create_csv(TRAIN_DIR,"train.csv")

create_csv(TEST_DIR,"test.csv")

data_train = pd.read_csv(os.path.join(os.getcwd(),"csv_files","train.csv"))

data_test = pd.read_csv(os.path.join(os.getcwd(),"csv_files","test.csv"))
data_train.info()
data_train['ClassName'].value_counts()
nf = data_train['ClassName'].value_counts(sort=False)

labels = data_train['ClassName'].value_counts(sort=False).index.tolist()

y = np.array(nf)

width = 1/1.5

N = len(y)

x = range(N)



fig = plt.figure(figsize=(20,15))

ay = fig.add_subplot(211)



plt.xticks(x, labels, size=15)

plt.yticks(size=15)



ay.bar(x, y, width, color="blue")



plt.title('Bar Chart',size=25)

plt.xlabel('classname',size=15)

plt.ylabel('Count',size=15)



plt.show()
data_test.head()


data_test.shape
labels_list = list(set(data_train['ClassName'].values.tolist()))

labels_id = {label_name:id for id,label_name in enumerate(labels_list)}

print(labels_id)

data_train['ClassName'].replace(labels_id,inplace=True)
with open(os.path.join(os.getcwd(),"pickle_files","labels_list.pkl"),"wb") as handle:

    pickle.dump(labels_id,handle)
labels = to_categorical(data_train['ClassName'])

print(labels.shape)
from sklearn.model_selection import train_test_split



xtrain,xtest,ytrain,ytest = train_test_split(data_train.iloc[:,0],labels,test_size = 0.2,random_state=42)
def path_to_tensor(img_path):

    # loads RGB image as PIL.Image.Image type

    img = image.load_img(img_path, target_size=(64, 64))

    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)

    x = image.img_to_array(img)

    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor

    return np.expand_dims(x, axis=0)



def paths_to_tensor(img_paths):

    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]

    return np.vstack(list_of_tensors)


from PIL import ImageFile                            

ImageFile.LOAD_TRUNCATED_IMAGES = True                 



# pre-process the data for Keras

train_tensors = paths_to_tensor(xtrain).astype('float32')/255 - 0.5
valid_tensors = paths_to_tensor(xtest).astype('float32')/255 - 0.5


model = Sequential()

# 64 conv2d filters with relu

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(64,64,3), kernel_initializer='glorot_normal'))

model.add(MaxPooling2D(pool_size=2)) #Maxpool

model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu', kernel_initializer='glorot_normal'))

model.add(MaxPooling2D(pool_size=2)) #Maxpool

model.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu', kernel_initializer='glorot_normal'))

model.add(MaxPooling2D(pool_size=2)) #Maxpool

model.add(Conv2D(filters=512, kernel_size=2, padding='same', activation='relu', kernel_initializer='glorot_normal'))

model.add(MaxPooling2D(pool_size=2)) #Maxpool

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(500, activation='relu', kernel_initializer='glorot_normal'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax', kernel_initializer='glorot_normal'))





model.summary()
plot_model(model,to_file=os.path.join(MODEL_PATH,"model_distracted_driver.png"),show_shapes=True,show_layer_names=True)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



filepath = os.path.join(MODEL_PATH,"distracted-{epoch:02d}-{val_accuracy:.2f}.hdf5")



checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',period=1)

callbacks_list = [checkpoint]

epochs = 20

model_history = model.fit(train_tensors,ytrain,validation_data = (valid_tensors, ytest),epochs=epochs, batch_size=40, shuffle=True,callbacks=callbacks_list)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(model_history.history['loss'], color='b', label="Training loss")

ax1.plot(model_history.history['val_loss'], color='r', label="validation loss")

ax1.set_xticks(np.arange(1, 25, 1))

ax1.set_yticks(np.arange(0, 1, 0.1))



ax2.plot(model_history.history['accuracy'], color='b', label="Training accuracy")

ax2.plot(model_history.history['val_accuracy'], color='r',label="Validation accuracy")

ax2.set_xticks(np.arange(1, 25, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    df_cm = pd.DataFrame(

        confusion_matrix, index=class_names, columns=class_names, 

    )

    fig = plt.figure(figsize=figsize)

    try:

        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    except ValueError:

        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    fig.savefig(os.path.join(MODEL_PATH,"confusion_matrix.png"))

    return fig



def print_heatmap(n_labels, n_predictions, class_names):

    labels = n_labels #sess.run(tf.argmax(n_labels, 1))

    predictions = n_predictions #sess.run(tf.argmax(n_predictions, 1))



#     confusion_matrix = sess.run(tf.contrib.metrics.confusion_matrix(labels, predictions))

    matrix = confusion_matrix(labels.argmax(axis=1),predictions.argmax(axis=1))

    row_sum = np.sum(matrix, axis = 1)

    w, h = matrix.shape



    c_m = np.zeros((w, h))



    for i in range(h):

        c_m[i] = matrix[i] * 100 / row_sum[i]



    c = c_m.astype(dtype = np.uint8)



    

    heatmap = print_confusion_matrix(c, class_names, figsize=(18,10), fontsize=20)



class_names = list()

for name,idx in labels_id.items():

    class_names.append(name)

# print(class_names)

ypred = model.predict(valid_tensors)



print_heatmap(ytest,ypred,class_names)
#Precision Recall F1 Score



ypred_class = np.argmax(ypred,axis=1)

# print(ypred_class[:10])

ytest = np.argmax(ytest,axis=1)



accuracy = accuracy_score(ytest,ypred_class)

print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)

precision = precision_score(ytest, ypred_class,average='weighted')

print('Precision: %f' % precision)

# recall: tp / (tp + fn)

recall = recall_score(ytest,ypred_class,average='weighted')

print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(ytest,ypred_class,average='weighted')

print('F1 score: %f' % f1)
from keras.models import load_model

from keras.utils import np_utils

import shutil

BASE_MODEL_PATH = os.path.join(os.getcwd(),"model")

TEST_DIR = os.path.join(os.getcwd(),"csv_files","test.csv")

PREDICT_DIR = os.path.join(os.getcwd(),"pred_dir")

PICKLE_DIR = os.path.join(os.getcwd(),"pickle_files")

JSON_DIR = os.path.join(os.getcwd(),"json_files")

if not os.path.exists(PREDICT_DIR):

    os.makedirs(PREDICT_DIR)

else:

    shutil.rmtree(PREDICT_DIR)

    os.makedirs(PREDICT_DIR)

if not os.path.exists(JSON_DIR):

    os.makedirs(JSON_DIR)



BEST_MODEL = os.path.join(BASE_MODEL_PATH,"self_trained","distracted-07-0.99.hdf5") #loading checkpoint with best accuracy and min epochs

model = load_model(BEST_MODEL)

model.summary()
data_test = pd.read_csv(os.path.join(TEST_DIR))

#testing on the only 10000 images as loading the all test images requires ram>8gb

data_test = data_test[:10000] 

data_test.info()
with open(os.path.join(PICKLE_DIR,"labels_list.pkl"),"rb") as handle:

    labels_id = pickle.load(handle)

print(labels_id)
ImageFile.LOAD_TRUNCATED_IMAGES = True  

test_tensors = paths_to_tensor(data_test.iloc[:,0]).astype('float32')/255 - 0.5
ypred_test = model.predict(test_tensors,verbose=1)

ypred_class = np.argmax(ypred_test,axis=1)



id_labels = dict()

for class_name,idx in labels_id.items():

    id_labels[idx] = class_name

print(id_labels)



for i in range(data_test.shape[0]):

    data_test.iloc[i,1] = id_labels[ypred_class[i]]
#to create a human readable and understandable class_name

import json

class_name = dict()

class_name["c0"] = "SAFE_DRIVING"

class_name["c1"] = "TEXTING_RIGHT"

class_name["c2"] = "TALKING_PHONE_RIGHT"

class_name["c3"] = "TEXTING_LEFT"

class_name["c4"] = "TALKING_PHONE_LEFT"

class_name["c5"] = "OPERATING_RADIO"

class_name["c6"] = "DRINKING"

class_name["c7"] = "REACHING_BEHIND"

class_name["c8"] = "HAIR_AND_MAKEUP"

class_name["c9"] = "TALKING_TO_PASSENGER"





with open(os.path.join(JSON_DIR,'class_name_map.json'),'w') as secret_input:

    json.dump(class_name,secret_input,indent=4,sort_keys=True)
# creating the prediction results for the image classification and shifting the predicted images to another folder

#with renamed filename having the class name predicted for that image using model

with open(os.path.join(JSON_DIR,'class_name_map.json')) as secret_input:

    info = json.load(secret_input)





for i in range(data_test.shape[0]):

    new_name = data_test.iloc[i,0].split("/")[-1].split(".")[0]+"_"+info[data_test.iloc[i,1]]+".jpg"

    shutil.copy(data_test.iloc[i,0],os.path.join(PREDICT_DIR,new_name))

    

#saving the model predicted results into a csv file

data_test.to_csv(os.path.join(os.getcwd(),"csv_files","short_test_result.csv"),index=False)