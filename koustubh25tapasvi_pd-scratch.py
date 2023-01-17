# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from pydicom import dcmread
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics


Project_path ='../input/rsna-pneumonia-detection-challenge/'

Train_Image_path = Project_path + 'stage_2_train_images/'

Train_Lables= pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')

Train_Lables.head()

patient_id = Train_Lables['patientId'].values.tolist()
Target_Label = Train_Lables['Target'].values.tolist()
len(Target_Label)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation

def vgg_face():	
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model
patients=len(patient_id)

Train_Data=[]
Val_Data=[]

Train_Lables=[]
Val_Labels=[]


for i in range(patients):
    Image_name = Train_Image_path + patient_id[i] +'.dcm'
    Image=dcmread(Image_name)
    rs = cv2.resize(Image.pixel_array,(224,224))
    print(i)
    if i<24182:
        Train_Data.append(rs)
        Train_Lables.append(Target_Label[i])
        
    else:
        Val_Data.append(rs)
        Val_Labels.append(Target_Label[i])
        
print(len(Train_Lables))
print(len(Val_Labels))
import numpy as np
def Batch_Generator_Train(Batch_no):
    Batch_Size= 226
    batch_images_train = np.zeros((Batch_Size, 224, 224, 3), dtype=np.float32)

    Batch_no = Batch_no
    batch_index_initializer= Batch_no * 226

    for i in range(0,225):
        batch_images_train[i][:,:,0]= preprocess_input(np.array(Train_Data[i+batch_index_initializer], dtype=np.float32))
        batch_images_train[i][:,:,1]= preprocess_input(np.array(Train_Data[i+batch_index_initializer], dtype=np.float32))
        batch_images_train[i][:,:,2]= preprocess_input(np.array(Train_Data[i+batch_index_initializer], dtype=np.float32))
    return batch_images_train

def Val_Data_Generator(Batch_no):
    Batch_Size=195
    batch_images_val = np.zeros((Batch_Size, 224, 224, 3), dtype=np.float32)
    Batch_no = Batch_no
    batch_index_initializer= Batch_no * 195
    for i in range(0,194):
        batch_images_val[i][:,:,0]= preprocess_input(np.array(Val_Data[i+batch_index_initializer], dtype=np.float32))
        batch_images_val[i][:,:,1]= preprocess_input(np.array(Val_Data[i+batch_index_initializer], dtype=np.float32))
        batch_images_val[i][:,:,2]= preprocess_input(np.array(Val_Data[i+batch_index_initializer], dtype=np.float32))
        
    return batch_images_val
def Label_Generator_Train(Batch_no):
    Batch_size=226
    index_initializer = Batch_no * Batch_size
    Start_index = index_initializer - Batch_size
    End_index = Start_index + Batch_size
    Batch_Label_train=Train_Lables[Start_index:End_index]
    return Batch_Label_train
    
def Label_Generator_Val(Batch_no):
    Batch_size=195
    index_initializer = Batch_no * Batch_size
    Start_index = index_initializer - Batch_size
    End_index = Start_index + Batch_size
    Batch_Label_val=Val_Labels[Start_index:End_index]
    return Batch_Label_val
    
model = vgg_face()
from tensorflow.keras.models import Model
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
def Generate_Embeddings_Train(batch_images_train):
    embeddings=[]
    for i in range(0,226):
        embedding_vector = vgg_face_descriptor.predict(np.expand_dims(batch_images_train[i], axis=0))[0]
        embeddings.append(embedding_vector)
    return embeddings
def Generate_Embeddings_Val(batch_images_val):
    embeddings=[]
    for i in range(0,195):
        embedding_vector = vgg_face_descriptor.predict(np.expand_dims(batch_images_val[i], axis=0))[0]
        embeddings.append(embedding_vector)
    return embeddings
# Training 
classifier= SVC(kernel='rbf')
for i in range(1,10):
    print("Training Batch ->",i)
    X=Batch_Generator_Train(i)
    
    X_train=Generate_Embeddings_Train(X)
    
    X_train=np.array(X_train)
    
    y_train=Label_Generator_Train(i)
    
    #SVM
    classifier.fit(X_train,y_train)
    
X=Val_Data_Generator(2)
    
X_val=Generate_Embeddings_Val(X)
    
X_val=np.array(X_val)
    
y_val=Label_Generator_Val(2)
    
y_pred_train=classifier.predict(X_train)
y_pred_val=classifier.predict(X_val)

print("Train_Accuracy=",metrics.accuracy_score(y_train,y_pred_train))
print("Val_Accuracy=",metrics.accuracy_score(y_val,y_pred_val))

