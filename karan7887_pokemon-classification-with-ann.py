# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array,load_img
        
from pathlib import Path
p_train = Path('../input/pokemonclassification/Pokemon-dataset/train/')
p_test = Path('../input/pokemonclassification/Pokemon-dataset/test/')
p_val = Path('../input/pokemonclassification/Pokemon-dataset/valid/')
train_x = []
train_y = []
for di in p_train.glob('*'):
    label = str(di).split("\\")[-1]
    
    for image_path in di.glob('*.jpg'):
        
        img_l = image.load_img(image_path,target_size=(240,240))
        img_a = image.img_to_array(img_l)
        train_x.append(img_a)
        train_y.append(label)
       
    for image_path in di.glob('*.png'):
        img_l2 = image.load_img(image_path,target_size=(240,240))
        img_a2 = image.img_to_array(img_l2)
        train_x.append(img_a2)
        train_y.append(label.split('/')[-1])
    for image_path in di.glob('*.jpeg'):
        img_l2 = image.load_img(image_path,target_size=(240,240))
        img_a2 = image.img_to_array(img_l2)
        train_x.append(img_a2)
        train_y.append(label.split('/')[-1])
        
test_x = []
test_y = []
for di in p_test.glob('*'):
    label = str(di).split("\\")[-1]
    for image_path in di.glob('*.jpg'):
        
        img_l = image.load_img(image_path,target_size=(240,240))
        img_a = image.img_to_array(img_l)
        test_x.append(img_a)
        test_y.append(label.split('/')[-1])
       
    for image_path in di.glob('*.png'):
        img_l2 = image.load_img(image_path,target_size=(240,240))
        img_a2 = image.img_to_array(img_l2)
        test_x.append(img_a2)
        test_y.append(label.split('/')[-1])
    for image_path in di.glob('*.jpeg'):
        img_l2 = image.load_img(image_path,target_size=(240,240))
        img_a2 = image.img_to_array(img_l2)
        test_x.append(img_a2)
        test_y.append(label.split('/')[-1])
        
val_x = []
val_y = []
for di in p_val.glob('*'):
    label = str(di).split("\\")[-1]
    for image_path in di.glob('*.jpg'):
        
        img_l = image.load_img(image_path,target_size=(240,240))
        img_a = image.img_to_array(img_l)
        val_x.append(img_a)
        val_y.append(label.split('/')[-1])
       
    for image_path in di.glob('*.png'):
        img_l2 = image.load_img(image_path,target_size=(240,240))
        img_a2 = image.img_to_array(img_l2)
        val_x.append(img_a2)
        val_y.append(label.split('/')[-1])
    for image_path in di.glob('*.jpeg'):
        img_l2 = image.load_img(image_path,target_size=(240,240))
        img_a2 = image.img_to_array(img_l2)
        val_x.append(img_a2)
        val_y.append(label.split('/')[-1])

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
X_train = train_x
Y_train = train_y
X_test = test_x
Y_test = test_y
X_val = val_x
Y_val = val_y



X_train = np.array(X_train)
X_test = np.array(X_test)
X_val = np.array(X_val)

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
Y_val = np.array(Y_val)
print(X_train.shape,X_test.shape,X_val.shape)
print(Y_train.shape,Y_test.shape,Y_val.shape)
X_train = X_train.reshape(303,-1)
X_test = X_test.reshape(123,-1)
X_val = X_val.reshape(66,-1)
def fun(labels):
    labels_arr = []
    for temp in labels:
        labels_arr.append(temp.split('/')[-1])
    labels_arr = np.array(labels_arr)
    return labels_arr
Y_train = fun(Y_train)
Y_test = fun(Y_test)
Y_val = fun(Y_val)
print(np.unique(Y_test,return_counts=True))
from sklearn.preprocessing import OneHotEncoder
hot  = OneHotEncoder()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


Y_train = Y_train.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)
Y_val = Y_val.reshape(-1,1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
X_val = scaler.fit_transform(X_val)

Y_train = hot.fit_transform(Y_train)
Y_test = hot.fit_transform(Y_test)
Y_val = hot.fit_transform(Y_val)
print(X_train.shape,X_test.shape,X_val.shape)
print(Y_train.shape,Y_test.shape,Y_val.shape)
from sklearn.utils import shuffle

X_train,Y_train = shuffle(X_train,Y_train,random_state=101)
X_test,Y_test = shuffle(X_test,Y_test,random_state=101)
X_val,Y_val = shuffle(X_val,Y_val,random_state=101)
print(X_train.shape,X_test.shape,X_val.shape)
print(Y_train.shape,Y_test.shape,Y_val.shape)
Y_train = Y_train.toarray()
Y_test = Y_test.toarray()
Y_val = Y_val.toarray()
Y_train[0:10]
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
# Ann Model Create


model = Sequential()
model.add(Dense(150,input_shape=(172800,),activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
hist = model.fit(X_train,Y_train,epochs = 14,validation_data=(X_val,Y_val),shuffle=True,batch_size=16)



# plt.plot(hist.history['accuracy'])

# plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])
plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])
pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
actual = []
for num in Y_test:
    actual.append(num.argmax())
prediction = []
for num in pred:
    prediction.append(num.argmax())
print(accuracy_score(actual,prediction))
