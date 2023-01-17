# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="whitegrid")

import keras
import cv2

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(_)
#     for filename in filenames:
#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_path = "/kaggle/input/intel-image-classification/seg_train/seg_train"
pred_path = "/kaggle/input/intel-image-classification/seg_pred/seg_pred"
test_path = "/kaggle/input/intel-image-classification/seg_test/seg_test"


code = {'buildings':0 ,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}

def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x
def get_x_y(path):
    y = []
    x = []
    for folders,arr, files in os.walk(path):
        for a in arr:
            f=os.path.join(path, a)
    #         print(a, len(os.listdir(f)))
            for i in os.listdir(f):
                link = os.path.join(path,a ,i)
                img = cv2.imread(link)
                img_array = cv2.resize(img, (128,128))
                x.append(img_array)
                y.append(code[a]) 
    return x,y           
x_train, y_train= get_x_y(train_path)
def draw_sample(x,y):
    rand_list = np.random.randint(0, len(x), 36)

    plt.figure(figsize=(20,20))
    for index, value in enumerate(rand_list):
        plt.subplot(6,6,index+1)
        plt.imshow(x[value])
        plt.axis("off")
        plt.title(getcode(y[value]))

draw_sample(x_train, y_train)
    
x_test, y_test = get_x_y(test_path)
draw_sample(x_test,y_test)
x_pred=[]
for folders,arr,files in os.walk(pred_path):
    for file in files:
        img = cv2.imread(os.path.join(pred_path, file))
        img = cv2.resize(img,(128,128))
        x_pred.append(img)
        
x_pred[1].shape
x_train_std = np.array(x_train)/255
y_train = np.array(y_train)
x_test_std = np.array(x_test)/255
y_test = np.array(y_test)
x_pred_std = np.array(x_pred)/255

print(f"x_train shape is{x_train_std.shape}")
print(f"y_train shape is{y_train.shape}")
print(f"x_test shape is{x_test_std.shape}")
print(f"y_test shape is{y_test.shape}")
print(f"x_pred shape is{x_pred_std.shape}")
print(len(np.unique(y_train)))

print(x_train_std[1].max())
model = keras.models.Sequential([
        keras.layers.Conv2D(32,kernel_size=(3,3),input_shape=(128,128,3)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"), 
    
        keras.layers.Conv2D(64,kernel_size=(3,3)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
    
        keras.layers.MaxPool2D(5,5),
    
        keras.layers.Conv2D(128,kernel_size=(3,3)),
        keras.layers.Activation("relu"),
        keras.layers.BatchNormalization(),
    
        keras.layers.Conv2D(200,kernel_size=(3,3)),
        keras.layers.Activation("relu"),
        keras.layers.BatchNormalization(),
    
        keras.layers.MaxPool2D(5,5),
    
        keras.layers.Flatten() ,    
    
        keras.layers.Dense(128) ,
        keras.layers.Dropout(rate=0.2) ,
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
    
        keras.layers.Dense(64) ,
        keras.layers.Dropout(rate=0.2) , 
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
    
        keras.layers.Dense(32) ,        
        keras.layers.Dropout(rate=0.2) ,
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
    
        keras.layers.Dense(6,activation='softmax') ,    
        ])
adam = keras.optimizers.Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999)

model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()
epochs = 30
model.fit(x_train_std, y_train, epochs= epochs, validation_data=[x_test_std,y_test], batch_size=128)
model.evaluate(x_test_std, y_test)
model.evaluate(x_train_std, y_train)
epo_range = np.arange(1,epochs+1,1)
val_loss = model.history.history['val_loss']
val_acc=model.history.history["val_accuracy"]
loss=model.history.history["loss"]
acc= model.history.history["accuracy"]


plt.figure(figsize=(10,7))
sns.lineplot(epo_range, acc, color="b", label="accuracy")
sns.lineplot(epo_range, val_acc, color="r", label="val_accuracy")
plt.legend()
plt.show()
pred= model.predict_classes(x_pred_std)
pred_prob = model.predict_proba(x_pred_std)

draw_sample(x_pred, pred)
for i in range(1,6):
    
    fig, ax = plt.subplots(1,4, figsize=(19,4))
    
    r = np.random.randint(1,len(pred_prob))
    ax[0].imshow(x_pred[r])
    ax[0].set_title(getcode(pred[r]),size=14)
    ax[0].axis("off")
    ax[1].bar(code.keys(),pred_prob[r])
    ax[1].tick_params(axis='x', rotation=70, labelsize=14)
    
    r = np.random.randint(1,len(pred_prob))
    ax[2].imshow(x_pred[r])
    ax[2].set_title(getcode(pred[r]), size=14)
    ax[2].axis("off")
    ax[3].bar(code.keys(),pred_prob[r])
    ax[3].tick_params(axis='x', rotation=70, labelsize=14)
    plt.show()
        
