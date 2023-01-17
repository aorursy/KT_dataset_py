import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
print(os.listdir("/kaggle/input/nnfl-lab-1"))
import cv2
from sklearn.model_selection import train_test_split
train_images = []       
train_labels = []
shape = (200,200)  
train_path = '/kaggle/input/nnfl-lab-1/training/training/'

for filename in os.listdir('/kaggle/input/nnfl-lab-1/training/training/'):
    if filename.split('.')[1] == 'jpg':
        img = cv2.imread(os.path.join(train_path,filename))
        
        # Spliting file names and storing the labels for image in list
        name=filename.split('_')[0]
        if name=='chair':
            train_labels.append(0)
        elif name=='kitchen':
            train_labels.append(1)
        elif name=='knife':
            train_labels.append(2)
        elif name=='saucepan':
            train_labels.append(3)
        
        # Resize all images to a specific shape
        img = cv2.resize(img,shape)
        
        train_images.append(img)

# Converting labels into One Hot encoded sparse matrix
train_labels = pd.get_dummies(train_labels).values

# Converting train_images to array
train_images = np.array(train_images)

# Splitting Training data into train and validation dataset
x_train,x_val,y_train,y_val = train_test_split(train_images,train_labels,test_size=0.2,random_state = 1)
train_labels.shape
x_train.shape
x_val.shape
test_images = []
test_labels = []
shape = (200,200)
test_path = '/kaggle/input/nnfl-lab-1/testing/testing'

for filename in os.listdir('/kaggle/input/nnfl-lab-1/testing/testing'):
    if filename.split('.')[1] == 'jpg':
        img = cv2.imread(os.path.join(test_path,filename))
        
        # Spliting file names and storing the labels for image in list
        test_labels.append(filename.split('_')[0])
        
        # Resize all images to a specific shape
        img = cv2.resize(img,shape)
        
        test_images.append(img)
        
# Converting test_images to array
test_images = np.array(test_images)
len(test_labels)
print(train_labels[3])
plt.imshow(train_images[3])
FAST_RUN = False
IMAGE_WIDTH=200
IMAGE_HEIGHT=200
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
# Model 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.30))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.40))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax')) 

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()
# Training the model
history = model.fit(x_train,y_train,epochs=60,batch_size=8,validation_data=(x_val,y_val))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
evaluate = model.evaluate(x_val,y_val,batch_size=1)
print(evaluate)
evaluate = model.evaluate(x_train,y_train, batch_size=1)
print(evaluate)
model.save_weights("model.h5")
predict = model.predict(test_images, batch_size=1)
(np.argmax(predict[4]))
outputs=[]
for i in range(len(predict)) :
    temp=[]
    temp.append(test_labels[i])
    temp.append(np.argmax(predict[i]))
    outputs.append(temp)
output=pd.DataFrame(outputs)
output=output.rename(columns={0: "id", 1: "label"})
output
output.to_csv('sub2.csv', index=False)
print(test_labels[0])
print(np.argmax(predict[0]))
plt.imshow(test_images[0])
plt.figure(figsize=(20,100))
for n , i in enumerate(list(np.random.randint(0,len(predict),100))) : 
    plt.subplot(20,5,n+1)
    plt.imshow(test_images[i])    
    plt.axis('off')
    classes = {'chair':0 ,'kitchen':1,'knife':2,'saucepan':3}
    def get_img_class(n):
        for x , y in classes.items():
            if n == y :
                return x
    plt.title(get_img_class(np.argmax(predict[i])))
from IPython.display import HTML 
import pandas as pd 
import numpy as np
import base64 
def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 
    csv = df.to_csv(index=False) 
    b64 = base64.b64encode(csv.encode()) 
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(output)