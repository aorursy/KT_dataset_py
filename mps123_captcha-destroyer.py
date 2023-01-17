# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/captchadata/temp'):
    for filename in filenames:
        a = 1
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

lst = os.listdir('../input/temp')
len(lst)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization,LSTM, Reshape, SimpleRNN
from keras.layers.recurrent import GRU
import os
import string
import cv2
characters = string.ascii_lowercase+"0123456789"+string.ascii_uppercase
characters
len(characters)
all_images =  os.listdir('../input/temp')
all_images[-1], len(all_images)

def preprocess_data():
    
    inp_x = np.zeros((len(all_images), 100,160,1))
    inp_y = np.zeros((len(all_images), 6,len(characters)))
    for idx, image in enumerate(all_images):
        # ----------------prepare input container
        
        
        img = cv2.imread("../input/temp/"+image, cv2.IMREAD_GRAYSCALE)
        #print(np.max(img))
        #plt.imshow(img)
        #print(img.shape)
        
        # ---------------------Scale images --------------
        
        img = img/255.0
        
        image_txt = image[0:6]
        target_oht = np.zeros((5,len(characters)))
        if(len(image_txt)<7):
            img = np.reshape(img, (100,160,1))
            inp_x[idx] = img
            
             # ------------------Define targets and code them using OneHotEncoding
            target_oht = np.zeros((6,len(characters)))
            for k, char in enumerate(image_txt):
                target_oht[k, characters.find(char)] = 1
            inp_y[idx] = target_oht
    return inp_x, inp_y
        
               
        
        
        
X, Y = preprocess_data()
from sklearn.model_selection import train_test_split
train_x ,test_x, train_y ,test_y = train_test_split(X,Y, shuffle = True, test_size = .1)
train_x.shape,train_y.shape, test_x.shape,test_y.shape
model = Sequential()
def prepare_model():
    model.add(Conv2D(128, (3,3), input_shape = (100,160,1) ,padding = 'same', activation = 'relu'))
    #model.add(Dropout(.1))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2),padding ='same'))
    model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same',))
    #model.add(Dropout(.1))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size =(3,3),padding ='same'))
    model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
    #model.add(Dropout(.1))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (3,3),padding ='same'))
    model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same'))
  
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (3,3),padding ='same'))
    model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (1,1),padding ='same'))
    model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (1,1),padding ='same'))
    print(model.output_shape)
    model.add(Reshape((model.output_shape[1], model.output_shape[2]*model.output_shape[3])))
    print(model.output_shape)
    #model.add(LSTM(50, batch_input_shape=(2,5,36),stateful = True, return_sequences = True))
    model.add(SimpleRNN(50, return_sequences = True,activation = 'relu'))
    model.add(SimpleRNN(50, return_sequences = True,activation = 'relu'))
    model.add(SimpleRNN(50, return_sequences = True,activation = 'relu'))
  


    model.add(Dense(62, activation = "softmax"))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
   
    return model
    
model =prepare_model()
model.summary()
#model = create_model()
hist = model.fit(train_x, train_y, batch_size=32, epochs=50,verbose=1, validation_data=(test_x, test_y))
score= model.evaluate(test_x,test_y,verbose=1)
print('Test Loss and accuracy:', score)
# Define function to predict captcha
def predict(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    #print("img",img.shape)
    if(img is not None):
        img = cv2.resize(img, (160,100))
    if img is not None:
        img = img / 255.0
    else:
        print("Not detected");
    res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis]))
    ans = np.reshape(res, (6, 62))
    l_ind = []
    probs = []
    for a in ans:
        l_ind.append(np.argmax(a))
        #probs.append(np.max(a))

    capt = ''
    for l in l_ind:
        capt += characters[l]
    return capt#, sum(probs) / 5
for i in all_images[:20]:

    #model.evaluate(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]])
    print(i[:6],"predicted : ",predict('../input/temp/'+i))
    
# --------------------Saving Model --------------------------------------


model.save_weights('../model_weights.h5')

# Save the model architecture
with open('../model_architecture.json', 'w') as f:
    f.write(model.to_json())
# ------------------------------Loading ---------------------------------

from keras.models import model_from_json

# Model reconstruction from JSON file
with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('model_weights.h5')
