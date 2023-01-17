import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
import gc
train_df = pd.read_csv("../input/humpback-whale-identification/train.csv")
train_df.head()
def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 224, 224, 3))
    count = 0
    
    for fig in data['Image']:
        #load images into images of size 100x100x3
        img = image.load_img("../input/humpback-whale-identification/"+dataset+"/"+fig, target_size=(224, 224, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train
def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder
y, label_encoder = prepare_labels(train_df['Id'])
y.shape
X = prepareImages(train_df, train_df.shape[0], "train")
X /= 255
model1 = Sequential()

model1.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (224, 224, 3)))
model1.add(BatchNormalization(axis = 3, name = 'bn0'))
model1.add(Activation('relu'))
model1.add(MaxPooling2D((2, 2), name='max_pool_0'))

model1.add(Conv2D(64, (5, 5), strides = (2, 2), name="conv1"))
model1.add(BatchNormalization(axis = 3, name = 'bn2'))
model1.add(Activation('relu'))
model1.add(AveragePooling2D((2, 2), name='avg_pool'))

model1.add(Conv2D(128, (3, 3), strides = (1,1), name="conv2"))
model1.add(Conv2D(128, (1, 1), strides = (1,1), name="conv3"))
model1.add(BatchNormalization(axis = 3, name = 'bn3'))
model1.add(Activation('relu'))
model1.add(MaxPooling2D((2, 2), name='max_pool_2'))

model1.add(Flatten())

model1.add(Dense(500, activation="relu", name='rl'))
model1.add(Dropout(0.8))

model1.add(Dense(y.shape[1], activation='softmax', name='sm'))

model1.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model1.summary()
history1 = model1.fit(X, y, epochs=50, batch_size=128, validation_split=0.30)
gc.collect()
plt.plot(history1.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
# summarize history for accuracy
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
test = os.listdir("../input/humpback-whale-identification/test/")
print(len(test))

col = ['Image']
test_df = pd.DataFrame(test, columns=col)
test_df['Id'] = ''
X_test = prepareImages(test_df, test_df.shape[0], "test")
X_test /= 255
predictions = model1.predict(np.array(X_test), verbose=1)
for i, pred in enumerate(predictions):
    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
test_df.head(10)
test_df.to_csv('submission.csv', index=False)
y_test, label_encoder = prepare_labels(test_df['Id'])
y_test.shape
preds = model.evaluate(X_test, y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
import base64
import pandas as pd
from IPython.display import HTML

def create_download_link( df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

df = pd.DataFrame(data = [[1,2],[3,4]], columns=['Col 1', 'Col 2'])
create_download_link(test_df)
model = Sequential() 
  
# 1st Convolutional Layer 
model.add(Conv2D(filters = 96, input_shape = (224, 224, 3),  
            kernel_size = (11, 11), strides = (4, 4),  
            padding = 'valid')) 
model.add(Activation('relu')) 
# Max-Pooling  
model.add(MaxPooling2D(pool_size = (2, 2), 
            strides = (2, 2), padding = 'valid')) 
# Batch Normalisation 
model.add(BatchNormalization()) 
  
# 2nd Convolutional Layer 
model.add(Conv2D(filters = 256, kernel_size = (11, 11),  
            strides = (1, 1), padding = 'valid')) 
model.add(Activation('relu')) 
# Max-Pooling 
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2),  
            padding = 'valid')) 
# Batch Normalisation 
model.add(BatchNormalization()) 
  
# 3rd Convolutional Layer 
model.add(Conv2D(filters = 384, kernel_size = (3, 3),  
            strides = (1, 1), padding = 'valid')) 
model.add(Activation('relu')) 
# Batch Normalisation 
model.add(BatchNormalization()) 
  
# 4th Convolutional Layer 
model.add(Conv2D(filters = 384, kernel_size = (3, 3),  
            strides = (1, 1), padding = 'valid')) 
model.add(Activation('relu')) 
# Batch Normalisation 
model.add(BatchNormalization()) 
  
# 5th Convolutional Layer 
model.add(Conv2D(filters = 256, kernel_size = (3, 3),  
            strides = (1, 1), padding = 'valid')) 
model.add(Activation('relu')) 
# Max-Pooling 
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2),  
            padding = 'valid')) 
# Batch Normalisation 
model.add(BatchNormalization()) 
  
# Flattening 
model.add(Flatten()) 
  
# 1st Dense Layer 
model.add(Dense(4096, input_shape = (224*224*3, ))) 
model.add(Activation('relu')) 
# Add Dropout to prevent overfitting 
model.add(Dropout(0.4)) 
# Batch Normalisation 
model.add(BatchNormalization()) 
  
# 2nd Dense Layer 
model.add(Dense(4096)) 
model.add(Activation('relu')) 
# Add Dropout 
model.add(Dropout(0.4)) 
# Batch Normalisation 
model.add(BatchNormalization()) 
  
# Output Softmax Layer 
model.add(Dense(5005)) 
model.add(Activation('softmax')) 

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()
history1 = model.fit(X, y, epochs=50, batch_size=128, validation_split=0.30)