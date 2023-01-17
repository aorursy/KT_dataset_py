import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from keras.layers import Conv2D, Dense, Dropout, AveragePooling2D, MaxPool2D, BatchNormalization, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

import os
print(os.listdir("../input"))

%matplotlib inline

# Any results you write to the current directory are saved as output.
train_dir = "../input/train/"
test_dir = "../input/test/"

sample_submission = pd.read_csv("../input/sample_submission.csv")
# display the train image
train_img_list = os.listdir(train_dir) 

print("No of images = "+str(len(train_img_list)))

img = plt.imread(train_dir+train_img_list[0])
plt.imshow(img)
#print(img.shape)
# display the train image
test_img_list = os.listdir(test_dir) 

print("No of images = "+str(len(test_img_list)))

img = plt.imread(test_dir+test_img_list[0])
plt.imshow(img)
#print(img.shape)
# train.csv
train_df = pd.read_csv("../input/train.csv")

print("train.csv shape = "+str(train_df.shape))

train_df.head()
# unique ids - also includes "new values" 
ids = train_df["Id"]
ids.value_counts().shape[0]
# image preprocessing 
def preprocessing(dir_name,image_list):
    print("Preprocessing "+dir_name)
    m = len(image_list)
    
    X = np.zeros((m,100,100,3))
    
    count = 0
    for img_name in image_list:
        img = image.load_img(path=dir_name+img_name,target_size=(100,100,3)) #images may have different size hence compressing into same size
        img = image.img_to_array(img)
        img = preprocess_input(img)
        X[count] = img
    
        
        if count%1000 == 0:
            print("Preprocessing "+str(count))
        count += 1
    return X
x_train = preprocessing(train_dir,train_df["Image"])
# x_test = preprocessing(test_dir,sample_submission["Image"])
def label_preprocessing(y):
    label_encoder = LabelEncoder() #to convert string labels to integer
    label_encoder.fit(y)
    labels_encoded = label_encoder.transform(y)
    # print(labels_encoded.shape)
    
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(labels_encoded.reshape(-1,1))
    one_hot_encoded = one_hot_encoder.transform(labels_encoded.reshape(-1,1))
    
    # print(one_hot_encoded.shape)
    
    y = one_hot_encoded
    
    return y, label_encoder
label_encoder = None
y_train, label_encoder = label_preprocessing(train_df["Id"])
x_train = x_train/255.0
# x_test = x_test/255.0
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(7,7),name="conv0",input_shape=(100,100,3)))
model.add(BatchNormalization(name="batch0"))
model.add(Activation(activation='relu'))

model.add(MaxPool2D(pool_size=(2,2),name="max0"))
model.add(Conv2D(filters=64,kernel_size=(3,3),name="conv1"))
model.add(Activation(activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3),name="avg0"))

model.add(Flatten())
model.add(Dense(units=1000,activation="relu",name="dense0"))
model.add(Dropout(rate=0.8))
model.add(Dense(units=y_train.shape[1],activation="softmax",name="dense1"))
model.summary()
adam = Adam()
model.compile(optimizer=adam,loss="categorical_crossentropy",metrics=['accuracy'])
history = model.fit(x_train,y_train,epochs=100, batch_size=100, verbose=1)
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
x_test = preprocessing(test_dir,sample_submission["Image"])
x_test = x_test/255.0
predictions = model.predict(np.array(x_test), verbose=1)
col = ['Image']
test_df = pd.DataFrame(sample_submission["Image"], columns=col)
test_df['Id'] = ''
for i, pred in enumerate(predictions):
    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
test_df.head(10)
test_df.to_csv("submission1.csv",index=False)
