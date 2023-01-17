import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import cv2
import keras
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.layers import *
from keras import Sequential

data = pd.read_csv('/kaggle/input/landmark-recognition-2020/train.csv')

images = data.shape[0]
print("Number of images:", images)
classes = len(data['landmark_id'].unique())
print("Number of classes:", classes)
print("Average number of images per class:",round(images/classes,2))
plt.figure(figsize=(20,6))
plt.hist(data.landmark_id, bins=1000);
plt.title('Images per class', fontsize=16)
plt.xlabel('Class number')
plt.ylabel('Number of images')
plt.show()
data5 = (data['landmark_id'].value_counts() <= 5).sum()
data10 = (data['landmark_id'].value_counts() <= 10).sum()
print("Number of classes with less than 5 training samples:", data5)
print("Number of classes with between 5 and 10 training samples:", data10-data5)
path='/kaggle/input/landmark-recognition-2020/train/'


print("4 sample images from random classes:")
fig=plt.figure(figsize=(16, 16))
for i in range(1,5):
    a = random.choices(os.listdir(path), k=3)
    folder = path+a[0]+'/'+a[1]+'/'+a[2]
    random_img = random.choice(os.listdir(folder))
    img = np.array(Image.open(folder+'/'+random_img))
    fig.add_subplot(1, 4, i)
    plt.imshow(img)
    plt.axis('off')

plt.show()


samples = 20000

data = data.loc[:samples,:]
classes = len(data['landmark_id'].unique())

lencoder = LabelEncoder()
lencoder.fit(data["landmark_id"])

model = Sequential()
model.add(Input(shape=(224,224,3)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(64, kernel_size = (3,3), padding = "same"))
model.add(MaxPooling2D())
model.add(Conv2D(128, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(128, kernel_size = (3,3), padding = "same"))
model.add(MaxPooling2D())
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(MaxPooling2D())
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(MaxPooling2D())
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(MaxPooling2D())
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(4096, activation = "relu"))
model.add(Dense(4096, activation = "relu"))
model.add(Dense(classes, activation="softmax"))
print(model.summary())
opt = keras.optimizers.Adagrad(learning_rate = 0.001, initial_accumulator_value=0.01, epsilon=1e-07)
model.compile(optimizer=opt,
             loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])
def encode_label(lbl):
    return lencoder.transform(lbl)
    
def decode_label(lbl):
    return lencoder.inverse_transform(lbl)

def get_image_from_number(num, data):
    fname, label = data.iloc[num,:]
    fname = fname + ".jpg"
    f1 = fname[0]
    f2 = fname[1]
    f3 = fname[2]
    path = os.path.join(f1,f2,f3,fname)
    im = cv2.imread(os.path.join(r"/kaggle/input/landmark-recognition-2020/train",path))
    return im, label

def image_reshape(im, target_size):
    return cv2.resize(im, target_size)
    
def get_batch(dataframe,start, batch_size):
    image_array = []
    label_array = []
    
    end_img = start+batch_size
    if end_img > len(dataframe):
        end_img = len(dataframe)

    for idx in range(start, end_img):
        n = idx
        im, label = get_image_from_number(n, dataframe)
        im = image_reshape(im, (224, 224)) / 255.0
        image_array.append(im)
        label_array.append(label)
        
    label_array = encode_label(label_array)
    return np.array(image_array), np.array(label_array)
batch_size = 10
epoch_shuffle = True
weight_classes = True
epochs = 10

train, validate = np.split(data.sample(frac=1), [int(.8*len(data))])
print("Training on:", len(train), "samples")
print("Validation on:", len(validate), "samples")
    
for e in range(epochs):
    print("Epoch: ", str(e+1) + "/" + str(epochs))
    if epoch_shuffle:
        train = train.sample(frac = 1)
    for it in range(int(np.ceil(len(train)/batch_size))):

        X_train, y_train = get_batch(train, it*batch_size, batch_size)

        model.train_on_batch(X_train, y_train)
        

model.save("Model.h5")
batch_size = 10

errors = 0
good_preds = []
bad_preds = []

for it in range(int(np.ceil(len(validate)/batch_size))):

    X_train, y_train = get_batch(validate, it*batch_size, batch_size)

    result = model.predict(X_train)
    cla = np.argmax(result, axis=1)
    for idx, res in enumerate(result):
        if cla[idx] != y_train[idx]:
            errors = errors + 1
            bad_preds.append([batch_size*it + idx, cla[idx], res[cla[idx]]])
        else:
            good_preds.append([batch_size*it + idx, cla[idx], res[cla[idx]]])

print("Total errors: ", errors, "out of", len(validate), "\nAccuracy:", np.round(100*(len(validate)-errors)/len(validate),2), "%")
n = plt.hist(data["landmark_id"],bins=data["landmark_id"].unique())
plt.close()
freq_info = n[0]

temp = []
for cla, amt in enumerate(freq_info):
    temp.append([cla, amt])

good_preds = np.array(good_preds)
good_preds = np.array(sorted(good_preds, key = lambda x: x[2], reverse=True))

print("5 images where classification went well:")
fig=plt.figure(figsize=(16, 16))
for i in range(1,6):
    n = int(good_preds[i,0])
    img, lbl = get_image_from_number(n, validate)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig.add_subplot(1, 5, i)
    plt.imshow(img)
    lbl2 = np.array(int(good_preds[i,1])).reshape(1,1)
    sample_cnt = int(temp[int(encode_label(np.array([lbl])))-1][1])
    plt.title("Label: " + str(lbl) + "\nClassified as: " + str(decode_label(lbl2)) + "\nSamples in class " + str(lbl) + ": " + str(sample_cnt))
    plt.axis('off')
plt.show()
bad_preds = np.array(bad_preds)
bad_preds = np.array(sorted(bad_preds, key = lambda x: x[2], reverse=True))

print("5 images where classification failed:")
fig=plt.figure(figsize=(16, 16))
for i in range(1,6):
    n = int(bad_preds[i,0])
    img, lbl = get_image_from_number(n, validate)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig.add_subplot(1, 5, i)
    plt.imshow(img)
    lbl2 = np.array(int(bad_preds[i,1])).reshape(1,1)
    sample_cnt = int(temp[int(encode_label(np.array([lbl])))-1][1])
    plt.title("Label: " + str(lbl) + "\nClassified as: " + str(decode_label(lbl2)) + "\nSamples in class " + str(lbl) + ": " + str(sample_cnt))
    plt.axis('off')
    
plt.show()