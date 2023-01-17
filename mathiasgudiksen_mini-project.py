import numpy as np

import pandas as pd

import keras

import cv2

from matplotlib import pyplot as plt

import os



sample_path = r"/kaggle/input/landmark-recognition-2020/sample_submission.csv"

train_path = r"/kaggle/input/landmark-recognition-2020/train.csv"

base_path = r"/kaggle/input/landmark-recognition-2020/train"

test_path = r"/kaggle/input/landmark-recognition-2020/test"

### Numerical exploration of the dataset.

df = pd.read_csv(train_path)

df_test = pd.read_csv(sample_path)



samples = 20000



df = df.loc[:samples,:]



num_classes = len(df["landmark_id"].unique())

num_data = len(df)



print(":::Training set:::")

print("Number of classes: ", num_classes)

print("Number of datapoints: ", num_data)
### Histogram of the distribution of labels.

fig=plt.figure(figsize=(12, 8))

n = plt.hist(df["landmark_id"],bins=df["landmark_id"].unique())

plt.ylim(top=250)

plt.title("Distribution of labels")
### Analysis of training set:

freq_info = n[0]   # Get the frequency information for all classes and reject the rest

sec0_5 = len(freq_info[freq_info<=5])

sec5_10 = len(freq_info[freq_info<=10]) - sec0_5

print("Classes with <5 instances:", sec0_5)

print("Classes with 5> and <10 instances:", sec5_10)





from sklearn.preprocessing import LabelEncoder

lencoder = LabelEncoder()

lencoder.fit(df["landmark_id"])



def encode_label(lbl):

    return lencoder.transform(lbl)

    

def decode_label(lbl):

    return lencoder.inverse_transform(lbl)
### Visualize random images from the dataset



def get_image_from_number(num):

    fname, label = df.loc[num,:]

    fname = fname + ".jpg"

    f1 = fname[0]

    f2 = fname[1]

    f3 = fname[2]

    path = os.path.join(f1,f2,f3,fname)

    im = cv2.imread(os.path.join(base_path,path))

    return im, label





fig=plt.figure(figsize=(16, 8))



columns = 4

rows = 2

for i in range(1, columns*rows +1):

    n = np.random.randint(num_data)

    img, lbl = get_image_from_number(n)

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

    plt.title("Label = " + str(lbl))

plt.show()

### Recreation of the VGG19 architecture for basic. Allowing adjustments like BatchNormalization and so on.

from keras.layers import *

from keras import Sequential



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

model.add(Dense(num_classes, activation="softmax"))

print(model.summary())



### Compile the network, with the new optimizer I just learned about ;)

opt = keras.optimizers.RMSprop(learning_rate = 0.0001, momentum = 0.09)

model.compile(optimizer=opt,

             loss="sparse_categorical_crossentropy",

             metrics=["accuracy"])

### Function used for processing the data, fitted into a data generator.

def get_image_from_number(num, df):

    fname, label = df.iloc[num,:]

    fname = fname + ".jpg"

    f1 = fname[0]

    f2 = fname[1]

    f3 = fname[2]

    path = os.path.join(f1,f2,f3,fname)

    im = cv2.imread(os.path.join(base_path,path))

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

batch_size = 50

epoch_shuffle = True

weight_classes = True

epochs = 10



# Split train data up into 80% and 20% validation

train, validate = np.split(df.sample(frac=1), [int(.8*len(df))])

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
### Test on training set

batch_size = 50



errors = 0

good_preds = []

bad_preds = []



for it in range(int(np.ceil(len(validate)/batch_size))):



    X_train, y_train = get_batch(validate, it*batch_size, batch_size)



    result = model.predict(X_train)

    cla = np.argmax(result, axis=1)

    for idx, res in enumerate(result):

        print("Class:", cla[idx], "- Confidence:", np.round(res[cla[idx]],2), "- GT:", y_train[idx])

        if cla[idx] != y_train[idx]:

            errors = errors + 1

            bad_preds.append([batch_size*it + idx, cla[idx], res[cla[idx]]])

        else:

            good_preds.append([batch_size*it + idx, cla[idx], res[cla[idx]]])



print("Errors: ", errors, "Acc:", np.round(100*(len(validate)-errors)/len(validate),2))



        

### Plot 4 best predictions



fig=plt.figure(figsize=(16, 8))



good_preds = np.array(good_preds)

good_preds = np.array(sorted(good_preds, key = lambda x: x[2], reverse=True))

print(good_preds.shape)



columns = 4

rows = 2

for i in range(1, columns*rows +1):

    n = int(good_preds[i,0])

    print(n)

    img, lbl = get_image_from_number(n, validate)

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

    lbl2 = np.array(int(good_preds[i,1])).reshape(1,1)

    plt.title("Label = " + str(lbl) + " Classified:" + str(decode_label(lbl2)) + " Confidence:" + str(np.round(good_preds[i,2],2)))

plt.show()



### Plot 4 worst predictions



fig=plt.figure(figsize=(16, 8))



bad_preds = np.array(bad_preds)

bad_preds = np.array(sorted(bad_preds, key = lambda x: x[2], reverse=True))



columns = 4

rows = 2

for i in range(1, columns*rows +1):

    n = int(bad_preds[i,0])

    print(n)

    img, lbl = get_image_from_number(n, validate)

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

    lbl2 = np.array(int(bad_preds[i,1])).reshape(1,1)

    plt.title("Label = " + str(lbl) + " Classified:" + str(decode_label(lbl2)) + " Confidence:" + str(np.round(bad_preds[i,2],2)))

plt.show()



temp = []

for cla, amt in enumerate(freq_info):

    temp.append([cla, amt])



temp2 = np.array(sorted(temp, key = lambda x: x[1], reverse=True))



print("Top 5 most frequent labels.")

for t in range(5):

    lbl = np.array(int(temp2[t,0])).reshape(1,)

    print("Class:", decode_label(lbl)[0], "has", int(temp2[t,1]), "instances.")

    



errors = np.array([1668, 1983, 2037, 1469, 1468, 2586, 226])

encoded_errors = encode_label(errors)

wrong_preds = np.array([1189, 2434, 1346, 1924, 1127, 309])

wrong_preds_encoded = encode_label(wrong_preds)

print("\nClasses with wrong predicitions.")

for idx, lbl in enumerate(encoded_errors):



    print("Label:", errors[idx], "has", int(temp[lbl-1][1]), "instances.")

    

print("\nClasses with high tendency to be predicted")

for idx, lbl in enumerate(wrong_preds_encoded):

    print("Label:", wrong_preds[idx], "has", int(temp[lbl-1][1]), "instances.")
### Submission

def get_image_from_number(num):

    fname = df_test.loc[num,"id"]

    fname = fname + ".jpg"

    f1 = fname[0]

    f2 = fname[1]

    f3 = fname[2]

    path = os.path.join(f1,f2,f3,fname)

    im = cv2.imread(os.path.join(test_path,path))

    return im



def get_max_class(preds):

    p = preds

    confidence = np.max(p)

    cla = np.argmax(p)

    label = decode_label(cla.reshape(1,1))[0]

    

    return label, np.round(confidence,2)

    

test_samples = len(df_test)

test_df = df_test.copy()

for sample in range(test_samples):

    img = get_image_from_number(sample)

    img = image_reshape(img, (224, 224)).reshape(1, 224, 224, 3)

    

    result = model.predict(img)

    

    label, conf = get_max_class(result)

    test_df.at[sample, 'landmark'] = str(label) + " " + str(conf)

    print(label, conf)



test_df.to_csv("submission.csv", index = False, header = True)


