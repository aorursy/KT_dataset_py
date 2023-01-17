# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
#import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Data parameter
input_dir = os.path.join('..', 'input')
output_dir = os.path.join('..', 'output')

dataset_dir = os.path.join(input_dir, 'landmark-recognition-2020')
train_dir = os.path.join(dataset_dir, 'train')
train_labelmap_dir = os.path.join(dataset_dir, 'train.csv')
test_dir = os.path.join(dataset_dir, 'test')


train_df = pd.read_csv(train_labelmap_dir)
num_data = len(train_df)

print("Shape of train_data :", train_df.shape)
print('Number of classes:')
landmark = train_df.landmark_id.value_counts()
print(landmark.size)
landmark_df = pd.DataFrame({'landmark_id':landmark.index, 'frequency':landmark.values})#.head(30)

landmark_df.reset_index(inplace=True)

print(landmark_df)

print(landmark_df['frequency'].describe())

plt.hist(landmark_df['frequency'], 100, range = (0, 950), label = 'test')
plt.xlabel("Amount of images")
plt.ylabel("Occurences")

print("Amount of classes with less than 5 trainning samples:", (landmark_df['frequency'].between(0,4)).sum())
print("Amount of classes with between 5 and 10 training samples:", (landmark_df['frequency'].between(5,10)).sum())
from sklearn.preprocessing import LabelEncoder
lencoder = LabelEncoder()
lencoder.fit(train_df["landmark_id"])

def encode_label(lbl):
    return lencoder.transform(lbl)
    
def decode_label(lbl):
    return lencoder.inverse_transform(lbl)
### Visualize random images from the dataset

def get_image_from_number(num):
    fname, label = train_df.loc[num,:]
    fname = fname + ".jpg"
    f1 = fname[0]
    f2 = fname[1]
    f3 = fname[2]
    path = os.path.join(f1,f2,f3,fname)
    im = cv2.imread(os.path.join(train_dir,path))
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
train_df["filename"] = train_df.id.str[0]+"/"+train_df.id.str[1]+"/"+train_df.id.str[2]+"/"+train_df.id+".jpg"
train_df["label"] = train_df.landmark_id.astype(str)
print(train_df)
from collections import Counter

no_classes_keep = 1000

c = train_df.landmark_id.values
count = Counter(c).most_common(no_classes_keep)
print(len(count), count[-1])
keep_labels = [i[0] for i in count]
print(len(keep_labels))
train_keep = train_df[train_df.landmark_id.isin(keep_labels)]
print(len(train_keep))
train_keep = train_keep.sample(frac=1).reset_index(drop=True)
print(train_keep)
val_split = 0.2
batch_size = 50

datagen = ImageDataGenerator(validation_split=val_split)

train_datagen = datagen.flow_from_dataframe(
    train_keep, # Pandas dataframe containing the filepaths relative to directory (or absolute paths if directory is None) and classes label
    directory=train_dir + "/",
    x_col="filename",
    y_col="label",
    weight_col=None,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    subset="training",
    interpolation="nearest",
    validate_filenames=False)

val_datagen = datagen.flow_from_dataframe(
    train_keep, # Pandas dataframe containing the filepaths relative to directory (or absolute paths if directory is None) and classes label
    directory=train_dir + "/",
    x_col="filename",
    y_col="label",
    weight_col=None,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    subset="validation",
    interpolation="nearest",
    validate_filenames=False)
from keras.applications import VGG19
from keras.layers import *
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

vgg_model = VGG19(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

#model = Sequential([
#    VGG19(
#    input_shape=(256, 256, 3),
#        weights='imagenet',
#        include_top=False
#    ),
#    GlobalAveragePooling2D(),
#    Dense(no_classes_keep, activation='softmax')
#])

model = Sequential()
for layer in vgg_model.layers:
    if layer == vgg_model.layers[-21]:
        model.add(BatchNormalization())
    model.add(layer)
model.add(Flatten())
model.add(Dense(4096, activation = "relu"))
model.add(Dense(4096, activation = "relu"))
model.add(Dense(no_classes_keep, activation="softmax"))

#for layer in model.layers[19:]:
#    layer.trainable = False

model.summary()

model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics=['categorical_accuracy']
)
epochs = 5

train_steps = int(len(train_keep)*(1-val_split))//batch_size
val_steps = int(len(train_keep)*val_split)//batch_size

model_checkpoint = ModelCheckpoint("model_vgg19.h5", save_best_only=True, verbose=1)
history = model.fit(train_datagen, steps_per_epoch=train_steps, epochs=epochs,validation_data=val_datagen, validation_steps=val_steps, callbacks=[model_checkpoint])

model.save("model.h5")
predict = model.predict(val_datagen, val_steps)
good_preds = []
bad_preds = []

val_filenames = val_datagen.filenames
label_map = (val_datagen.class_indices)
#label_categories = to_categorical(np.asarray(labels)) 
cla = np.argmax(predict, axis=1)
label_map = list(map(int, label_map.keys()))
val_label = val_datagen.labels

for idx, res in enumerate(predict):
    #print("image_id: ", val_filenames[idx], ", class predict: ", label_map[cla[idx]], "class: ", label_map[val_label[idx]])
    
    if label_map[cla[idx]] != label_map[val_label[idx]]:
        bad_preds.append([val_filenames[idx], label_map[cla[idx]], label_map[val_label[idx]], res[cla[idx]]])
    else:
        good_preds.append([val_filenames[idx], label_map[cla[idx]], label_map[val_label[idx]], res[cla[idx]]])
print("wrong predictions: ", len(bad_preds), " right predictions: ", len(good_preds), " acc: ", np.round(100*(len(predict)-len(bad_preds))/len(predict),2))
### plot some of the best predictions

fig=plt.figure(figsize=(16, 8))

good_preds = np.array(good_preds)
good_preds = np.array(sorted(good_preds, key = lambda x: x[3], reverse=True))
#print(good_preds.shape)

columns = 4
rows = 1
for i in range(1, columns*rows +1):
    n = good_preds[i,0]
    #print(n)
    img = cv2.imread(os.path.join(train_dir,n))
    lbl = good_preds[i,2]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    lbl2 = good_preds[i,1]
    plt.title("Label = " + str(lbl) + " Classified:" + str(lbl2) + " Confidence:" + str(good_preds[i,3]))
plt.show()
### plot the worst predictions

fig=plt.figure(figsize=(16, 8))

bad_preds = np.array(bad_preds)
bad_preds = np.array(sorted(bad_preds, key = lambda x: x[3], reverse=True))
#print(bad_preds.shape)

columns = 4
rows = 1
for i in range(1, columns*rows +1):
    n = bad_preds[i,0]
    #print(n)
    img = cv2.imread(os.path.join(train_dir,n))
    lbl = bad_preds[i,2]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    lbl2 = bad_preds[i,1]
    plt.title("Label = " + str(lbl) + " Classified:" + str(lbl2) + " Confidence:" + str(good_preds[i,3]))
plt.show()
train_val = train_keep.landmark_id.value_counts()
train_keep_df = pd.DataFrame({'landmark_id':train_val.index, 'frequency':train_val.values})#.head(30)
train_keep_df.reset_index(inplace=True)
#print(train_keep_df)

print("Top 5 training classes with most data:")
for i in range(5):
    print("label:", train_keep_df.landmark_id[i], "has", train_keep_df.frequency[i], "instances in training set" )

train_keep_df.set_index("landmark_id", inplace = True)
print("\nTop 5 classes with the worst prediction")
    
for i in range(5):
    label = bad_preds[i, 2]
    #print(label)
    label_counts = train_keep_df.loc[int(label)]
    #print(label_counts)
    print("label:", label, "has", label_counts["frequency"], "instances in training set" )
    
    
print("\nTop 5 classes with the best prediction")
for i in range(5):
    label = good_preds[i, 2]
    #print(label)
    label_counts = train_keep_df.loc[int(label)]
    #print(label_counts)
    print("label:", label, "has", label_counts["frequency"], "instances in training set" )
sub = pd.read_csv("/kaggle/input/landmark-recognition-2020/sample_submission.csv")
sub["filename"] = sub.id.str[0]+"/"+sub.id.str[1]+"/"+sub.id.str[2]+"/"+sub.id+".jpg"
sub
best_model = load_model("model_vgg19.h5")

test_gen = ImageDataGenerator().flow_from_dataframe(
    sub,
    directory="/kaggle/input/landmark-recognition-2020/test/",
    x_col="filename",
    y_col=None,
    weight_col=None,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode=None,
    batch_size=1,
    shuffle=True,
    subset=None,
    interpolation="nearest",
    validate_filenames=False)
y_pred_one_hot = best_model.predict_generator(test_gen, verbose=1, steps=len(sub))
y_pred = np.argmax(y_pred_one_hot, axis=-1)
y_prob = np.max(y_pred_one_hot, axis=-1)
print(y_pred.shape, y_prob.shape)
y_uniq = np.unique(train_keep.landmark_id.values)

y_pred = [y_uniq[Y] for Y in y_pred]
for i in range(len(sub)):
    sub.loc[i, "landmarks"] = str(y_pred[i])+" "+str(y_prob[i])
sub = sub.drop(columns="filename")
sub.to_csv("submission.csv", index=False)
sub