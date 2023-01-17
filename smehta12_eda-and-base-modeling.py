import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import cv2
import gc
import random
root_dir="../input"#r"C:\my_projects\protein_identification"
train_dir=os.path.join(root_dir, "train")
test_dir=os.path.join(root_dir, "test")
train_csv_path=os.path.join(root_dir,"train.csv")
ORIG_IMAGE_SIZE=512
# read the training csv
train_csv = pd.read_csv(train_csv_path)
print(train_csv.shape)
train_csv.head()
class_names = {
    0:"Nucleoplasm",
    1:"Nuclear membrane", 
    2:"Nucleoli", 
    3:"Nucleoli fibrillar center", 
    4:"Nuclear speckles", 
    5:"Nuclear bodies", 
    6:"Endoplasmic reticulum", 
    7:"Golgi apparatus", 
    8:"Peroxisomes", 
    9:"Endosomes", 
    10:"Lysosomes", 
    11:"Intermediate filaments", 
    12:"Actin filaments", 
    13:"Focal adhesion sites", 
    14:"Microtubules", 
    15:"Microtubule ends", 
    16:"Cytokinetic bridge", 
    17:"Mitotic spindle", 
    18:"Microtubule organizing center", 
    19:"Centrosome", 
    20:"Lipid droplets", 
    21:"Plasma membrane", 
    22:"Cell junctions", 
    23:"Mitochondria", 
    24:"Aggresome", 
    25:"Cytosol", 
    26:"Cytoplasmic bodies", 
    27:"Rods & rings" 
}
# split the targets in train csv

def split_classes(row):
    for cls_num in row["Target"].split():
        train_csv.loc[row.name, class_names[int(cls_num)]]=1

for cls_num, cls_name in class_names.items():
    train_csv[cls_name]=0

# train_csv["splitted"] = train_csv["Target"].apply(lambda x: i+1 for i in x.split())
train_csv.apply(split_classes, axis=1)
train_csv.head()
counts=train_csv[list(class_names.values())].sum().sort_values(ascending=False)
counts = counts.to_frame("cnts")
plt.figure(figsize=(20, 10))
sns.barplot(counts.index, counts.cnts)
plt.xticks(rotation=70)
plt.show()
counts
corr=train_csv[list(class_names.values())].corr()
plt.figure(figsize=(7, 7))
sns.heatmap(corr, linewidths=0.05, linecolor='b', square=True)
plt.show()
#for labels in train_csv.values():

co_occur_map = pd.DataFrame(index=list(class_names.values()), columns=list(class_names.values()))
co_occur_map=co_occur_map.fillna(0)

def find_create_map(row):
    classes = row.split()
    for r in classes:
        for c in classes:
            co_occur_map.loc[class_names[int(r)], class_names[int(c)]] += 1

train_csv["Target"].apply(find_create_map)
co_occur_map.head()
# making 0s to same classes so the map can be seen well
for cls in class_names.values():
    co_occur_map.loc[cls, cls]=0

plt.figure(figsize=(7, 7))
sns.heatmap(co_occur_map, cmap="jet", linewidths=0.05, linecolor='b', square=True)
plt.show()
# If we remove the the Nucleoplasm then see which classes has more co-occurence.
co_map = co_occur_map.drop(index=["Nucleoplasm"])
co_map = co_map.drop(columns=["Nucleoplasm"])

plt.figure(figsize=(7, 7))
sns.heatmap(co_map, cmap="jet", linewidths=0.05, linecolor='b', square=True)
plt.show()
# show the random protein images

num_random = 4

colors = [ "_green.png", "_blue.png", "_red.png", "_yellow.png"]
cmaps =["Greens", "Blues", "Reds", "YlOrBr"]

rnd_imgs = random.sample(train_csv["Id"].values.tolist(), num_random)

for img in rnd_imgs:
    plt.figure(figsize=(20, 10))
    
    green_img_name = []
    classes = train_csv[train_csv["Id"]==img]["Target"].values.tolist()
    for cls in  classes[0].split():
        green_img_name.append(class_names[int(cls)])
    
    green_img_name = ",".join(green_img_name)
      
    
    for j, color in enumerate(colors):
        plt.subplot(1, 4, j+1)
        if j == 0:
            plt.title(green_img_name)
        if j == 1:
            plt.title("Nucleus")
        if j == 2:
            plt.title("Microtubules")
        if j == 3:
            plt.title("ER")
        plt.grid(False)
        image = cv2.imread(os.path.join(train_dir,img+color), 0)
        image.astype(float)
        plt.imshow(image, cmap=cmaps[j])
plt.show()
def display_combined_rgb_img(img):
    print(os.path.join(train_dir,img))
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    all_images = np.empty((512,512,4))
    for i, color in enumerate(['red', 'green', 'yellow', 'blue']):
        all_images[:,:,i] = plt.imread(os.path.join(train_dir,img+"_{}.png").format(color))

    # define transformation matrix
    # note that yellow is made usign red and green
    # but you can tune this color conversion yourself
    T = np.array([[1,0,1,0],[0,1,1,0],[0,0,0,1]])

    # convert to rgb
    rgb_image = np.matmul(all_images.reshape(-1, 4), np.transpose(T))
    rgb_image = rgb_image.reshape(all_images.shape[0], all_images.shape[0], 3)
    rgb_image = np.clip(rgb_image, 0, 1)

    # plot
    ax.imshow(rgb_image)
    ax.set(xticks=[], yticks=[])

    return rgb_image
img=random.sample(train_csv.Id.values.tolist(), 1)[0]
display_combined_rgb_img(img)
plt.show()
T = np.array([[1,0,1,0],[0,1,1,0],[0,0,0,1]])
np.transpose(T)
# let's see random image of Nucleoplasm and Cytosol to understand the co-occurence between those classes
fig_size=(1,1)

# get first image where there's only Nucleoplasm
nue=train_csv[train_csv.Target.isin(['0'])]
nue=nue.reset_index(drop=True).iloc[0]["Id"]
display_combined_rgb_img(nue)
plt.title("Nucleoplasm")
plt.figure(figsize=fig_size)
plt.show()

# get first image where there's only Cytosol
cyt=train_csv[train_csv.Target.isin(['25'])]
cyt=cyt.reset_index(drop=True).iloc[0]["Id"]
plt.figure(figsize=fig_size)
display_combined_rgb_img(cyt)
plt.title("Cytosol")
plt.show()

# get first image where there's both Nucleoplasm and Cytosol together
nue_cyt=train_csv.query('Nucleoplasm==1 & Cytosol==1')
nue_cyt=nue_cyt.reset_index(drop=True).iloc[1]["Id"]
plt.figure(figsize=fig_size)
display_combined_rgb_img(nue_cyt)
plt.title("Nucleoplasm & Cytosol")
plt.show()
img=random.sample(train_csv.Id.values.tolist(), 1)[0]
rgb_image=display_combined_rgb_img(img)
#print(rgb_image.shape)
rgb_image=np.uint16(rgb_image)
gray_img=cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) # converting due to thresholding should be applied on gray.


plt.title("Original Image")
plt.imshow(gray_img, cmap="gray")
plt.show()
img=random.sample(train_csv.Id.values.tolist(), 1)[0]

poi = cv2.imread(os.path.join(train_dir,img+"_green.png"), 0)
nuc = cv2.imread(os.path.join(train_dir,img+"_blue.png"), 0)
mt = cv2.imread(os.path.join(train_dir,img+"_yellow.png"), 0)
er = cv2.imread(os.path.join(train_dir,img+"_red.png"), 0)
composit = cv2.add(poi,nuc, mt, er)
#composit = cv2.resize(composit, (256,256))

plt.figure()
plt.grid(False)
plt.title("original")
plt.imshow(composit, cmap="gray")
plt.show()


ret,thresh1 = cv2.threshold(composit,0,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(composit,0,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(composit,0,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(composit,0,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(composit,0,255,cv2.THRESH_TOZERO_INV)
ret,thresh6 = cv2.threshold(composit,0,255,cv2.THRESH_OTSU)

plt.figure(figsize=(10,10))
titles = ['BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV', 'THRESH_OTSU']
images = [thresh1, thresh2, thresh3, thresh4, thresh5, thresh6]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
nuc = cv2.imread(os.path.join(train_dir,img+"_green.png"), 0)
#nuc = cv2.resize(nuc, (256,256))
plt.figure(figsize=(15,5))

plt.subplot(131)
plt.grid(False)
plt.title("Protein of Interest staining")
plt.imshow(nuc, cmap='gray')

t, thresh = cv2.threshold(nuc, 0,255,cv2.THRESH_OTSU)

plt.subplot(132)
plt.grid(False)
plt.title("OTSU thresholding\n of POI staining")
plt.imshow(thresh, cmap="gray")

kernel = np.ones((6,6),np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

plt.subplot(133)
plt.grid(False)
plt.title("OTSU thresholding\nafter closing operation")
plt.imshow(closing, cmap="gray")

im, contours,hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

t=cv2.drawContours(composit, contours, -1, (0,0,0), 2)

plt.figure(figsize=(10,10))
plt.grid(False)
plt.title("POI contours drawn in composit image")
plt.imshow(t)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# Load the green images in train and test

gc.collect()

MAX_IMG_FOR_MODELING = 7000 # choosing only this much images to avoid memory error 
shuffle(train_csv)
subset_data=train_csv[:MAX_IMG_FOR_MODELING]

train_imgs, test_valid_imgs, train_labels, test_valid_labels = train_test_split(subset_data["Id"], 
                                                                    subset_data[list(class_names.values())], 
                                                                    test_size=0.4)

valid_imgs, test_imgs, valid_labels, test_labels = train_test_split(test_valid_imgs, 
                                                                    test_valid_labels, 
                                                                    test_size=0.5)

train_labels = train_labels.values
valid_labels = valid_labels.values
test_labels = test_labels.values

print(train_imgs.shape)
print(valid_imgs.shape)
print(test_imgs.shape)

def load_images(img_array):
    images = []
    for i in img_array:
        image = cv2.imread(os.path.join(train_dir, i+"_green.png"))
        image.resize(ORIG_IMAGE_SIZE, ORIG_IMAGE_SIZE, 1)
        images.append(image.astype(np.float32))

    images=np.array(images)
    return images

train_images = load_images(train_imgs)
print(train_images.shape)
valid_images = load_images(valid_imgs)
print(valid_images.shape)
test_images = load_images(test_imgs)
print(test_images.shape)
batch_size=16
patch_size=3 #5
depth=1
num_hidden=64
num_channels=1 # grayscale
image_size=ORIG_IMAGE_SIZE
num_labels=len(list(class_names.values()))
num_steps = 20 #50 #100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D , BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras import regularizers
data_gen = ImageDataGenerator()
data_gen.fit(train_images)
l2_rate = 0.5
model = Sequential()
model.add(Conv2D(32, (patch_size, patch_size), activation='relu', input_shape=(ORIG_IMAGE_SIZE, ORIG_IMAGE_SIZE, 1), 
                 strides=(1,1), padding="same", kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
# model.add(Conv2D(64, (patch_size, patch_size), activation='relu', strides=(1,1), padding="same", kernel_regularizer=regularizers.l2(l2_rate)))
model.add(Conv2D(64, (patch_size, patch_size), activation='relu', strides=(1,1), padding="same", kernel_regularizer=regularizers.l2(l2_rate)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (patch_size, patch_size), activation='relu', strides=(1,1), padding="same", kernel_regularizer=regularizers.l2(l2_rate)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (patch_size, patch_size), activation='relu', strides=(1,1), padding="same", kernel_regularizer=regularizers.l2(l2_rate)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(512, (patch_size, patch_size), activation='relu', strides=(1,1), padding="same", kernel_regularizer=regularizers.l2(l2_rate)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(1024, (patch_size, patch_size), activation='relu', strides=(1,1), padding="same"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
model.add(Dense(len(list(class_names.values())), activation="sigmoid"))

opt = Adam(lr=0.0001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

model.fit_generator(data_gen.flow(train_images, train_labels, batch_size=batch_size), 
                    validation_data=(valid_images, valid_labels),
                    steps_per_epoch=len(train_images) / batch_size, epochs=num_steps)
model.summary()
# Train vs Validation accuracy and loss

def loss_over_epochs(model, add_valid=True):
    hist=model.history.history
    plt.plot(list(range(num_steps)), hist['loss'], color="blue", label="train")
    if add_valid:
        plt.plot(list(range(num_steps)), hist['val_loss'], color="orange", label="valid")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.title("Losses over the Epochs")

# accuracy vs epochs
def acc_over_epochs(model, add_valid=True):
    hist=model.history.history
    plt.plot(list(range(num_steps)), hist["acc"], color="blue", label="train")
    if add_valid:
        plt.plot(list(range(num_steps)), hist["val_acc"], color="orange", label="valid")
    plt.xlabel("Epochs")
    plt.ylabel("accuracy")
    plt.title("Accuracy over the Epochs")
loss_over_epochs(model, add_valid=True)
plt.show()
acc_over_epochs(model, add_valid=True)
def save_model(model):
    model_dir=os.path.join("models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model.save(os.path.join(model_dir, "{}_green_img_cnn_reg_stride1.model".format(MAX_IMG_FOR_MODELING)))
score = model.evaluate(test_images, test_labels, batch_size=batch_size)
print(score)
# Getting image from the test data

rand_img = test_imgs.reset_index(drop=True).sample(1)
img_idx = rand_img.index.values.tolist()[0]
img_class = test_labels[img_idx]
classes=[]
for k,v in zip(img_class, list(class_names.values())):
    if k:
        classes.append(v)
print(classes)

image=test_images[img_idx]
image = image.reshape(1, 512, 512, 1)

proba = model.predict(image)
proba
pred_cls=[]
for k,v in zip(proba[0], list(class_names.values())):
    if k:
        pred_cls.append(v)
print(pred_cls)
# Most pre-trained models require 3 Channel images. So need to convert grayscale green images to RGB

# def convert_gray_to_rgb(img_set):
#     rgb_imgs = []
    
#     for gray_img in img_set:
#         rgb_imgs.append(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB))
    
#     return np.array(rgb_imgs)

# train_rgbs = convert_gray_to_rgb(train_images)
# valid_rgbs = convert_gray_to_rgb(valid_images)
# test_rgbs = convert_gray_to_rgb(test_images)

# print(train_rgbs.shape)
# print(valid_rgbs.shape)
# print(test_rgbs.shape)
# from keras.applications.vgg16 import VGG16
# from keras.models import Model
# # from keras.models import load_model
# # vgg_model = load_model("https://www.kaggle.com/jaccojurg/vgg16-weights-tf/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
# vgg_model = VGG16(include_top=False, input_shape=(512, 512, 3))
# vgg_model.summary()
# x = Flatten()(vgg_model.output)
# x = Dense(4096, activation='relu')(x)
# x = Dense(4096, activation='relu')(x)
# predictions = Dense(len(class_names), activation='sigmoid')(x)
# new_vgg = Model(inputs = vgg_model.input, outputs = predictions)
# new_vgg.summary()
# # Make the layer weights constant
# for layer in new_vgg.layers:
#     layer.trainable = False
# opt = Adam(lr=0.0001)
# new_vgg.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# new_vgg.fit_generator(data_gen.flow(train_rgbs, train_labels, batch_size=batch_size),
#                     validation_data=(valid_rgbs, valid_labels),
#                     steps_per_epoch=len(train_rgbs) / batch_size, epochs=num_steps)
# new_vgg.save("vgg16_200img.h5")
# loss_over_epochs(new_vgg)
# plt.show()
# acc_over_epochs(new_vgg)
# plt.show()